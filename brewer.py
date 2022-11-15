# adapted from cms-btv-pog/BTVNanoCommissioning

import argparse
import os
import sys
import yaml
import gzip
import logging
import pickle
import uproot
import socket
import awkward as ak
from itertools import islice
from typing import List, Dist
from coffea import processor


from dask.distributed import Worker, WorkerPlugin
from coffea.nanoevents import NanoAODSchema, BaseSchema
from qawa.process.zz2l2nu_vbs import zzinc_processor
from dataclasses import dataclass


logging.basicConfig(
    filename="log-brew-dask.txt",
    level=logging.INFO
)

class WorkItem:
    dataset: str
    filename: str
    treename: str
    entrystart: int
    entrystop: int
    fileuuid: str
    usermeta: Optional[Dict] = field(default=None, compare=False)

    def __len__(self) -> int:
        return self.entrystop - self.entrystart

@dataclass
class DaskExecutor(processor.ExecutorBase):
    client: Optional["dask.distributed.Client"] = None
    treereduction: int = 20
    priority: int = 0
    retries: int = 3
    heavy_input: Optional[bytes] = None
    use_dataframes: bool = False
    worker_affinity: bool = False
    
    def __call__(
        self,
        items: Iterable,
        function: Callable,
        accumulator: Accumulatable,
    ):
        if len(items) == 0:
            return accumulator
        
        from dask.distributed import Client
        from distributed.scheduler import KilledWorker
        
        if self.client is None:
            self.client = Client(threads_per_worker=1)
        
        work = []
        key_to_item = {}
        
        work = self.client.map(
                function,
                items,
                pure=(self.heavy_input is not None),
                priority=self.priority,
                retries=self.retries,
        )
        key_to_item.update({future.key: item for future, item in zip(work, items)})

        while len(work) > 1:
            work = self.client.map(
                reducer,
                [
                    work[i : i + self.treereduction]
                    for i in range(0, len(work), self.treereduction)
                ],
                pure=True,
                priority=self.priority,
                retries=self.retries,
            )
            key_to_item.update({future.key: "(output reducer)" for future in work})
        work = work[0]

        try:
            if self.status:
                from distributed import process
                process(work, multi=True, notebook=False)
            return (accumulate([work.result()]), 0)
        except KilledWorker as ex:
            baditem = key_to_item[ex.task]
            raise RuntimeError(
                f"work item {baditem} caused a KilledWorker"
            )





class coffea_sumw(processor.ProcessorABC):
    def __init__(self):
        super().__init__()

    def process(self, event: processor.LazyDataFrame):
        dataset_name = event.metadata['dataset']
        is_data = event.metadata.get("is_data")

        sumw = 1.0
        if is_data:
            sumw = -1.0
        else:
            sumw = ak.sum(event.genEventSumw)

        return {dataset_name: sumw}

    def postprocess(self, accumulator):
        return accumulator
        
        
def validate(file):
    try:
        fin = uproot.open(file)
        return fin["Events"].num_entries
    except Exception:
        return
    
    
def check_port(port):
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("0.0.0.0", port))
        available = True
    except Exception:
        available = False
    sock.close()
    return available

def sample_split(data, SIZE=10000):
    it = iter(data)
    for _ in range(0, len(data), SIZE):
        yield {k:data[k] for k in islice(it, SIZE)}
                
class WheelInstaller(WorkerPlugin):
    def __init__(self, logging, dependencies: List[str]):
        self._logger = logging
        self._package = " ".join(f"'{dep}'" for dep in dependencies)
        

    def setup(self, worker: Worker):
        self._logger.info(" ... installing dependencies ... ")
        workdr = worker.local_directory
        output = os.popen(f"python -m pip install --upgrade {workdr}/{self._package}").read()
        os.system(f"python -m pip install --upgrade {workdr}/{self._package}")
        os.system(f"python -m pip install --upgrade {workdr}/{self._package}")
        self._logger.info(output)
        
def fix_uproot():
    import uproot
    uproot.open.defaults["xrootd_handler"] = uproot.source.xrootd.XRootDSource

qawa_wheel = "Qawa-0.0.2-py3-none-any.whl"
dependency_installer = WheelInstaller(logging, [qawa_wheel])

def main():
    parser = argparse.ArgumentParser("")
    parser.add_argument('-jobs' , '--jobs', type=int, default=50   , help="")
    parser.add_argument('-era'  , '--era' , type=str, default="2018", help="")
    parser.add_argument(
        "--executor",
        choices=[
            "dask/casa",
            "dask/condor",
            "dask/lxplus",
            "iterative",
            "futures", 
        ],
        default="dask/lxplus",
    )
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Do not process, just check all files are accessible",
    )
    parser.add_argument(
        "--split",
        type=int,
        default=1000,
        help="split the process into n dask instances",
    )
    options = parser.parse_args()
    
    
    # loading datasets
    samples = {}
    with open('./data/datasetUL2018.yaml') as s_file:
        samples = yaml.full_load(s_file)
        
    if options.executor == "dask/casa":
        for key in samples.keys():
            samples[key]['files'] = [
                path.replace(
                    "xrootd-cms.infn.it", "xcache"
                ).replace(
                    "eoscms.cern.ch", "xcache"
                ).replace(
                    "cms-xrd-global.cern.ch", "xcache"
                )
                for path in samples[key]['files']
            ]
    if options.executor == "dask/lxplus":
        for key in samples.keys():
            samples[key]['files'] = list(
                    filter(lambda fn: fn.find('cms-xrd-global.cern.ch') < 0, samples[key]['files'])
            )
    for proc, cont in samples.items():
        _name = proc[0:25] + "..."
        print(_name, f" --> #files = {len(cont['files'])}")        
        
    # Scan if files can be opened
    if options.validate:
        exit(0)
        
    processor_instance = zzinc_processor(era='2018')
    
    if "dask" in options.executor:
        from dask_jobqueue import HTCondorCluster
        from distributed import Client
        from dask.distributed import performance_report
        
        
        njobs = 0
        for sample_item in sample_split(samples, options.split):
            client = None
            if options.executor == "dask/lxplus":
                _x509_localpath = [l for l in os.popen('voms-proxy-info').read().split("\n") if l.startswith('path')][0].split(":")[-1].strip()
                _x509_path = os.environ['HOME'] + f'/.{_x509_localpath.split("/")[-1]}'
                os.system(f'cp {_x509_localpath} {_x509_path}')

                env_extra = [
                    'export XRD_RUNFORKHANDLER=1',
                    f'export X509_USER_PROXY={_x509_path}',
                    'ulimit -u 32768',
                ]

                cluster = HTCondorCluster(
                    cores=1,
                    memory='10GB',
                    disk='20GB',
                    death_timeout = '60',
                    nanny = False,
                    scheduler_options={
                        'port': 8786,
                        'host': f"{socket.gethostname()}"
                    },
                    job_extra={
                        'log'   : 'dask_job_output.log',
                        'output': 'dask_job_output.out',
                        'error' : 'dask_job_output.err',
                        'should_transfer_files'  : 'YES',
                        'when_to_transfer_output': 'ON_EXIT',
                        '+JobFlavour'            : '"workday"',
                    },
                    env_extra= env_extra,
                    extra = ['--worker-port 10000:10100']
                )
                client = Client(cluster)
                cluster.scale(jobs=options.jobs)
                print('------------------------------------')
                print('dask client based on HTCondorCluster')
                print(client)
                print('Socket  : ', socket.socket)
                print(cluster.job_script())
                print('------------------------------------')

            if options.executor == "dask/casa":
                client = Client("tls://yacine-2ehaddad-40cern-2ech.dask.cmsaf-prod.flatiron.hollandhpc.org:8786")
                client.wait_for_workers(1)
                client.upload_file("./dist/" + qawa_wheel)
                client.register_worker_plugin(dependency_installer)
                client.run(fix_uproot)
            else:
                cluster.adapt(minimum=options.jobs)
                client = Client(cluster)
                client.wait_for_workers(1)

            if not client:
                sys.exit(0)

            with performance_report(filename=f"dask-report-{njobs}.html"):
                def check_files(nfiles: List):
                    import uproot 
                    status = {}
                    for fn in nfiles:
                        try:
                            uproot.open(fn)
                            status[fn] = 'OK'
                        except:
                            status[fn] = 'BAD'
                    return status

                for s in sample_item:
                    print('sample : ', s)
                    out = client.submit(check_files, sample_item[s]['files'])
                    
                    fff = out.result()

                    for f,s in fff.items():
                        print(f, s)
                    break

                """
                dask_runner = processor.Runner(
                    executor = processor.DaskExecutor(client=client),
                    schema = NanoAODSchema,
                    chunksize=10000,
                )
                sumw_runner = processor.Runner(
                    executor = processor.DaskExecutor(client=client),
                    schema = BaseSchema,
                )
                sumw = sumw_runner(
                    sample_item, 
                    treename="Runs",
                    processor_instance=coffea_sumw(),
                )
                client.restart()
                
                outh = dask_runner(
                    sample_item,
                    treename="Events",
                    processor_instance=processor_instance,
                )
                client.restart()
                
                bh_output = {}
                for key, content in outh.items():
                    bh_output[key] = {
                        "hist": content,
                        "sumw": sumw[key]
                }
                with gzip.open(f"histogram-2018-part-{njobs}.pkl.gz", "wb") as f:
                    pickle.dump(bh_output, f)
                """
            njobs += 1
            logging.info(" .. retsrating the client .. ")
            
            logging.info(" --- that's all folks --- ")
            break
                
                
      
if __name__ == "__main__":
    main()
    
    
        

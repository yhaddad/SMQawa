import os
from coffea import processor
from coffea import nanoevents
from qawa.process.zz2l2nu_vbs import *
from qawa.coffea_sumw import coffea_sumw
from dask_jobqueue.htcondor import HTCondorCluster
from dask.distributed import Client
from dask.distributed import performance_report
import warnings
import socket
import yaml
import numpy as np
import argparse
import pickle
import gzip

def main():
    parser = argparse.ArgumentParser("")
    parser.add_argument('-jobs' , '--jobs', type=int, default=200   , help="")
    parser.add_argument('-era'  , '--era' , type=str, default="2018", help="")
    parser.add_argument('-nt'   , '--ntry', type=int, default= 10, help="")
    options = parser.parse_args()

    _x509_localpath = [l for l in os.popen('voms-proxy-info').read().split("\n") if l.startswith('path')][0].split(":")[-1].strip()
    _x509_path = os.environ['HOME'] + f'/.{_x509_localpath.split("/")[-1]}'
    os.system(f'cp {_x509_localpath} {_x509_path}')

    env_extra = [
        f'export XRD_RUNFORKHANDLER=1',
        f'export X509_USER_PROXY={_x509_path}',
        #f'export X509_CERT_DIR={os.environ["X509_CERT_DIR"]}',
        #'ulimit -u 32768',
    ]


    cluster = HTCondorCluster(
        cores=1, # This should be forced to 1 core
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
            'transfer_input_files'   : './qawa/data/GNNmodel/bestEpoch-10-2Jets.onnx,./qawa/data/GNNmodel/bestEpoch-10-3Jets.onnx,',
            'should_transfer_files'  : 'YES',
            'when_to_transfer_output': 'ON_EXIT',
            '+JobFlavour'            : '"tomorrow"',
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

    samples = {}
    with open('./qawa/data/datasetUL2018.yaml') as s_file:
        samples = yaml.full_load(s_file)

    for proc, cont in samples.items():
        print(proc, " --> #files = {}".format(len(cont['files'])) )

    # this is the list of systematics used in 2l2nu analysis
    v_syst_list = [
            "ElectronEn", "MuonEn", "jesTotal", "jer"
    ]
    w_syst_list = [
        "puWeight", "PDF", "MuonSF",
        "ElecronSF", "EWK", "nvtxWeight",
        "TriggerSF", "BTagSF",
        "QCDScale0", "QCDScale1", "QCDScale2"
    ]


    import time
    while len(client.ncores()) < 4:
        print('   -- waiting for more cores to spin up: {0} available'.format(len(client.ncores())))
        print('   -- Dask client info ->', client)

        time.sleep(10)

    # run_instance = processor.Runner(
    #     metadata_cache={},
    #     executor=processor.DaskExecutor(client=client),
    #     schema=processor.NanoAODSchema,
    #     savemetrics=True,
    #     skipbadfiles=True,
    # )

    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")
    #     hosts, metrics = run_instance(samples, processor_instance=zzinc_processor(), treename="Events")

    sumw_out = processor.run_uproot_job(
        samples,
        treename="Runs",
        processor_instance=coffea_sumw(),
        executor=processor.dask_executor,
        executor_args={
            "client": client,
            "align_clusters": True,
            "skipbadfiles":True,
            'retries': 10,
        },
        chunksize=50000,
    )

    vbs_out = processor.run_uproot_job(
            samples,
            processor_instance=zzinc_processor(),
            treename='Events',
            executor=processor.dask_executor,
            executor_args={
                "client": client,
                "skipbadfiles":True,
                "schema": nanoevents.BaseSchema,
                'align_clusters': True,
                'retries': 2,
            },
    )
    # append the sumw on the boost histograms
    bh_output = {}

    for key, content in vbs_out.items():
        bh_output[key] = {
            "hist": content,
            # "sumw": sumw_out[key]
    }

    with gzip.open("histogram-vbs-zz-2018.pkl.gz", "wb") as f:
        pickle.dump(bh_output, f)

    print(bh_output)

if __name__ == "__main__":
    main()

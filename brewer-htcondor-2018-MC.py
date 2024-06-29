import os
import argparse
import logging
import subprocess
import shutil
import time
import importlib.metadata
import yaml

logging.basicConfig(level=logging.DEBUG)

#qawa_version = importlib.metadata.version('qawa')
qawa_version = '0.0.7'

script_TEMPLATE = """#!/bin/bash
export X509_USER_PROXY={proxy}
export XRD_REQUESTTIMEOUT=6400
export XRD_REDIRECTLIMIT=64
mkdir -p /tmp/hgao/{sample_name}
if [ "{sample_name}" = "ZZZ_TuneCP5_13TeV-amcatnlo-pythia8" ] || \
   [ "{sample_name}" = "WWZ_4F_TuneCP5_13TeV-amcatnlo-pythia8" ] || \
   [ "{sample_name}" = "WZZ_TuneCP5_13TeV-amcatnlo-pythia8" ]; then
    cp /eos/user/h/hgao/ZZTo2L2Nu/HZZsample/{sample_name}/*.root /tmp/hgao/{sample_name}/
else
    cp /eos/user/h/hgao/ZZTo2L2Nu/HZZsample/{sample_name}/*_$(expr $1 + 1).root /tmp/hgao/{sample_name}/
fi
voms-proxy-info -all
voms-proxy-info -all -file {proxy}

python -m venv --without-pip --system-site-packages jobenv
source jobenv/bin/activate
python -m pip install scipy --upgrade --no-cache-dir
python -m pip install --no-deps --ignore-installed --no-cache-dir Qawa-{qawa_version}-py2.py3-none-any.whl

echo "----- JOB STARTS @" `date "+%Y-%m-%d %H:%M:%S"`
echo "----- X509_USER_PROXY    : $X509_USER_PROXY"
echo "----- XRD_REDIRECTLIMIT  : $XRD_REDIRECTLIMIT"
echo "----- XRD_REQUESTTIMEOUT : $XRD_REQUESTTIMEOUT"
ls -lthr
echo "----- download the file locally"

echo "----- processing the files : "
python brewer-remote.py --jobNum=$1 --isMC={ismc} --era={era} --infile=$2 --runperiod=period --dataset={sample_name}
echo "----- directory after running :"
ls -lthr
if [ ! -f "histogram_$1.pkl.gz" ]; then
  exit 1;
fi
rm -rf /tmp/hgao/{sample_name}
echo " ------ THE END (everyone dies !) ----- "
"""


condor_TEMPLATE = """
universe              = vanilla
request_disk          = 10000000

executable            = {jobdir}/script.sh
arguments             = $(ProcId) $(jobfn)
transfer_input_files  = {transfer_file}
# transfer_output_files = histogram_$(ProcId).pkl.gz 
should_transfer_files = YES
WhenToTransferOutput  = ON_EXIT_OR_EVICT
initialdir            = {jobdir}

output                = $(ClusterId).$(ProcId).out
error                 = $(ClusterId).$(ProcId).err
log                   = $(ClusterId).$(ProcId).log

on_exit_remove        = (ExitBySignal == False) && (ExitCode == 0)
max_retries           = 2
requirements          = Machine =!= LastRemoteHost
# MY.XRDCP_CREATE_DIR   = True
+SingularityImage     = "/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask:latest"
+JobFlavour           = "{queue}"

queue jobfn from {jobdir}/inputfiles.dat
"""

def main():
    parser = argparse.ArgumentParser(description='Famous Submitter')
    parser.add_argument("-i"   , "--input" , type=str, default="data.txt" , help="input datasets", required=True)
    parser.add_argument("-t"   , "--tag"   , type=str, default="atakour"  , help="production tag", required=True)
    parser.add_argument("-isMC", "--isMC"  , type=int, default=1          , help="")
    parser.add_argument("-q"   , "--queue" , type=str, default="longlunch", help="")
    parser.add_argument("-e"   , "--era"   , type=str, default="2018"     , help="")
    parser.add_argument("-f"   , "--force" , action="store_true"          , help="recreate files and jobs")
    parser.add_argument("-s"   , "--submit", action="store_true"          , help="submit only")
    parser.add_argument("-dry" , "--dryrun", action="store_true"          , help="running without submission")
    parser.add_argument("--redo-proxy"     , action="store_true"          , help="redo the voms proxy")
    options = parser.parse_args()

    # Making sure that the proxy is good
    proxy_base = 'x509up_u{}'.format(os.getuid())
    home_base  = os.environ['HOME']
    user_name  = os.environ['USER']
    proxy_copy = os.path.join(home_base,proxy_base)
    eosbase = f"/eos/user/{user_name[0]}/{user_name}/ZZTo2L2Nu/"
    regenerate_proxy = False

    with open(options.input, 'r') as stream:
        for sample in stream.read().split('\n'):
            sample_name = sample.split(".yaml")[0] 
            period = ''
            if options.isMC == 0:
                sample_name = sample_name.split("_")[0] 
                period = sample.split(".yaml")[0].split("_")[1] 
            era_directory = '2018'  
            os.makedirs('2018', exist_ok=True)
            jobs_dir111 = '_'.join(['jobs', options.tag, options.era, sample.split(".yaml")[0] ])
            jobs_dir = os.path.join(era_directory, jobs_dir111)
            logging.info("-- sample_name : " + sample)

            if os.path.isdir(jobs_dir):
                if not options.force:
                    logging.error(" " + jobs_dir + " already exist !")
                    continue
                else:
                    logging.warning(" " + jobs_dir + " already exists, forcing its deletion!")
                    shutil.rmtree(jobs_dir)
                    jobs_dir = '_'.join(['jobs', options.tag, options.era, sample_name])
            else:
                os.mkdir(jobs_dir)

            if not options.submit:
                with open("data/config/"+sample, "r") as file:
                    data = yaml.safe_load(file)
                sample_files = []
                sample_files = data[sample_name]['files']
                with open(os.path.join(jobs_dir, "inputfiles.dat"), 'w') as infiles:
                    for fn in sample_files:
                        infiles.write(fn)
                        infiles.write('\n')
                    infiles.close()
            time.sleep(2)
            eosoutdir = os.path.join(eosbase, jobs_dir)
            #eosoutdir =  eosbase.format(tag=options.tag,sample=sample_name)
            # crete a directory
            #os.system("mkdir -p {}".format(eosoutdir))

            with open(os.path.join(jobs_dir, "script.sh"), "w") as scriptfile:
                script = script_TEMPLATE.format(
                    proxy=proxy_copy,
                    ismc=options.isMC,
                    era=options.era,
                    eosdir=eosoutdir, 
                    qawa_version=qawa_version,
                    sample_name=sample_name,
                    period=period
                )
                scriptfile.write(script)
                scriptfile.close()

            with open(os.path.join(jobs_dir, "condor.sub"), "w") as condorfile:
                condor = condor_TEMPLATE.format(
                    transfer_file= ",".join([
                        f"../../brewer-remote.py",
                        f"../../dist/Qawa-{qawa_version}-py2.py3-none-any.whl",
                    ]),
                    jobdir=jobs_dir,
                    queue=options.queue
                )
                condorfile.write(condor)
                condorfile.close()
            if options.dryrun:
                continue

            htc = subprocess.Popen(
                "condor_submit " + os.path.join(jobs_dir, "condor.sub"),
                shell  = True,
                stdin  = subprocess.PIPE,
                stdout = subprocess.PIPE,
                stderr = subprocess.PIPE,
                close_fds=True
            )
            
            htc.communicate()
            exit_status = htc.returncode
            logging.info("condor submission status : {}".format(exit_status))

if __name__ == "__main__":
    main()

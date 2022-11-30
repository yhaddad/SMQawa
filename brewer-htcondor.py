import os
import argparse
import logging
import subprocess
import shutil
import time

logging.basicConfig(level=logging.DEBUG)

qawa_version = '0.0.5'


script_TEMPLATE = """#!/bin/bash
export X509_USER_PROXY={proxy}

python -m venv --without-pip --system-site-packages jobenv
source jobenv/bin/activate
python -m pip install --no-deps --ignore-installed --no-cache-dir Qawa-{qawa_version}-py2.py3-none-any.whl

echo "... start job at" `date "+%Y-%m-%d %H:%M:%S"`
echo "----- directory before running:"
echo "----- Found Proxy in: $X509_USER_PROXY"
ls -lthr

echo "python brewer-remote.py --jobNum=$1 --isMC={ismc} --era={era} --infile=$2"
python brewer-remote.py --jobNum=$1 --isMC={ismc} --era={era} --infile=$2

echo "----- directory after running :"
ls -lthr .

if [ ! -f "histogram_$1.pkl.gz" ]; then
  exit 1;
fi

echo "----- transfert output to eos {eosdir}"
xrdcp -s -f histogram_$1.pkl.gz {eosdir}

echo " ------ THE END (everyone dies !) ----- "
"""


condor_TEMPLATE = """
universe              = vanilla
request_disk          = 10000000

executable            = {jobdir}/script.sh
arguments             = $(ProcId) $(jobid)
transfer_input_files  = {transfer_file}
should_transfer_files = YES
WhenToTransferOutput  = ON_EXIT_OR_EVICT
initialdir            = {jobdir}

output                = $(ClusterId).$(ProcId).out
error                 = $(ClusterId).$(ProcId).err
log                   = $(ClusterId).$(ProcId).log

on_exit_remove        = (ExitBySignal == False) && (ExitCode == 0)
max_retries           = 2
requirements          = Machine =!= LastRemoteHost

+SingularityImage     = "/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask:latest"
+JobFlavour           = "{queue}"

queue jobid from {jobdir}/inputfiles.dat
"""

def main():
    parser = argparse.ArgumentParser(description='Famous Submitter')
    parser.add_argument("-i"   , "--input" , type=str, default="data.txt" , help="input datasets", required=True)
    parser.add_argument("-t"   , "--tag"   , type=str, default="algiers"  , help="production tag", required=True)
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
    proxy_copy = os.path.join(home_base,proxy_base)
    eosbase = "/eos/user/y/yixiao/ZZTo2L2Nu/{tag}/{sample}/"

    regenerate_proxy = False
    if not os.path.isfile(proxy_copy):
        logging.warning('--- proxy file does not exist')
        regenerate_proxy = True
    else:
        lifetime = subprocess.check_output(
            ['voms-proxy-info', '--file', proxy_copy, '--timeleft']
        )
        print (lifetime)
        lifetime = float(lifetime)
        lifetime = lifetime / (60*60)
        logging.info("--- proxy lifetime is {} hours".format(lifetime))
        if lifetime < 10.0:
            logging.warning("--- proxy has expired !")
            regenerate_proxy = True

    if regenerate_proxy:
        redone_proxy = False
        while not redone_proxy:
            status = os.system('voms-proxy-init -voms cms')
            if os.WEXITSTATUS(status) == 0:
                redone_proxy = True
        shutil.copyfile('/tmp/'+proxy_base,  proxy_copy)


    with open(options.input, 'r') as stream:
        for sample in stream.read().split('\n'):
            if '#' in sample: continue
            if len(sample.split('/')) <= 1: continue
            sample_name = sample.split("/")[1] if options.isMC else '_'.join(sample.split("/")[1:3])
            sample_name = sample_name.replace("*", "")
            jobs_dir = '_'.join(['jobs', options.tag, sample_name])
            logging.info("-- sample_name : " + sample)

            if os.path.isdir(jobs_dir):
                if not options.force:
                    logging.error(" " + jobs_dir + " already exist !")
                    continue
                else:
                    logging.warning(" " + jobs_dir + " already exists, forcing its deletion!")
                    shutil.rmtree(jobs_dir)
                    os.mkdir(jobs_dir)
            else:
                os.mkdir(jobs_dir)

            if not options.submit:
                # ---- getting the list of file for the dataset
                sample_files = subprocess.check_output(
                    ['dasgoclient','--query',"file dataset={}".format(sample)]
                )
                sample_files=str(sample_files)
                sample_files=sample_files.split('b\'')[1].split('\\n')
                del sample_files[-1]
                time.sleep(15)
                with open(os.path.join(jobs_dir, "inputfiles.dat"), 'w') as infiles:
                    for fn in sample_files:
                        infiles.write(fn)
                        infiles.write('\n')
                    infiles.close()
            time.sleep(10)
            eosoutdir =  eosbase.format(tag=options.tag,sample=sample_name)
            # crete a directory
            os.system("mkdir -p {}".format(eosoutdir))

            with open(os.path.join(jobs_dir, "script.sh"), "w") as scriptfile:
                script = script_TEMPLATE.format(
                    proxy=proxy_copy,
                    ismc=options.isMC,
                    era=options.era,
                    eosdir=eosoutdir, 
                    qawa_version=qawa_version
                )
                scriptfile.write(script)
                scriptfile.close()

            with open(os.path.join(jobs_dir, "condor.sub"), "w") as condorfile:
                condor = condor_TEMPLATE.format(
                    transfer_file= ",".join([
                        f"../brewer-remote.py",
                        f"../dist/Qawa-{qawa_version}-py2.py3-none-any.whl",
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

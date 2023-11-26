import os
import argparse
import logging
import subprocess
import shutil
import time
import importlib.metadata

logging.basicConfig(level=logging.DEBUG)

qawa_version = importlib.metadata.version('qawa')
#qawa_version = '0.0.5'


script_TEMPLATE = """#!/bin/bash
export X509_USER_PROXY={proxy}
export XRD_REQUESTTIMEOUT=6400
export XRD_REDIRECTLIMIT=64

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

echo "----- processing the files : "
python brewer-dump-local.py --jobNum=$1 --isMC={ismc} --era={era} --infile=$2

echo "----- directory after running :"
ls -lthr
if [ ! -f "df_$1.parquet" ]; then
  exit 1;
fi
echo " ------ THE END (everyone dies !) ----- "
"""


condor_TEMPLATE = """
universe              = vanilla
request_disk          = 10000000

executable            = {jobdir}/script.sh
arguments             = $(ProcId) $(jobfn)
transfer_input_files  = {transfer_file}
# transfer_output_files = df_$(ProcId).parquet 
should_transfer_files = YES
WhenToTransferOutput  = ON_EXIT_OR_EVICT
initialdir            = {jobdir}
# output_destination    = root://eosuser.cern.ch//eos/user/y/yhaddad/condor_jobs/{jobdir}/

output                = $(ClusterId).$(ProcId).out
error                 = $(ClusterId).$(ProcId).err
log                   = $(ClusterId).$(ProcId).log

on_exit_remove        = (ExitBySignal == False) && (ExitCode == 0)
max_retries           = 2
requirements          = Machine =!= LastRemoteHost
# MY.XRDCP_CREATE_DIR   = True
+SingularityImage     = "/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask:0.7.21-fastjet-3.4.0.1-g6238ea8"
+JobFlavour           = "{queue}"

queue jobfn from {jobdir}/inputfiles.dat
"""

def main():
    parser = argparse.ArgumentParser(description='Famous Submitter')
    parser.add_argument("-i"   , "--input" , type=str, default="data.txt" , help="input datasets", required=True)
    parser.add_argument("-t"   , "--tag"   , type=str, default="atakour"  , help="production tag", required=True)
    parser.add_argument("-isMC", "--isMC"  , type=int, default=1          , help="")
    parser.add_argument("-q"   , "--queue" , type=str, default="workday", help="")
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
    eosbase = f"/eos/user/{user_name[0]}/{user_name}/ZZTo2L2Nu/" + "{tag}/{sample}/"

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
            jobs_dir = '_'.join(['jobs', options.tag, options.era, sample_name])
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
                sample_file = []
                if '*' in sample:
                    sample_with_ext = subprocess.check_output(
                        ['dasgoclient', '--query', f"dataset={sample}"]
                    )
                    print(" --- found these samples : ")
                    print(sample_with_ext.decode('UTF-8'))
                    print("and these are the files : ")
                    for sample_ in sample_with_ext.decode("UTF-8").split("\n")[:-1]:
                        output_ = subprocess.check_output(
                            ['dasgoclient', '--query', f'file dataset={sample_}']
                        )
                        sample_files += list(filter(lambda x: x != '', output_.decode('UTF-8').split('\n')))
                else:
                    output_ = subprocess.check_output(
                        ['dasgoclient','--query', f"file dataset={sample}"]
                    )
                    sample_files = list(filter(lambda x: x != '', output_.decode('UTF-8').split('\n')))

                time.sleep(1)
                with open(os.path.join(jobs_dir, "inputfiles.dat"), 'w') as infiles:
                    for fn in sample_files:
                        infiles.write(fn)
                        infiles.write('\n')
                    infiles.close()
            time.sleep(2)
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
                        f"../brewer-dump-local.py",
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

import os
import argparse
import shutil
import logging
import subprocess
from pandas.core.internals.array_manager import new_block
from termcolor import colored
import importlib.metadata
qawa_version = "0.0.7"

logging.basicConfig(level=logging.INFO)

rerun_script_header = f"""#!/bin/bash
cd /srv/
python -m venv --without-pip --system-site-packages jobenv
source jobenv/bin/activate
python -m pip install scipy --upgrade --no-cache-dir
python -m pip install --no-deps --ignore-installed --no-cache-dir Qawa-{qawa_version}-py2.py3-none-any.whl

echo "... start job at" `date "+%Y-%m-%d %H:%M:%S"`
echo "----- directory before running:"
ls -lthr
"""


resub_script_header = """#!/bin/bash
export X509_USER_PROXY={proxy}

python -m venv --without-pip --system-site-packages jobenv
source jobenv/bin/activate
python -m pip install scipy --upgrade --no-cache-dir
python -m pip install --no-deps --ignore-installed --no-cache-dir Qawa-{qawa_version}-py2.py3-none-any.whl

echo "... start job at" `date "+%Y-%m-%d %H:%M:%S"`
echo "----- directory before running:"
echo "----- Found Proxy in: $X509_USER_PROXY"
ls -lthr
{command}

echo "----- directory after running :"
ls -lthr .
if [ ! -f "histogram_{jobid}.pkl.gz" ]; then
  exit 1;
fi
echo " ------ THE END (everyone dies !) ----- "
"""

def main():
    parser = argparse.ArgumentParser(description='Famous Submitter')
    parser.add_argument("-i"   , "--input" , type=str, default="input"  , required=True)
    parser.add_argument("-t"   , "--tag"   , type=str, default="algiers", required=True)
    parser.add_argument("-isMC", "--isMC"  , type=int, default=1        , help="")
    parser.add_argument("--era"     , type=str, default="")
    parser.add_argument("--runlocal", action="store_true")
    parser.add_argument("--resubmit", action="store_true", help="resubmit failed jobs")
    parser.add_argument("--copyfile", action="store_true", help="")
    parser.add_argument("--dryrun"  , action="store_true")
    options = parser.parse_args()


    condor_status = os.popen('condor_q -nobatch').read()
   
    proxy_base = 'x509up_u{}'.format(os.getuid())
    home_base  = os.environ['HOME']
    proxy_copy = os.path.join(home_base,proxy_base)
    
    if not os.path.isfile(proxy_copy):
        logging.warning('--- proxy file does not exist')
    else:
        lifetime = subprocess.check_output(
            ['voms-proxy-info', '--file', proxy_copy, '--timeleft']
        )    
        lifetime = float(lifetime)
        lifetime = lifetime / (60*60)
        logging.info("--- proxy lifetime is {} hours".format(lifetime))
        if lifetime < 10.0: # we want at least 10 hours
            logging.warning("--- proxy has expired !")


    with open(options.input, 'r') as stream:
        for sample in stream.read().split('\n'):
            if '#' in sample: continue
            if len(sample.split('/')) <= 1: continue
            sample_name = sample.split("/")[1] if options.isMC else '_'.join(sample.split("/")[1:3])
            jobs_dir = '_'.join(['jobs', options.tag, options.era, sample_name])

            input_root_files = list(open(jobs_dir + "/" + "inputfiles.dat").read().splitlines())
            
            n_jobs = len(input_root_files)

            job_running = []
            job_failed = []
            job_finished = []
            resubmit_list = {}
            for idf, rfn in enumerate(input_root_files):
                if rfn in condor_status:
                    job_running.append(rfn)
                elif os.path.exists(jobs_dir + f'/histogram_{idf}.pkl.gz'):
                    job_finished.append(rfn)
                else:
                    job_failed.append(rfn)
                    resubmit_list[idf] = rfn
            logging.info(
                "-- {:62s}".format((sample_name[:60] + '..') if len(sample_name)>60 else sample_name) +
                (
                    colored(f" --> {n_jobs:5d} : completed", "green") if n_jobs==len(job_finished) else colored(
                        f" --> {n_jobs:5d} : {len(job_running):5d}", 'yellow'
                    )+colored(
                        f"{n_jobs-len(job_failed)-len(job_running):5d}", "green"
                    )+colored(
                        f"{len(job_failed):5d}", 'red'
                    )
                )
            )

            if len(job_running)>0:
                for rfn in job_running:
                    logging.debug(colored(f'running : {rfn}', 'yellow'))
            if len(job_failed)>0:
                for rfn in job_failed:
                    logging.debug(colored(f'failed  : {rfn}', 'red'))
            
            if options.resubmit and len(job_failed)>0: 
                shutil.copyfile('brewer-remote-inclusive.py', jobs_dir+'/brewer-remote-inclusive.py')
                local_rerun_lines = [rerun_script_header]
                for jid,infile in resubmit_list.items():
                    infile_name = infile 
                    dataset_name = infile.split('/')[4]
                    run_period = ''
                    if options.isMC:
                        run_period = ''
                    else:
                        run_period = infile.split('/')[3].replace(f'Run{options.era}','')
                    if options.runlocal:
                        assert options.era != "", f"please specify the era you are rerunning ... example: --era=2018"
                        if options.copyfile:
                            local_rerun_lines.append(
                                "xrdcp {infile} ."
                            )
                            infile_name = infile.split('/')[-1]
                        local_rerun_lines.append(
                            f"python brewer-remote-inclusive.py --jobNum={jid} --isMC={options.isMC} --era={options.era} --infile={infile_name} --dataset={dataset_name}\n"
                        )
                    else:
                        if options.copyfile:
                            assert options.era != "", f'please specify the era of the dataset you are running. ex: --era=2018'
                            script_command = f"xrdcp root://cms-xrd-global.cern.ch/$2 . \n"
                            infile_name = infile.split('/')[-1]
                            script_command += f"python brewer-remote-inclusive.py --jobNum=$1 --isMC={options.isMC} --era={options.era} --infile={infile_name} --dataset={dataset_name} --runperiod={run_period}\n"
                            script_command += f"rm {infile_name}\n"
                            script_command += "ls -lthr\n"
                            with open(os.path.join(jobs_dir, f"resub-script-{jid}.sh"), "w") as _stream:
                                script_file_ = resub_script_header.format(
                                    proxy=proxy_copy, 
                                    qawa_version=qawa_version,
                                    command=script_command,
                                    jobid=jid
                                )
                                _stream.write(script_file_)
                                _stream.close()
                        else: 
                            assert options.era != "", f'please specify the era of the dataset you are running. ex: --era=2018'
                            script_command = f"python brewer-remote-inclusive.py --jobNum=$1 --isMC={options.isMC} --era={options.era} --infile=$2\n"
                            script_command += "ls -lthr\n"
                            with open(os.path.join(jobs_dir, f"resub-script-{jid}.sh"), "w") as _stream:
                                script_file_ = resub_script_header.format(
                                    proxy=proxy_copy, 
                                    qawa_version=qawa_version,
                                    command=script_command,
                                    jobid=jid
                                )
                                _stream.write(script_file_)
                                _stream.close()

                        condor_sub = open(jobs_dir + "/condor.sub").readlines()
                        for il, line in enumerate(condor_sub):
                            if 'executable' in line.lower(): 
                                condor_sub [il] = f'executable = {jobs_dir}/resub-script-{jid}.sh\n'
                                #condor_sub [il] = f'executable = {jobs_dir}/script.sh\n'
                            if 'arguments' in line.lower():
                                condor_sub [il] = f"arguments = {jid} {infile}\n"
                            if 'jobflavour' in line.lower():
                                condor_sub [il] = '+JobFlavour           = "workday"\n'
                            if 'queue' in line.lower():
                                condor_sub[il] = "queue"
                        with open(jobs_dir + f'/condor_resub_{jid}.sub', 'w') as new_condor:
                            new_condor.writelines(condor_sub)
                            new_condor.close()
                        
                        if not options.dryrun:
                            htc = os.popen("condor_submit " + os.path.join(jobs_dir, f"condor_resub_{jid}.sub")).read()
                            logging.info(htc)
                
                if options.runlocal:
                    with open(os.path.join(jobs_dir, f"rerun-script.sh"), "w") as _stream:
                        _stream.writelines(local_rerun_lines)

                    coffea_image = "/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask:latest" 
                    os.system(f"cp dist/Qawa-0.0.7-py2.py3-none-any.whl {jobs_dir}")   
                    if not options.dryrun:
                        htc = os.popen(f"singularity exec -B {jobs_dir}:/srv/ {coffea_image} bash /srv/rerun-script.sh").read()
                        print(htc)
                    
                    


if __name__ == "__main__":
    main()


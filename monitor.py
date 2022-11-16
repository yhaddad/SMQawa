import os
import argparse
import shutil
import logging
import subprocess
from termcolor import colored

logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(description='Famous Submitter')
    parser.add_argument("-i"   , "--input" , type=str, default="input"  , required=True)
    parser.add_argument("-t"   , "--tag"   , type=str, default="algiers", required=True)
    parser.add_argument("-isMC", "--isMC"  , type=int, default=1        , help="")
    parser.add_argument("--era", type=str, default='2018')
    parser.add_argument("--runlocal", action='store_true')
    parser.add_argument("--resubmit", action="store_true", help="resubmit failed jobs")
    options = parser.parse_args()


    condor_status = os.popen('condor_q -nobatch').read()
   
    regenerate_proxy = False


    proxy_base = 'x509up_u{}'.format(os.getuid())
    home_base  = os.environ['HOME']
    proxy_copy = os.path.join(home_base,proxy_base)
    
    if not os.path.isfile(proxy_copy):
        logging.warning('--- proxy file does not exist')
        regenerate_proxy = True
    else:
        lifetime = subprocess.check_output(
            ['voms-proxy-info', '--file', proxy_copy, '--timeleft']
        )
    
        lifetime = float(lifetime)
        lifetime = lifetime / (60*60)
        logging.info("--- proxy lifetime is {} hours".format(lifetime))
        if lifetime < 10.0: # we want at least 10 hours
            logging.warning("--- proxy has expired !")
            regenerate_proxy = True


    with open(options.input, 'r') as stream:
        for sample in stream.read().split('\n'):
            if '#' in sample: continue
            if len(sample.split('/')) <= 1: continue
            sample_name = sample.split("/")[1] if options.isMC else '_'.join(sample.split("/")[1:3])
            jobs_dir = '_'.join(['jobs', options.tag, sample_name])

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
            
            if options.runlocal and len(job_failed)>0: 
                shutil.copyfile('brewer-remote.py', jobs_dir+'/brewer-remote.py')
                for jid,infile in resubmit_list.items():
                    condor_sub = open(jobs_dir + "/condor.sub").readlines()
                    for il, line in enumerate(condor_sub):
                        if 'arguments' in line.lower():
                            condor_sub[il] = f"arguments             = {jid} {infile}\n"
                        if 'queue' in line.lower():
                            condor_sub[il] = "queue"
                    with open(jobs_dir + f'/condor_resub_{jid}.sub', 'w') as new_condor:
                        new_condor.writelines(condor_sub)
                        new_condor.close()

                    htc = os.popen("condor_submit " + os.path.join(jobs_dir, f"condor_resub_{jid}.sub")).read()
                    logging.info(htc)
                    

            if options.resubmit and len(job_failed)>0:
                attempt = 1
                with open(os.path.join(jobs_dir, f"inputfiles-attempt-{attempt}.dat"), 'w') as infiles:
                    infiles.write('\n'.join(job_failed))
                    infiles.close()
                
                with open(os.path.join(jobs_dir, 'condor.sub'), 'r') as condor_file:
                    condor = condor_file.read()
                    condor = condor.replace('inputfiles', f'inputfiles-attempt-{attempt}')

                with open(os.path.join(jobs_dir, f'condor_attempt_{attempt}.sub'), 'w') as condor_file:
                    condor_file.write(condor)

                if regenerate_proxy:
                    redone_proxy = False
                    while not redone_proxy:
                        status = os.system('voms-proxy-init -voms cms')
                        if os.WEXITSTATUS(status) == 0:
                            redone_proxy = True
                
                    shutil.copyfile('/tmp/'+proxy_base,  iroxy_copy)


                htc = os.popen("condor_submit " + os.path.join(jobs_dir, f"condor_attempt_{attempt}.sub")).read()
                logging.info(htc)



if __name__ == "__main__":
    main()


import os
import yaml
import time
import argparse
import logging
import subprocess
import shutil
import uproot
from termcolor import colored

logging.basicConfig(level=logging.DEBUG)

aliases = [
        "root://eoscms.cern.ch/",
        "root://xrootd-cms.infn.it/",
        "root://cms-xrd-global.cern.ch/",
]


def check_file(file):
    print (file)
    for alias in aliases:
        testfile = None
        success = False
        try:
            testfile = uproot.open(alias  + file)
        except:
            pass
        if testfile:
            print(colored(f'--> {alias} OK', 'green'))
            return alias
        else:
            print(colored(f'--> {alias} FAILD', 'red'))
    return ' -- bad file'

def main():
    parser = argparse.ArgumentParser(description='Famous Submitter')
    parser.add_argument("-i"   , "--input" , type=str, default="data.txt" , help="input datasets", required=True)

    options = parser.parse_args()

    proxy_base = 'x509up_u{}'.format(os.getuid())
    home_base  = os.environ['HOME']
    proxy_copy = os.path.join(home_base,proxy_base)

    regenerate_proxy = False
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

    print(regenerate_proxy)
    if regenerate_proxy:
        redone_proxy = False
        while not redone_proxy:
            status = os.system('voms-proxy-init -voms cms')
            if os.WEXITSTATUS(status) == 0:
                redone_proxy = True
        shutil.copyfile('/tmp/'+proxy_base,  proxy_copy)

    catalog = {}
    with open(options.input, 'r') as stream:
        for sample in stream.read().split('\n'):
            if '#' in sample: continue
            if len(sample.split('/')) <= 1: continue
            isMC = 'SIM' in sample
            sample_name = sample.split("/")[1] if isMC else '_'.join(sample.split("/")[1:3])
            sample_name = sample_name.replace("*", "")

            sample_files = subprocess.check_output(
                ['dasgoclient','--query',"file dataset={}".format(sample)]
            )

            print(sample_files)

            good_files = []
            for infile in sample_files.decode('utf-8').split('\n'):
                if len(infile) == 0: continue
                print('checking : ', infile)
                filename = check_file(infile)
                if 'bad' not in filename:
                    good_files.append(filename)
            catalog[sample_name] = good_files

            time.sleep(15)
        with open(r'catalogUL.yaml', 'w') as file:
            yaml.dump(catalog, file)



if __name__ == "__main__":
    main()

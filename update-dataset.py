import os, sys
import argparse
import logging
import pwd
import subprocess
import shutil
import time
from numba.core.utils import stream_list
from termcolor import colored
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description='Famous Submitter')
    parser.add_argument("-i"   , "--input" , type=str, default="data.txt" , help="input datasets", required=True)
    parser.add_argument("-t"   , "--tag"   , type=str, default="RunIISummer20UL16NanoAODv9", required=True)

    options = parser.parse_args()

    new_list = set()
    file_name_tag = ''
    with open(options.input, 'r') as stream:
        for sample in tqdm(stream.read().split('\n'), desc="format", ascii=False, ncols=75):
            #for sample in stream.read().split('\n'):
            if '#' in sample:
                continue
            if len(sample.split('/')) <= 1: continue
            
            sample_name = sample.split("/")[1]
            sample_tag = sample.split("/")[2]
            sample_files = subprocess.check_output(
                    [
                        'dasgoclient','--query',"dataset={}".format(
                            sample.replace(sample_tag, options.tag)
                        )
                    ]
            )
            if len(sample_files)>0:
                sample_newtag = sample_files.decode('UTF-8').split("/")[2]
                file_name_tag = sample_newtag.split('-')[0]
                new_list.add(sample_files.decode('UTF-8'))
            else:
                sample_newtag = ''
                new_list.add(sample + '\n')
            #time.sleep(2)
            #print(colored(f"{sample_name} : {sample_tag} : {sample_newtag}", "blue"))
    

    print(new_list)
    with open(f"./data/dataset-{file_name_tag}.txt", 'w') as stream:
        stream.writelines(sorted(list(new_list)))
        stream.close()

if __name__ == "__main__":
    main()



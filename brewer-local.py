from coffea import processor
from coffea import nanoevents
from qawa.process.zz2l2nu_vbs import zzinc_processor
from qawa.process.coffea_sumw import coffea_sumw
import argparse
import pickle
import gzip
import os, re
import uproot
import numpy as np

np.seterr(all='ignore')

uproot.open.defaults["xrootd_handler"] = uproot.source.xrootd.MultithreadedXRootDSource
uproot.open.defaults["timeout"] = 60 * 10 # wait more


def validate_input_file(nanofile):
    pfn = nanofile
    pfn=re.sub("\n","",pfn)
    aliases = [
        "root://eoscms.cern.ch/",
        "root://xrootd-cms.infn.it/",
        "root://cms-xrd-global.cern.ch/",
        "root://cmsxrootd.fnal.gov/"
    ]

    valid = False
    for alias in aliases:
        testfile = None
        try:
            testfile=uproot.open(alias + pfn)
        except:
            pass
        if testfile:
            nanofile=alias + pfn
            print(f'--> {alias} OK')
            valid = True
            break
        else:
            print(f'--> {alias} FAILD')

        if valid==False:
            # all faild force AAA anyways
            nanofile = aliases[-1] + pfn
    return nanofile

def main():
    parser = argparse.ArgumentParser("")
    parser.add_argument('--jobNum', type=int, default=1, help="")
    parser.add_argument('-era'  , '--era' ,   type=str, default="2018", help="")
    parser.add_argument('-isMC' , '--isMC',   type=int, default=1, help="")
    parser.add_argument('-infile','--infile', type=str, default=None  , help="")
    
    options = parser.parse_args()
    dataset = options.infile.split('/')[4]
    #options.infile = validate_input_file(options.infile)
    
    era=options.era
    is_data = not options.isMC

    ixrd = 0
    aliases = [
        "root://xrootd-cms.infn.it/",
        #"root://cmsxrootd.fnal.gov/",
        #"root://cms-xrd-global.cern.ch/",
    ]
    
    samples ={
        dataset:{
            'files': [aliases[ixrd] + options.infile],
            'metadata':{
                'era': era,
                'is_data': is_data
            }
        }
    }
    

    print(
        "---------------------------"
        f"-- options  = {options}"
        f"-- is MC    = {options.isMC}"
        f"-- jobNum   = {options.jobNum}"
        f"-- era      = {options.era}"
        f"-- in file  = {aliases[ixrd] + options.infile}"
        f"-- dataset  = {dataset}"
        "---------------------------"
    )

    sumw_out = processor.run_uproot_job(
        samples,
        treename="Runs",
        processor_instance=coffea_sumw(),
        executor=processor.futures_executor,
        executor_args={
            "schema" : nanoevents.BaseSchema,
            "workers": 8,
        },
    )
    
    ewk_flag = None
    if "ZZTo" in options.infile and "GluGluTo" not in options.infile and "ZZJJ" not in options.infile:
        ewk_flag= 'ZZ'
    if "WZTo" in options.infile and "GluGluTo" not in options.infile:
        ewk_flag = 'WZ'
        
    # extarct the run period
    run_period = '' 
    if 'Run20' in options.infile and is_data:
        run_period = options.infile.split('/store/data/')[1].split('/')[0].replace(f'Run{options.era}','')
    
    print(" --------------------------- ")
    vbs_out = processor.run_uproot_job(
        samples,
        processor_instance=zzinc_processor(
            era=options.era,
            ewk_process_name=ewk_flag,
            dump_gnn_array=True,
            run_period=run_period if is_data else ''
        ),
        treename='Events',
        executor=processor.futures_executor,
        executor_args={
            "schema" : nanoevents.NanoAODSchema,
            "workers": 8,
        },
        #chunksize=50000,
    )
    bh_output = {}
    for key, content in vbs_out.items():
        bh_output[key] = {
            "hist": content,
            "sumw": sumw_out[key],
    }
    with gzip.open("histogram_%s.pkl.gz" % str(options.jobNum), "wb") as f:
        pickle.dump(bh_output, f)

if __name__ == "__main__":
    main()

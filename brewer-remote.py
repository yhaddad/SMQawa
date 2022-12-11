from coffea import processor
from coffea import nanoevents
from qawa.process.zz2l2nu_vbs import zzinc_processor
from qawa.process.coffea_sumw import coffea_sumw
import argparse
import pickle
import gzip
import re, sys
import uproot
import numpy as np

np.seterr(all='ignore')

uproot.open.defaults["xrootd_handler"] = uproot.source.xrootd.MultithreadedXRootDSource
uproot.open.defaults["timeout"] = 60 * 5 # wait more


def validate_input_file(nanofile):
    pfn = nanofile
    pfn=re.sub("\n","",pfn)
    aliases = [
        "root://eoscms.cern.ch/",
        "root://xrootd-cms.infn.it/",
        "root://cmsxrootd.fnal.gov/"
        "root://cms-xrd-global.cern.ch/",
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
    parser.add_argument('--jobNum' ,   type=int, default=1     , help="")
    parser.add_argument('--era'    ,   type=str, default="2018", help="")
    parser.add_argument('--isMC'   ,   type=int, default=1     , help="")
    parser.add_argument('--infile' ,   type=str, default=None  , help="input root file")
    parser.add_argument('--dataset',   type=str, default=None  , help="dataset name. need to specify if file is not in EOS") 
    options = parser.parse_args()

    if options.dataset is None: 
        options.dataset = options.infile.split('/')[4]

    era=options.era
    is_data = not options.isMC

    failed = True
    ixrd = 0
    aliases = [
        "root://llrxrd-redir.in2p3.fr/",
        "root://xrootd-cms.infn.it/",
        "root://cms-xrd-global01.cern.ch/", 
        "root://cms-xrd-global02.cern.ch/",
        "root://cmsxrootd.fnal.gov/",
        "root://xrootd-cms-redir-int.cr.cnaf.infn.it/",
        "root://xrootd-redic.pi.infn.it/"
    ]
    while failed:
        try:
            file_name = options.infile
            if '/store/' in options.infile:
                file_name = aliases[ixrd] + options.infile
            else:
                file_name = options.infile 

            samples ={
                options.dataset:{
                    'files': [file_name],
                    'metadata':{
                        'era': era,
                        'is_data': is_data
                    }
                }
            }
            
            print(
                f"""---------------------------
                -- options  = {options}
                -- is MC    = {options.isMC}
                -- jobNum   = {options.jobNum}
                -- era      = {options.era}
                -- in file  = {aliases[ixrd] + options.infile}
                -- dataset  = {options.dataset}
                ---------------------------"""
            )
            
            sumw_out = processor.run_uproot_job(
                samples,
                treename="Runs",
                processor_instance=coffea_sumw(),
                executor=processor.futures_executor,
                executor_args={
                    "schema" : nanoevents.BaseSchema,
                    "workers": 8,
                    "desc": "SumW"
                },
            )
            
            print(" --- zz2l2nu_vbs processor ... ")
            vbs_out = processor.run_uproot_job(
                samples,
                processor_instance=zzinc_processor(era=options.era),
                treename='Events',
                executor=processor.futures_executor,
                executor_args={
                    "schema" : nanoevents.NanoAODSchema,
                    "workers": 8,
                    "desc": "ZZinC"
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
            failed=False
        except:
            print(f"[WARNING] {aliases[ixrd]} failed with the following error : ")
            print(sys.exc_info()[0])
            print("-------------------------------------------")
            failed=True
            ixrd += 1
            if ixrd > (len(aliases) - 1):
                break

if __name__ == "__main__":
    main()

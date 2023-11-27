from coffea import processor
from coffea import nanoevents
from qawa.process.zz2l2nu_vbs import zzinc_processor
from qawa.process.coffea_sumw import coffea_sumw
from qawa.process.coffea_gnn_input import coffea_gnn_input
import argparse
import pickle
import gzip
import re, sys
import uproot
import numpy as np
import pandas as pd

np.seterr(all='ignore')

uproot.open.defaults["xrootd_handler"] = uproot.source.xrootd.MultithreadedXRootDSource
uproot.open.defaults["timeout"] = 60 * 5 # wait more


def validate_input_file(nanofile):
    pfn = nanofile
    pfn=re.sub("\n","",pfn)
    aliases = [
        "root://eoscms.cern.ch/",
        "root://xrootd-cms.infn.it/",
        "root://cmsxrootd.fnal.gov/",
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
        options.dataset = options.infile.split('/')[4] #for none aQGC it's [4], for aQGC it's[7]

    era=options.era
    is_data = not options.isMC

    failed = True
    ixrd = 0
    aliases = [
        "root://eoscms.cern.ch/",
        "root://llrxrd-redir.in2p3.fr/",
        "root://xrootd-cms.infn.it/",
#        "root://cms-xrd-global.cern.ch/",
        "root://cms-xrd-global01.cern.ch/", 
        "root://cms-xrd-global02.cern.ch/",
        "root://cmsxrootd.fnal.gov/",
        "root://xrootd-cms-redir-int.cr.cnaf.infn.it/",
        "root://xrootd-redic.pi.infn.it/",
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
            -- in file  = {aliases[ixrd] + options.infile}
            -- era      = {options.era}
            -- dataset  = {options.dataset}
            ---------------------------"""
            )
            
            

            # sumw_out = processor.run_uproot_job(
            #     samples,
            #     treename="Runs",
            #     processor_instance=coffea_sumw(),
            #     executor=processor.futures_executor,
            #     executor_args={
            #         "schema" : nanoevents.BaseSchema,
            #         "workers": 8,
            #         "desc": "SumW"
            #     },
            # )

            ewk_flag = None
            if "ZZTo2L2Nu" in options.infile and "GluGluTo" not in options.infile and "ZZJJ" not in options.infile:
                ewk_flag= 'ZZ'
            if "WZTo" in options.infile and "GluGluTo" not in options.infile:
                ewk_flag = 'WZ'

        # vbs_out = processor.run_uproot_job(
        #     samples,
        #     processor_instance=zzinc_processor(
        #         era=options.era,
        #         ewk_process_name=ewk_flag,
        #         dump_gnn_array=options.dumpgnn
        #     ),
        #     treename='Events',
        #     executor=processor.futures_executor,
        #     executor_args={
        #         "schema" : nanoevents.NanoAODSchema,
        #         "workers": 8,
        #         "desc": "ZZinC"
        #     },
        #         #chunksize=50000,
        # )
            print(" --- gnn_input processor ... ")
            gnn_input = processor.run_uproot_job(
                samples,
                processor_instance=coffea_gnn_input(
                    era=options.era,
                    ewk_process_name=ewk_flag
                ),
                treename="Events",
                executor=processor.futures_executor,
                executor_args={
                    "schema" : nanoevents.NanoAODSchema,
                    "workers": 8,
                    "desc": "gnn_input"
                },
            )


            df=pd.DataFrame()
            for key, content in gnn_input[options.dataset].items():
                df[key] = content.value.tolist()
            
            # apply cuts for cutflow plot
            df['ewk_flag'] = ewk_flag

#            df = df[df['metfilter']==1]
#            df = df[df['ossf']==1]
#            df.drop('metfilter', inplace=True, axis=1)
#            df.drop('ossf', inplace=True, axis=1)

            #apply cuts for gnninput
            # print ('cuts for gnn input')
            # df = df[df['metfilter']==1]
            # df = df[df['ossf']==1]
            # df = df[df['met_pt']>100]
            # df = df[df['dijet_mass']>200]
            # df = df[(df['dilep_m']>81) & (df['dilep_m']<101)]
            # df = df[df['dijet_deta']>1.5]
            # df = df[df['ngood_bjets']<=1 & (df['ngood_bjets'] >=0)]
            # df.drop('metfilter', inplace=True, axis=1)
            # df.drop('ossf', inplace=True, axis=1)
            # df.drop('require-3lep', inplace=True, axis=1)
            # df.drop('genweight', inplace=True, axis=1)
            # print (df)

            # #apply cuts for gnn flatenning
            print ('cuts for gnn flatenning')
            df = df[df['metfilter']==1]
            df = df[df['ossf']==1]
            df = df[df['met_pt']>120]
            #df = df[df['dijet_mass']>200]
            df = df[(df['dilep_m']>76.1873) & (df['dilep_m']<106.1873)]
            df = df[(df['nhtaus'] == 0)]
            df = df[df['dijet_mass']>400]
            df = df[df['dijet_deta']>2.5]
            df = df[df['ngood_bjets']==0]
            df = df[df['ngood_jets']>=2]
            df = df[df['dilep_dphi_met']>1]
            df = df[df['dilep_dphi_met']>0.5]
            sel_dilep_pt = df.apply(lambda row: row['dilep_pt'] > 45 if row['require-3lep'] ==1 else row['dilep_pt'] > 60, axis=1)
            df = df[sel_dilep_pt]
            df.drop('require-3lep', inplace=True, axis=1)
            df.drop('metfilter', inplace=True, axis=1)
            df.drop('ossf', inplace=True, axis=1)
            print (df)

            print ("finished")
            df.to_parquet('df_%s.parquet' % str(options.jobNum)) 

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

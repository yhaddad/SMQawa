from coffea import processor
from coffea import nanoevents
from qawa.process.zz2l2nu_vbs import *
import yaml
import argparse
import pickle
import numpy as np
import gzip

def main():
    parser = argparse.ArgumentParser("")
    parser.add_argument('-jobs' , '--jobs'  , type=int, default=10    , help="")
    parser.add_argument('-era'    , '--era' , type=str, default="2018", help="")
    
    options  = parser.parse_args()

    samples = {}
    with open('./qawa/data/input-NanoAOD-2018UL-test.yaml') as s_file:
        samples = yaml.full_load(s_file)

    #trigger_sf_map = np.load('./data/trigger-sf-table-2018.npy')


    # this is the list of systematics used in 2l2nu analysis
    v_syst_list = [
            "ElectronEn", "MuonEn", "jesTotal", "jer"
    ]
    w_syst_list = [
            "puWeight", "PDF", "MuonSF", 
            "ElecronSF", "EWK", "nvtxWeight",
            "TriggerSF", "BTagSF",
            "QCDScale0", "QCDScale1", "QCDScale2"
    ]
    

    vbs_out = processor.run_uproot_job(
            samples,
            processor_instance=zzinc_processor(),
            treename='Events',
            executor=processor.futures_executor,
            executor_args={
                "schema": nanoevents.NanoAODSchema,
                "workers": 5,
            },
    )
    #
    # sumw_out = processor.run_uproot_job(
    #         samples,
    #         treename="Runs",
    #         processor_instance=coffea_sumw(),
    #         executor=processor.futures_executor,
    #         executor_args={
    #             "schema": nanoevents.BaseSchema,
    #             "workers": options.jobs
    #         },
    # )
    
    # append the sumw on the boost histograms
    bh_output = {}

    for key, content in vbs_out.items():
        bh_output[key] = {
            "hist": content,
            #"sumw": sumw_out[key]
    }

    with open("histogram-zz2l2nu-ristretto-test.pkl", "wb") as f:
        pickle.dump(bh_output, f)


if __name__ == "__main__":
    main()


from coffea import processor, nanoevents
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from qawa.process.trig_eff import trig_processor
from qawa.process.coffea_sumw import coffea_sumw
import argparse
import pickle
import gzip
import re, sys
import uproot
import numpy as np
import os

np.seterr(all='ignore')

uproot.open.defaults["xrootd_handler"] = uproot.source.xrootd.MultithreadedXRootDSource
uproot.open.defaults["timeout"] = 60 * 10  # wait more


def validate_input_file(nanofile):
    """Checks if a file is accessible through different XRootD aliases."""
    pfn = nanofile.strip()
    aliases = [
        "root://eoscms.cern.ch/",
        "root://xrootd-cms.infn.it/",
        "root://cmsxrootd.fnal.gov/",
        "root://cms-xrd-global.cern.ch/",
    ]

    for alias in aliases:
        try:
            with uproot.open(alias + pfn) as testfile:
                print(f'--> {alias} OK')
                return alias + pfn  # Return the first working alias
        except:
            print(f'--> {alias} FAILED')

    # If all fail, force the last alias
    return aliases[-1] + pfn


def process_file(options, file_name, alias):
    """Runs the Coffea processor on a single file."""
    samples = {
        options.dataset: {
            'files': [file_name],
            'metadata': {
                'era': options.era,
                'is_data': not options.isMC,
            }
        }
    }

    print(f"Processing file: {file_name}")

    # Run sum weights processor
    sumw_out = processor.run_uproot_job(
        samples,
        treename="Runs",
        processor_instance=coffea_sumw(),
        executor=processor.futures_executor,
        executor_args={"schema": nanoevents.BaseSchema, "workers": 8, "desc": "SumW"},
    )

    # Run trigger processor
    print("Instantiating trig_processor")
    trig_proc_instance = trig_processor(isMC=options.isMC, era=options.era)

    result = processor.run_uproot_job(
        samples,
        processor_instance=trig_proc_instance,
        treename='Events',
        #executor=processor.futures_executor,
        executor=processor.iterative_executor,
        executor_args={"schema": nanoevents.NanoAODSchema},
        #executor_args={"schema": nanoevents.NanoAODSchema, "workers": 8},
    )

    print(result)

    # Save result to a pkl file
    pkl_filename = f"histogram_{str(options.jobNum)}.pkl"
    with open(pkl_filename, "wb") as pkl_file:
        pickle.dump(result, pkl_file)

    print(f"Saved result to {pkl_filename}")
    return pkl_filename


def merge_results(pkl_files, output_filename="merged_results.pkl"):
    """Merges multiple pickled results into one."""
    merged_result = None

    for pkl_file in pkl_files:
        with open(pkl_file, "rb") as f:
            result = pickle.load(f)
            if merged_result is None:
                merged_result = result
            else:
                merged_result += result  # Assuming Coffea results support addition

    with open(output_filename, "wb") as f_out:
        pickle.dump(merged_result, f_out)

    print(f"Merged results saved to {output_filename}")


def main():
    parser = argparse.ArgumentParser("")
    parser.add_argument('--jobNum', type=int, default=1, help="")
    parser.add_argument('--era', type=str, default="2022", help="")
    parser.add_argument('--isMC', type=int, default=1, help="")
    parser.add_argument('--infile', type=str, default=None, help="input root file")
    parser.add_argument('--dataset', type=str, default=None, help="dataset name. Need to specify if file is not in EOS")

    options = parser.parse_args()

    if options.dataset is None:
        options.dataset = options.infile.split('/')[4]

    era = options.era
    is_data = not options.isMC

    aliases = [
        "root://eoscms.cern.ch/",
        "root://llrxrd-redir.in2p3.fr/",
        "root://xrootd-cms.infn.it/",
        "root://cms-xrd-global01.cern.ch/",
        "root://cms-xrd-global02.cern.ch/",
        "root://cmsxrootd.fnal.gov/",
        "root://xrootd-cms-redir-int.cr.cnaf.infn.it/",
        "root://xrootd-redic.pi.infn.it/"
    ]

    pkl_files = []
    for ixrd, alias in enumerate(aliases):
        try:
            file_name = options.infile
            if '/store/' in options.infile:
                file_name = alias + options.infile

            pkl_file = process_file(options, file_name, alias)
            pkl_files.append(pkl_file)

            # Break after a successful run
            break

        except Exception as e:
            print(f"[WARNING] {alias} failed with error: {e}")
            print("-------------------------------------------")

    if pkl_files:
        merge_results(pkl_files)


if __name__ == "__main__":
    main()

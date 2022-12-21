import re
import uproot
import numpy as np

np.seterr(all='ignore')

def _validate_input_file(nanofile):
    #forceaaa = False
    pfn = nanofile
    pfn=re.sub("\n","",pfn)
    aliases = [
        "root://eoscms.cern.ch/",
        "root://xrootd-cms.infn.it/",
        "root://cms-xrd-global.cern.ch/",
    ]
    #if (os.getenv("GLIDECLIENT_Group","") != "overflow" and os.getenv("GLIDECLIENT_Group","") != "overflow_conservative" and not forceaaa ):
    for alias in aliases:
        testfile = None
        try:
            testfile=uproot.open(alias + pfn)
        except:
            pass
        if testfile:
            nanofile=alias + pfn
            print(f'--> {alias} OK')
            break
        else:
            print(f'--> {alias} FAILD')
            #    if 'xrd-global' in alias:
            #        forceaaa = True
    return nanofile


def validate_input_file(nanofile):
    pfn = nanofile
    pfn=re.sub("\n","",pfn)
    aliases = [
        "root://eoscms.cern.ch/",
        "root://xrootd-cms.infn.it/",
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
    with open('./jobs_dune_DYJetsToLL_0J_TuneCP5_13TeV-amcatnloFXFX-pythia8/inputfiles.dat') as fn:
        for infile in fn.readlines():
            infile = re.sub("\n", "", infile)
            infile = validate_input_file(infile)
            print (f'   +: {infile}')
if __name__ == "__main__":
    main()

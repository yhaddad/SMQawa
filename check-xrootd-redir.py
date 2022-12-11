from socket import timeout
import uproot

fname = "/store/data/Run2018A/MuonEG/NANOAOD/UL2018_MiniAODv2_NanoAODv9-v1/280000/438DE06B-AF30-6241-B987-A82372053B0F.root"

redirsite = [
    #"root://xrootd.ba.infn.it/",
    "root://llrxrd-redir.in2p3.fr/",
    "root://xrootd-cms.infn.it/",
    "root://cms-xrd-global01.cern.ch/", 
    "root://cms-xrd-global02.cern.ch/",
    "root://cmsxrootd.fnal.gov/",
    "root://cmsxrootd2.fnal.gov/"
]

uproot.open.defaults["xrootd_handler"] = uproot.source.xrootd.MultithreadedXRootDSource
uproot.open.defaults["timeout"] = 60 * 5 # wait more
uproot.open.defaults["num_workers"] = 8


for site in redirsite:
    print('trying site:', site)
    try:
        options = {
            "xrootd_handler": uproot.source.xrootd.MultithreadedXRootDSource, 
            "num_workers": 8, 
            "timeout": 60 * 5
        }
        tree = uproot.open(site + fname)
        print('--> status: OK')
    except:
        print('--> status: FAILED')


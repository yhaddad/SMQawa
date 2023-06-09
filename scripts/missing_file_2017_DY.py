import os
import subprocess
from termcolor import colored

filelist=[
"DoubleEG_Run2017B-UL2017_MiniAODv2_NanoAODv9-v1",
"DoubleEG_Run2017C-UL2017_MiniAODv2_NanoAODv9-v1",
"DoubleEG_Run2017D-UL2017_MiniAODv2_NanoAODv9-v1",
"DoubleEG_Run2017E-UL2017_MiniAODv2_NanoAODv9-v1",
"DoubleEG_Run2017F-UL2017_MiniAODv2_NanoAODv9-v1",
"MuonEG_Run2017B-UL2017_MiniAODv2_NanoAODv9-v1",
"MuonEG_Run2017C-UL2017_MiniAODv2_NanoAODv9-v1",
"MuonEG_Run2017D-UL2017_MiniAODv2_NanoAODv9-v1",
"MuonEG_Run2017E-UL2017_MiniAODv2_NanoAODv9-v1",
"MuonEG_Run2017F-UL2017_MiniAODv2_NanoAODv9-v1",
"SingleElectron_Run2017B-UL2017_MiniAODv2_NanoAODv9-v1",
"SingleElectron_Run2017C-UL2017_MiniAODv2_NanoAODv9-v1",
"SingleElectron_Run2017D-UL2017_MiniAODv2_NanoAODv9-v1",
"SingleElectron_Run2017E-UL2017_MiniAODv2_NanoAODv9-v1",
"SingleElectron_Run2017F-UL2017_MiniAODv2_NanoAODv9-v1",
"SingleMuon_Run2017B-UL2017_MiniAODv2_NanoAODv9-v1",
"SingleMuon_Run2017C-UL2017_MiniAODv2_NanoAODv9-v1",
"SingleMuon_Run2017D-UL2017_MiniAODv2_NanoAODv9-v1",
"SingleMuon_Run2017E-UL2017_MiniAODv2_NanoAODv9-v1",
"SingleMuon_Run2017F-UL2017_MiniAODv2_NanoAODv9-v1",
"DoubleMuon_Run2017B-UL2017_MiniAODv2_NanoAODv9-v1",
"DoubleMuon_Run2017C-UL2017_MiniAODv2_NanoAODv9-v1",
"DoubleMuon_Run2017D-UL2017_MiniAODv2_NanoAODv9-v1",
"DoubleMuon_Run2017E-UL2017_MiniAODv2_NanoAODv9-v1",
"DoubleMuon_Run2017F-UL2017_MiniAODv2_NanoAODv9-v1",
"DYJetsToTauTau_M-50_AtLeastOneEorMuDecay_massWgtFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos",
"DYJetsToLL_0J_TuneCP5_13TeV-amcatnloFXFX-pythia8",
"DYJetsToLL_1J_TuneCP5_13TeV-amcatnloFXFX-pythia8",
"DYJetsToLL_2J_TuneCP5_13TeV-amcatnloFXFX-pythia8",
"DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8",
"DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8",
"DYJetsToLL_LHEFilterPtZ-0To50_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8",
"DYJetsToLL_LHEFilterPtZ-100To250_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8",
"DYJetsToLL_LHEFilterPtZ-250To400_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8",
"DYJetsToLL_LHEFilterPtZ-400To650_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8",
"DYJetsToLL_LHEFilterPtZ-50To100_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8",
"DYJetsToLL_LHEFilterPtZ-650ToInf_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8",
"DYJetsToLL_M-50_HT-70to100_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
"DYJetsToLL_M-50_HT-100to200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
"DYJetsToLL_M-50_HT-200to400_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
"DYJetsToLL_M-50_HT-400to600_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
"DYJetsToLL_M-50_HT-600to800_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
"DYJetsToLL_M-50_HT-800to1200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
"DYJetsToLL_M-50_HT-1200to2500_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
"DYJetsToLL_M-50_HT-2500toInf_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",
"GluGluToWWToENEN_TuneCP5_13TeV_MCFM701_pythia8",
"GluGluToWWToENMN_TuneCP5_13TeV_MCFM701_pythia8",
"GluGluToWWToENTN_TuneCP5_13TeV_MCFM701_pythia8",
"GluGluToWWToMNMN_TuneCP5_13TeV_MCFM701_pythia8",
"GluGluToWWToMNEN_TuneCP5_13TeV_MCFM701_pythia8",
"GluGluToWWToMNTN_TuneCP5_13TeV_MCFM701_pythia8",
"GluGluToWWToTNMN_TuneCP5_13TeV_MCFM701_pythia8",
"GluGluToWWToTNTN_TuneCP5_13TeV_MCFM701_pythia8",
"GluGluToContinToZZTo2e2mu_TuneCP5_13TeV-mcfm701-pythia8",
"GluGluToContinToZZTo2e2nu_TuneCP5_13TeV-mcfm701-pythia8",
"GluGluToContinToZZTo2e2tau_TuneCP5_13TeV-mcfm701-pythia8",
"GluGluToContinToZZTo2mu2nu_TuneCP5_13TeV-mcfm701-pythia8",
"GluGluToContinToZZTo2mu2tau_TuneCP5_13TeV-mcfm701-pythia8",
"GluGluToContinToZZTo4e_TuneCP5_13TeV-mcfm701-pythia8",
"GluGluToContinToZZTo4mu_TuneCP5_13TeV-mcfm701-pythia8",
"GluGluToContinToZZTo4tau_TuneCP5_13TeV-mcfm701-pythia8",
"ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8",
"ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8",
"ST_tW_antitop_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8",
"ST_tW_top_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8",
"ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8",
"tZq_ll_4f_ckm_NLO_TuneCP5_13TeV-amcatnlo-pythia8",
"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8",
"TTWJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8",
"TTWJetsToQQ_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8",
"TTZToLLNuNu_M-10_TuneCP5_13TeV-amcatnlo-pythia8",
"TTZToQQ_TuneCP5_13TeV-amcatnlo-pythia8",
"WWTo2L2Nu_TuneCP5_13TeV-powheg-pythia8",
"WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8",
"WZJJ_EWK_InclusivePolarization_TuneCP5_13TeV_madgraph-madspin-pythia8",
"ZZTo2L2Nu_TuneCP5_13TeV_powheg_pythia8",
"ZZJJ_ZZTo2L2Nu_EWK_dipoleRecoil_TuneCP5_13TeV-madgraph-pythia8",
"EWKZ2Jets_ZToLL_M-50_TuneCP5_withDipoleRecoil_13TeV-madgraph-pythia8",
"EWKWPlus2Jets_WToLNu_M-50_TuneCP5_withDipoleRecoil_13TeV-madgraph-pythia8",
"EWKWMinus2Jets_WToLNu_M-50_TuneCP5_withDipoleRecoil_13TeV-madgraph-pythia8",
"WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8",
"WWZ_4F_TuneCP5_13TeV-amcatnlo-pythia8",
"WZZ_TuneCP5_13TeV-amcatnlo-pythia8",
"ZZZ_TuneCP5_13TeV-amcatnlo-pythia8",
"VBS_OSWW_LL_TuneCP5_13TeV-madgraph-pythia8",
"VBS_OSWW_TT_TuneCP5_13TeV-madgraph-pythia8",
"VBS_OSWW_TL_TuneCP5_13TeV-madgraph-pythia8",
"VBS_OSWW_LT_TuneCP5_13TeV-madgraph-pythia8",
]


dirname="DYCR_210423"
pickle = "/afs/cern.ch/work/m/mmittal/private/VBS2l2nu/VBSCodeNew/CoffeaTools/1Jan2023/SMQawa/"+"jobs_"+dirname+"_2017_"
for list in filelist:

    pkl =   subprocess.check_output("ls -1 "+pickle+list+"/* | grep " + " pkl.gz " + "  | wc -l",shell=True)
    input = subprocess.check_output("cat "+ pickle+list+ "/inputfiles.dat  |   wc -l",shell=True)


    if(pkl == input):
        print(list , "while submitting on condor:", input, "outout pkl :", pkl, "number matches")
    else :
        print(colored('number differ','red'))
        print(list , "while submitting on condor:", colored(input,'red'), "outout pkl :", colored(pkl,'red'))
        
'''
vbs_plotting = "/afs/cern.ch/work/m/mmittal/private/VBS2l2nu/VBSCodeNew/CoffeaTools/1Jan2023/SMQawa/"+dirname+"/"
eos_prefix = "/eos/cms/store/user/mmittal/VBS/"+dirname+"/"
vbs_prefix = "jobs_"+dirname+"_2018_"


total_eos = subprocess.check_output("ls -1 "+ eos_prefix+ "* | grep " + " .root" +" | wc -l",shell=True)
total_vbs = subprocess.check_output("cat "+ vbs_prefix+ "*/inputfiles.dat  |   wc -l",shell=True)
total_plot = subprocess.check_output("ls -1 "+ vbs_plotting+ "* | grep "+ " .root" +" | wc -l",shell=True)
print("Total number of files:",total_eos,total_vbs,total_plot)



for list in filelist:
    eos_n =   subprocess.check_output("ls -1 "+eos_prefix+list+"  | wc -l",shell=True)
    vbs_n =   subprocess.check_output("cat "+vbs_prefix+list+"/inputfiles.dat | wc -l",shell=True)  
    # because of resubmit of job in the same dir they can be with different number abd that why it can be two in numbers
#    vbs_n =   subprocess.check_output("ls -1 "+vbs_prefix+list+ "/*.err | wc -l",shell=True)  



    #if(eos_n==vbs_n):
    #    print("In both",list ,": number matches")
    #else:
    #    print(colored('number differ  in EOS','red'))
    #    print(list , "in EOS: ", eos_n, "while submitting on condor:", vbs_n)

    Plotting=True
    if(Plotting):
        vbs_plotting_n = subprocess.check_output("ls -1 "+vbs_plotting+list+ "/*.root | wc -l",shell=True)
        if(eos_n==vbs_plotting_n):
            print("In both",list ,": number matches")
        else:
            print(colored('number differ  in EOS','green'))
            print(list , "in EOS: ", eos_n, "in VBS:", vbs_plotting_n)







'''

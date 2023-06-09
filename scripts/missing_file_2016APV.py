import os
import subprocess
from termcolor import colored

filelist=[
"SinglePhoton_Run2016B-ver2_HIPM_UL2016_MiniAODv2_NanoAODv9-v2",
"SinglePhoton_Run2016C-HIPM_UL2016_MiniAODv2_NanoAODv9-v4",
"SinglePhoton_Run2016D-HIPM_UL2016_MiniAODv2_NanoAODv9-v2",
"SinglePhoton_Run2016E-HIPM_UL2016_MiniAODv2_NanoAODv9-v2",
"SinglePhoton_Run2016F-HIPM_UL2016_MiniAODv2_NanoAODv9-v2",
"GJets_HT-40To100_TuneCP5_13TeV-madgraphMLM-pythia8",
"GJets_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8",
"GJets_HT-600ToInf_TuneCP5_13TeV-madgraphMLM-pythia8",
"GJets_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8",
"GJets_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8",
"ZNuNuGJets_MonoPhoton_PtG-130_TuneCP5_13TeV-amcatnloFXFX-pythia8",
"ZGTo2NuG_TuneCP5_13TeV-amcatnloFXFX-pythia8",
"ZGToLLG_01J_5f_TuneCP5_13TeV-amcatnloFXFX-pythia8",
"WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8",
"WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8",
"WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8",
"WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8",
"WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8",
"WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8",
"WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8",
"WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8",
"TGJets_TuneCP5_13TeV-amcatnlo-madspin-pythia8",
"TTGJets_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8",
"WGToLNuG_TuneCP5_13TeV-madgraphMLM-pythia8",
"WGToLNuG_01J_5f_PtG_130_TuneCP5_13TeV-amcatnloFXFX-pythia8",
"WGToLNuG_01J_5f_PtG_300_TuneCP5_13TeV-amcatnloFXFX-pythia8",
"WGToLNuG_01J_5f_PtG_500_TuneCP5_13TeV-amcatnloFXFX-pythia8",
]


dirname="final_PhCR_220323"
pickle = "/afs/cern.ch/work/m/mmittal/private/VBS2l2nu/VBSCodeNew/CoffeaTools/1Jan2023/SMQawa/"+"jobs_"+dirname+"_2016APV_"
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

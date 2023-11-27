import pandas as pd
import os
import yaml
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument("-e"   , "--era"   , type=str, default="2018"     , help="")
parser.add_argument("-d"   , "--DY"   , type=str, default="HT"     , help="")
parser.add_argument("-t"   , "--tag"   , type=str, default="oran"     , help="")
options = parser.parse_args()
era= options.era
DY = options.DY
tag = options.tag
lumi = {
		"2016" : 35.9,
		"2017" : 41.5,
		"2018" : 60.0
		}
#datapath=f'/afs/cern.ch/work/y/yixiao/SMQawa_gnn_input/jobs_{tag}_{era}_ZZJJ_ZZTo2L2Nu_EWK_dipoleRecoil_TuneCP5_13TeV-madgraph-pythia8'
#datapath=f'/eos/user/y/yixiao/ZZTo2L2Nu/{tag}/'
datapath=f'/afs/cern.ch/work/y/yixiao/SMQawa_gnn_input/'

jetBinnedDY = ['DYJetsToLL_0J_TuneCP5_13TeV-amcatnloFXFX-pythia8',
               'DYJetsToLL_1J_TuneCP5_13TeV-amcatnloFXFX-pythia8',
               'DYJetsToLL_2J_TuneCP5_13TeV-amcatnloFXFX-pythia8']
ptBinnedDY  = ['DYJetsToLL_LHEFilterPtZ-0To50_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8',
               'DYJetsToLL_LHEFilterPtZ-100To250_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8',
               'DYJetsToLL_LHEFilterPtZ-250To400_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8',
               'DYJetsToLL_LHEFilterPtZ-400To650_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8',
               'DYJetsToLL_LHEFilterPtZ-50To100_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8',
               'DYJetsToLL_LHEFilterPtZ-650ToInf_MatchEWPDG20_TuneCP5_13TeV-amcatnloFXFX-pythia8']
HTBinnedDY  = ['DYJetsToLL_M-50_HT-100to200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8',
               'DYJetsToLL_M-50_HT-1200to2500_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8',
               'DYJetsToLL_M-50_HT-200to400_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8',
               'DYJetsToLL_M-50_HT-2500toInf_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8',
               'DYJetsToLL_M-50_HT-400to600_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8',
               'DYJetsToLL_M-50_HT-600to800_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8',
               'DYJetsToLL_M-50_HT-70to100_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8',
               'DYJetsToLL_M-50_HT-800to1200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8']
totalDY     = ['DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8',
               'DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8'
               ]

def selectDY (dataset,DYsample):
	for i in DYsample:
		print(i)
		dataset.remove(i)
	return dataset

dataset=os.listdir(datapath)

dataset_remove=[]
for i in range(len(dataset)):
	if era == '2016':
		if 'jobs' in dataset[i] and f'{era}' in dataset[i] and 'APV' not in dataset[i]:
			dataset[i]='_'.join(dataset[i].split('_')[3:])
		else:
			dataset_remove.append(dataset[i])
	else:
		if 'jobs' in dataset[i] and f'{era}' in dataset[i]:
			dataset[i]='_'.join(dataset[i].split('_')[3:])
		else:
			dataset_remove.append(dataset[i])		

dataset = selectDY(dataset, dataset_remove)


if DY == 'HT':
	dataset = selectDY(dataset,jetBinnedDY)
	dataset = selectDY(dataset,ptBinnedDY)
	dataset = selectDY(dataset,totalDY)
elif DY =='PT':
	dataset = selectDY(dataset,jetBinnedDY)
	dataset = selectDY(dataset,HTBinnedDY)
	dataset = selectDY(dataset,totalDY)	
elif DY =='JET':
	dataset = selectDY(dataset,ptBinnedDY)
	dataset = selectDY(dataset,HTBinnedDY)
	dataset = selectDY(dataset,totalDY)
elif DY =='totalDY':
	dataset = selectDY(dataset,ptBinnedDY)
	dataset = selectDY(dataset,HTBinnedDY)
	dataset = selectDY(dataset,jetBinnedDY)
else:
	print ("error: DY sample selected")

# dataset=['ZZJJ_ZZTo2L2Nu_EWK_dipoleRecoil_TuneCP5_13TeV-madgraph-pythia8',
# 'ZZTo2E2Nu_TuneCP5_DipoleRecoil_13TeV_powheg_pythia8',
# 'ZZTo2Mu2Nu_TuneCP5_DipoleRecoil_13TeV_powheg_pythia8',
# ]

# dataset=[
# 'ZZTo2E2Nu_TuneCP5_DipoleRecoil_13TeV_powheg_pythia8',
# 'ZZTo2Mu2Nu_TuneCP5_DipoleRecoil_13TeV_powheg_pythia8',
# ]

#dataset=['ZZJJ_ZZTo2L2Nu_EWK_dipoleRecoil_TuneCP5_13TeV-madgraph-pythia8']

print (dataset)

#Bingran's old dataset
# dataset=['ZZJJ_ZZTo2L2Nu_EWK_dipoleRecoil_TuneCP5_13TeV-madgraph-pythia8',
# 'ZZTo2L2Nu_TuneCP5_13TeV_powheg_pythia8',
# #'ZZTo2L2Q_13TeV_amcatnloFXFX_madspin_pythia8',
# #'ZZTo4L_TuneCP5_13TeV_powheg_pythia8',
# 'GluGluToContinToZZTo2e2mu_TuneCP5_13TeV-mcfm701-pythia8',
# 'GluGluToContinToZZTo2e2nu_TuneCP5_13TeV-mcfm701-pythia8',
# 'GluGluToContinToZZTo2e2tau_TuneCP5_13TeV-mcfm701-pythia8',
# 'GluGluToContinToZZTo2mu2nu_TuneCP5_13TeV-mcfm701-pythia8',
# 'GluGluToContinToZZTo2mu2tau_TuneCP5_13TeV-mcfm701-pythia8',
# 'GluGluToContinToZZTo4e_TuneCP5_13TeV-mcfm701-pythia8',
# 'GluGluToContinToZZTo4mu_TuneCP5_13TeV-mcfm701-pythia8',
# 'GluGluToContinToZZTo4tau_TuneCP5_13TeV-mcfm701-pythia8',
# 'DYJetsToLL_M-50_HT-100to200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8',
# 'DYJetsToLL_M-50_HT-1200to2500_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8',
# 'DYJetsToLL_M-50_HT-200to400_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8',
# 'DYJetsToLL_M-50_HT-2500toInf_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8',
# 'DYJetsToLL_M-50_HT-400to600_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8',
# 'DYJetsToLL_M-50_HT-600to800_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8',
# 'DYJetsToLL_M-50_HT-70to100_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8',
# 'DYJetsToLL_M-50_HT-800to1200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8',
# 'WWTo2L2Nu_TuneCP5_13TeV-powheg-pythia8',
# # 'WZTo2L2Q_13TeV_amcatnloFXFX_madspin_pythia8',
# 'WZTo3LNu_TuneCP5_13TeV-amcatnloFXFX-pythia8',
# # 'WLLJJ_WToLNu_EWK_TuneCP5_13TeV_madgraph-madspin-pythia8',
# 'ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8',
# 'ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8',
# 'TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8',
# 'TTWJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8',
# 'TTWJetsToQQ_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8',
# 'TTZToLLNuNu_M-10_TuneCP5_13TeV-amcatnlo-pythia8',
# 'TTZToQQ_TuneCP5_13TeV-amcatnlo-pythia8',
# 'WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8',
# 'WWZ_4F_TuneCP5_13TeV-amcatnlo-pythia8',
# 'WZZ_TuneCP5_13TeV-amcatnlo-pythia8',
# 'ZZZ_TuneCP5_13TeV-amcatnlo-pythia8']


print (dataset)

with open(f'./Sumw_{era}.yaml',"r") as stream:
			readsumw=yaml.safe_load(stream)
#	readyaml[dataset_name][0]
with open('./xsections_2018.yaml',"r") as xsection:
			readxsec=yaml.safe_load(xsection)
df=pd.DataFrame()
for proc in dataset:
	print (proc)
	#procpath=f'/eos/user/y/yixiao/ZZTo2L2Nu/flattenGnn{era}/{proc}'
	#filename=os.listdir(procpath)
	filename = glob.glob(f"/afs/cern.ch/work/y/yixiao/SMQawa_gnn_input/jobs_{tag}_{era}_{proc}/df*.parquet")
	xsec = readxsec[proc]['xsec']
	kr = readxsec[proc]['kr']
	br = readxsec[proc]['br']
	sumw = readsumw[proc][0]
	print ("xsec")
	print (xsec)
	print ("sumw")
	print (sumw)
	if era == '2016APV':
		scale = (lumi['2016']/sumw)*xsec*kr*br*1000
	else:
		scale = (lumi[f'{era}']/sumw)*xsec*kr*br*1000
	for fn in filename:
		fn = fn.split('/')[-1]
		#filepath = f'/eos/user/y/yixiao/ZZTo2L2Nu/gnninput{era}/{proc}/{fn}'
		filepath = f'/afs/cern.ch/work/y/yixiao/SMQawa_gnn_input/jobs_{tag}_{era}_{proc}/{fn}'
		df_fn = pd.read_parquet(filepath, engine='pyarrow')
		df_fn["final_weight"] = df_fn["final_weight"]*scale
		df=pd.concat([df, df_fn], axis=0,ignore_index=True)
df.to_parquet(f'df_{era}_{tag}.parquet') 
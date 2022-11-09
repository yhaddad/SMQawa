import yaml
import uproot

with open('./qawa/data/datasetUL2018_2.yaml') as s_file:
	samples = yaml.full_load(s_file)
n=0
for i in samples:
	for j in samples[i]['files']:
		tree = uproot.open(j)
		print(n)
		n=n+1
#print (samples['ZZZ_TuneCP5_13TeV-amcatnlo-pythia8']['files'])
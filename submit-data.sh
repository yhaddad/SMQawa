#python brewer-htcondor.py -i ./data/dataset-UL2018_MiniAODv2_NanoAODv9.txt      -t amalfi --isMC=0 --era=2018
python brewer-htcondor.py -i ./data/dataset-UL2017_MiniAODv2_NanoAODv9.txt      -t amalfi --isMC=0 --era=2017 --force
python brewer-htcondor.py -i ./data/dataset-UL2016_MiniAODv2_NanoAODv9.txt      -t amalfi --isMC=0 --era=2016 --force 
python brewer-htcondor.py -i ./data/dataset-HIPM_UL2016_MiniAODv2_NanoAODv9.txt -t amalfi --isMC=0 --era=2016APV --force

# python brewer-htcondor-skim-tmp.py -i data/datasetUL2018-mc-skim.txt -t vbs --isMC=1 --era=2018
# python brewer-htcondor-skim-tmp.py -i data/datasetUL2018-data-skim.txt -t vbs --isMC=0 --era=2018
# python brewer-htcondor-skim-tmp.py -i data/datasetUL2017-mc-skim.txt -t vbs --isMC=1 --era=2017
# python brewer-htcondor-skim-tmp.py -i data/datasetUL2017-data-skim.txt -t vbs --isMC=0 --era=2017
# python brewer-htcondor-skim-tmp.py -i data/datasetUL2016-mc-skim.txt -t vbs --isMC=1 --era=2016
# python brewer-htcondor-skim-tmp.py -i data/datasetUL2016-data-skim.txt -t vbs --isMC=0 --era=2016
# python brewer-htcondor-skim-tmp.py -i data/datasetUL2016APV-mc-skim.txt -t vbs --isMC=1 --era=2016APV
# python brewer-htcondor-skim-tmp.py -i data/datasetUL2016APV-data-skim.txt -t vbs --isMC=0 --era=2016APV

# python brewer-htcondor.py -i data/datasetUL2018-mc-skim-ratio.txt -t vbs --isMC=1 --era=2016APV
# python brewer-htcondor.py -i data/datasetUL2018-mc-skim-ratio.txt -t vbs --isMC=1 --era=2016
# python brewer-htcondor.py -i data/datasetUL2018-mc-skim-ratio.txt -t vbs --isMC=1 --era=2017
# python brewer-htcondor.py -i data/datasetUL2018-mc-skim-ratio.txt -t vbs --isMC=1 --era=2018
# rm -rf 201*
python brewer-htcondor.py -i data/datasetUL2016APV-data-skim.txt -t vbs --isMC=0 --era=2016APV --dd=MC
python brewer-htcondor.py -i data/datasetUL2016-data-skim.txt -t vbs --isMC=0 --era=2016 --dd=MC
python brewer-htcondor.py -i data/datasetUL2017-data-skim.txt -t vbs --isMC=0 --era=2017 --dd=MC
python brewer-htcondor.py -i data/datasetUL2018-data-skim.txt -t vbs --isMC=0 --era=2018 --dd=MC
python brewer-htcondor.py -i data/datasetUL2016APV-mc-skim.txt -t vbs --isMC=1 --era=2016APV --dd=MC
python brewer-htcondor.py -i data/datasetUL2016-mc-skim.txt -t vbs --isMC=1 --era=2016 --dd=MC
python brewer-htcondor.py -i data/datasetUL2017-mc-skim.txt -t vbs --isMC=1 --era=2017 --dd=MC
python brewer-htcondor.py -i data/datasetUL2018-mc-skim.txt -t vbs --isMC=1 --era=2018 --dd=MC

python brewer-htcondor.py -i data/datasetUL2018-mc1.txt  -t vbs --isMC=1 --era=2016APV --dd=MC
python brewer-htcondor.py -i data/datasetUL2018-mc1.txt  -t vbs --isMC=1 --era=2016 --dd=MC
python brewer-htcondor.py -i data/datasetUL2018-mc1.txt  -t vbs --isMC=1 --era=2017 --dd=MC
python brewer-htcondor.py -i data/datasetUL2018-mc1.txt -t vbs --isMC=1 --era=2018 --dd=MC


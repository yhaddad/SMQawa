python setup.py check
python setup.py bdist_wheel --universal
python -m pip install \
    --no-deps \
    --ignore-installed \
    --no-cache-dir \
    dist/Qawa-0.0.6-py2.py3-none-any.whl

pip install . --upgrade --no-deps  --ignore-installed --no-cache-dir

scp dist/Qawa-0.0.6-py2.py3-none-any.whl /afs/cern.ch/work/m/mmittal/private/VBS2l2nu/VBSCodeNew/CoffeaTools/1Jan2023/SMQawa
source resub-script-9.sh  9 /store/data/Run2017B/SinglePhoton/NANOAOD/UL2017_MiniAODv2_NanoAODv9-v1/120000/CA053942-84ED-8F41-BE27-61822232FA42.root
#source resub-script-8.sh 8 /store/mc/RunIISummer20UL18NanoAODv9/GJets_HT-600ToInf_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_upgraderealistic_v16_L1v1-v1/2550000/F98E971A-9E41-0940-8B7F-9CA2F2CF31FF.root


#python brewer-local.py --jobNum=0 \
#  --isMC=0 --era=2018 \
#  --infile=/store/data/Run2018A/EGamma/NANOAOD/UL2018_MiniAODv1_NanoAODv2_GT36-v1/60000/7E31C206-C2AF-B340-BAD1-8D45231DA7EB.root

  

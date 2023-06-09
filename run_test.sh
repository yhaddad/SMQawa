#python setup.py check
#python setup.py bdist_wheel --universal
# python -m pip install \
#    --no-deps \
#    --ignore-installed \
#    --no-cache-dir \
#    dist/Qawa-0.0.5-py2.py3-none-any.whl

pip install . --upgrade --no-deps  --ignore-installed --no-cache-dir

python brewer-local.py --jobNum=0 \
  --isMC=1 --era=2017 \
  --infile=/store/mc/RunIISummer20UL17NanoAODv9/GJets_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/4cores5k_106X_mc2017_realistic_v9-v1/2560000/15563975-3AA2-D54D-A8AA-EC52DB7CF0A5.root
#  --infile=/store/data/Run2017B/SinglePhoton/NANOAOD/UL2017_MiniAODv2_NanoAODv9-v1/120000/4DEB63D4-999D-7946-AF29-BB04A0CBC247.root
#  --infile=/store/mc/RunIISummer20UL18NanoAODv9/ZZTo2L2Nu_TuneCP5_13TeV_powheg_pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/70000/691B674F-83CA#-0140-9754-14863CD3B950.root

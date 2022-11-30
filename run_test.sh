# python setup.py check
#python setup.py bdist_wheel --universal
# python -m pip install \
#    --no-deps \
#    --ignore-installed \
#    --no-cache-dir \
#    dist/Qawa-0.0.5-py2.py3-none-any.whl

pip install . --upgrade --no-deps  --ignore-installed --no-cache-dir

python brewer-local.py --jobNum=0 \
  --isMC=1 --era=2018 \
  --infile=/store/mc/RunIISummer20UL18NanoAODv9/ZZTo2L2Nu_TuneCP5_13TeV_powheg_pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/70000/691B674F-83CA-0140-9754-14863CD3B950.root

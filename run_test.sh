python setup.py check
python setup.py bdist_wheel --universal
python -m pip install \
    --no-deps \
    --ignore-installed \
    --no-cache-dir \
    dist/Qawa-0.0.6-py2.py3-none-any.whl

pip install . --upgrade --no-deps  --ignore-installed --no-cache-dir

python brewer-local.py --jobNum=0 \
  --isMC=0 --era=2018 \
  --infile=/store/data/Run2018A/EGamma/NANOAOD/UL2018_MiniAODv2_NanoAODv9-v1/270000/195CD88F-D50A-5340-B78D-A7B6345EBBFD.root

  

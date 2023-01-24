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
  --infile=/store/data/Run2018A/EGamma/NANOAOD/UL2018_MiniAODv1_NanoAODv2_GT36-v1/60000/7E31C206-C2AF-B340-BAD1-8D45231DA7EB.root

  

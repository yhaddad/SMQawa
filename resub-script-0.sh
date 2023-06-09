#!/bin/bash
export X509_USER_PROXY=/afs/cern.ch/user/m/mmittal/x509up_u6814

python -m venv --without-pip --system-site-packages jobenv
source jobenv/bin/activate
python -m pip install scipy --upgrade --no-cache-dir
python -m pip install --no-deps --ignore-installed --no-cache-dir Qawa-0.0.7-py2.py3-none-any.whl

echo "... start job at" `date "+%Y-%m-%d %H:%M:%S"`
echo "----- directory before running:"
echo "----- Found Proxy in: $X509_USER_PROXY"
ls -lthr
#xrdcp root://cms-xrd-global.cern.ch/$2 . 
python brewer-remote.py --jobNum=$1 --isMC=0 --era=2017 --infile=4BC53825-C4CC-264D-B251-87375F80A49E.root --dataset=DoubleEG
#rm 4BC53825-C4CC-264D-B251-87375F80A49E.root
ls -lthr


echo "----- directory after running :"
ls -lthr .
if [ ! -f "histogram_0.pkl.gz" ]; then
  exit 1;
fi
echo " ------ THE END (everyone dies !) ----- "

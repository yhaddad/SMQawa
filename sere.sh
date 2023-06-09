conda activate vbs
singularity shell -B ${PWD}:/srv/ /cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask:latest
cd /srv/
python setup.py build
python setup.py bdist_wheel --universal
#python -m pip install     --no-deps     --ignore-installed     --no-cache-dir     dist/Qawa-0.0.-py2.py3-none-any.whl

#!/usr/bin/env python

from dask_jobqueue.htcondor import HTCondorCluster
from dask.distributed import Client
import socket
import yaml
from termcolor import colored

def open_xrootd_files(sample, file):
    import uproot
    from termcolor import colored
    uproot.open.defaults["xrootd_handler"] = uproot.MultithreadedXRootDSource
    try:
        fn = uproot.open(file)
        fn.close()
        print(colored(sample + " -> [OK]", 'green'))
        return True
    except:
        print(colored(sample + " -> [BAD]", 'red'))
        return False


def socket_hostname():
    import socket
    print(' Yacine is in the condor .... ')
    return socket.gethostname()

cluster = HTCondorCluster(
    cores=1, # This should be forced to 1 core
    memory='2GB',
    disk='10GB',
    death_timeout = '10',
    nanny = False,
    scheduler_options={
        'port': 8786,
        'host': f"{socket.gethostname()}"
    },
    job_extra={
        'log'   : 'xrootd-dask-check-output.log',
        'output': 'xrootd-dask-check-output.out',
        'error' : 'xrootd-dask-check-output.err',
#        "use_x509userproxy"      : "true",
        'should_transfer_files'  : 'YES',
        'when_to_transfer_output': 'ON_EXIT',
        '+JobFlavour'            : '"tomorrow"',
    },
    extra = ['--worker-port 10000:10100']
)

print(' -----  script  ------- ')
print(cluster.job_script())

with open('./qawa/data/datasetUL2018_2.yaml') as s_file:
    samples = yaml.full_load(s_file)

#for sample, ct in samples.items():
#    file = ct['files'][0]
#    file = file.replace('cms-xrd-global.cern.ch', 'xrootd-cms.infn.it')
#    status = open_xrootd_files(sample, file)



with Client(cluster) as client:
    futures = []
    cluster.scale(10)

    import time
    maxinter = 10
    while len(client.ncores()) < 4:
        print('   -- waiting for more cores to spin up: {0} available'.format(len(client.ncores())))
        print('   -- Dask client info ->', client)
        time.sleep(10)
        if maxinter >= 10:
            break
        maxinter += 1


    for sample, ct in samples.items():
        file = ct['files'][0]
        file = file.replace('cms-xrd-global.cern.ch', 'xrootd-cms.infn.it')
        f = client.submit(open_xrootd_files, sample, file)
        futures.append(f)

    print('Result is {}'.format(client.gather(futures)))

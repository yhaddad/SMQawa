python brewer-htcondor.py -i ./data/input-NanoAOD-2018UL_data.txt     -t PhotonCR  --isMC=0 --era=2018 --force
python monitor.py -i ./data/input-NanoAOD-2018UL_data.txt -t PhotonCR --isMC=0 --era=2018 --resubmit --copyfile
python brewer-htcondor.py  -i ./data/list_2018_MC_Photon_UL.txt -t PhotonCR --isMC=1 --era=2018 --force
python monitor.py -i ./data/list_2018_MC_Photon_UL.txt -t PhotonCR --isMC=1 --era=2018 --resubmit --copyfile

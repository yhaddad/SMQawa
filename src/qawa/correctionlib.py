
# Nested  dictionary of the form dict[is_ultra_legacy][era][lumicorrection]
# From https://twiki.cern.ch/twiki/bin/view/CMS/BtagRecommendation#Recommendation_for_13_TeV_Data

_runIIUL_eras = (
    ("2016preVFP", "2016preVFP_UL"),
    ("2016postVFP", "2016postVFP_UL"),
    ("2017", "2017_UL"),
    ("2018", "2018_UL"),
)

_runIIUL_mapping = ( #for 
    # lookup name, POG, 
    ("ak8_xbbcc_tagging", "BTV", "ak8_xbbcc_tagging.json.gz"),
    ("btagging", "BTV", "btagging.json.gz"),
    ("ctagging", "BTV", "ctagging.json.gz"),
    ("qgtagging", "BTV", "qgtagging.json.gz"),
    ("subjet_btagging", "BTV", "subjet_btagging.json.gz"),
    ("subjet_btagging", "BTV", "subjet_btagging.json.gz"),

    ("muon_HighPt", "MUO", "muon_HighPt.json.gz"),
    ("muon_JPsi", "MUO", "muon_JPsi.json.gz"),
    ("muon_Z", "MUO", "muon_Z.json.gz"),

    ("electron", "EGM", "electron.json.gz"),
    ("photon", "EGM", "photon.json.gz"),

    # ("tau_embed", "TAU", "tau_embed.json.gz"), #Only exists for 2018, what is this anyway?
    ("tau", "TAU", "tau.json.gz"),

    ("fatJet_jerc", "JME", "fatJet_jerc.json.gz"),
    ("jet_jerc", "JME", "jet_jerc.json.gz"),
    ("jetvetomaps", "JME", "jetvetomaps.json.gz"),
    ("jmar", "JME", "jmar.json.gz"),
    ("met", "JME", "met.json.gz"),

    ("puWeights", "LUM", "puWeights.json.gz"),
)

_runIII_eras = (
    #### EXTRA 2022_27Jun2023EE and 2022_27Jun2023 eras, what are they for?
    ("2022preEE", "2022_Summer22"),
    ("2022postEE", "2022_Summer22EE"),
    ("2023preBPix", "2023_Summer23"),
    ("2023postBPix", "2023_Summer23BPix"),
    # 2024 is missing corrections for: TAU, MUO, LUM, BTV, JME (have only jet_jerc, jetid, jetvetomaps), EGM (
)
        
_runIII_mapping = ( #for 
    # lookup name, pog, 
    ("btagging", "BTV", "btagging.json.gz"),
    ("ctagging", "BTV", "ctagging.json.gz"),
    #("subjet_btagging", "BTV", "subjet_btagging.json.gz"), # Not available yet

    ("muon_HighPt", "MUO", "muon_HighPt.json.gz"),
    ("muon_JPsi", "MUO", "muon_JPsi.json.gz"),
    ("muon_Z", "MUO", "muon_Z.json.gz"),

    ("electron", "EGM", "electron.json.gz"),
    ("electronHlt", "EGM", "electronHlt.json.gz"),
    ("electronID_highPt", "EGM", "electronID_highPt.json.gz"),
    ("electronSS", "EGM", "electronSS.json.gz"),
    ("electronSS_EtDependent", "EGM", "electronSS_EtDependent.json.gz"),
    ("photon", "EGM", "photon.json.gz"),
    ("photonID_highPt", "EGM", "photonID_highPt.json.gz"),
    ("photonSS", "EGM", "photonSS.json.gz"),
    ("photonSS_EtDependent", "EGM", "photonSS_EtDependent.json.gz"),

    #("tau_embed", "TAU", "tau_embed.json.gz"), # Not available yet
    #("tau", "TAU", "tau.json.gz"), # REQUIRES SPECIAL CASING...

    ("fatJet_jerc", "JME", "fatJet_jerc.json.gz"),
    ("jet_jerc", "JME", "jet_jerc.json.gz"),
    ("jetvetomaps", "JME", "jetvetomaps.json.gz"),
    ("jetid", "JME", "jetid.json.gz"),
    #("jmar", "JME", "jmar.json.gz"), # Not available yet
    #("met", "JME", "met.json.gz"), # REQUIRES SPECIAL CASING...

    ("puWeights", "LUM", "puWeights.json.gz"),
)

CorrectionlibPathDict = {
    False: {myera: {name: f"/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/{pog}/{officialera}/{file}" for name, pog, file in _runIII_mapping} for myera, officialera in _runIII_eras}, #Not pro
    True:  {myera: {name: f"/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/{pog}/{officialera}/{file}" for name, pog, file in _runIIUL_mapping} for myera, officialera in _runIIUL_eras},
}

# Special cases because people couldn't stick to a pattern
CorrectionlibPathDict[False]["2022preEE"]["met"] = "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/2022_Summer22/met_xyCorrections_2022_2022.json.gz"
CorrectionlibPathDict[False]["2022postEE"]["met"] = "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/2022_Summer22EE/met_xyCorrections_2022_2022EE.json.gz"
CorrectionlibPathDict[False]["2023preBPix"]["met"] = "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/2023_Summer23/met_xyCorrections_2023_2023.json.gz"
CorrectionlibPathDict[False]["2023postBPix"]["met"] = "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/2023_Summer23BPix/met_xyCorrections_2023_2023BPix.json.gz"
CorrectionlibPathDict[False]["2022preEE"]["tau"] = "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/TAU/2022_Summer22/tau_DeepTau2018v2p5_2022_preEE.json.gz"
CorrectionlibPathDict[False]["2022postEE"]["tau"] = "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/TAU/2022_Summer22EE/tau_DeepTau2018v2p5_2022_postEE.json.gz"
CorrectionlibPathDict[False]["2023preBPix"]["tau"] = "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/TAU/2023_Summer23/tau_DeepTau2018v2p5_2023_preBPix.json.gz"
CorrectionlibPathDict[False]["2023postBPix"]["tau"] = "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/TAU/2023_Summer23BPix/tau_DeepTau2018v2p5_2023_postBPix.json.gz"

if "2024" not in CorrectionlibPathDict[False]:
    CorrectionlibPathDict[False]["2024"] = {}
CorrectionlibPathDict[False]["2024"]["electron"] = "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/EGM/2024_Summer24/electron_v1.json.gz"
CorrectionlibPathDict[False]["2024"]["electronID"] = "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/EGM/2024_Summer24/electronID_v1.json.gz"
CorrectionlibPathDict[False]["2024"]["electronSS_EtDependent"] = "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/EGM/2024_Summer24/electronSS_EtDependent_v1.json.gz"
CorrectionlibPathDict[False]["2024"]["photon_veto"] = "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/EGM/2024_Summer24/photon_veto_v1.json.gz"
CorrectionlibPathDict[False]["2024"]["photonID"] = "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/EGM/2024_Summer24/photonID_v1.json.gz"
CorrectionlibPathDict[False]["2024"]["photonSS_EtDependent"] = "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/EGM/2024_Summer24/photonSS_EtDependent_v1.json.gz"

# CorrectionlibPathDict[False]["2024"]["jet_jerc"] = "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/2024_Summer24/jet_jerc.json.gz"
# CorrectionlibPathDict[False]["2024"]["jetid"] = "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/2024_Summer24/jetid.json.gz"
# CorrectionlibPathDict[False]["2024"]["jetvetomaps"] = "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/2024_Summer24/jetvetomaps.json.gz"
CorrectionlibPathDict[False]["2024"]["jet_jerc"] = "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/2024_Winter24/jet_jerc.json.gz"
CorrectionlibPathDict[False]["2024"]["jetid"] = "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/2024_Winter24/jetid.json.gz"
CorrectionlibPathDict[False]["2024"]["jetvetomaps"] = "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/2024_Winter24/jetvetomaps.json.gz"

__all__ = ["CorrectionlibPathDict"]

if __name__ == "__main__":
    from pathlib import Path
    import rich
    print("isUltraLegacy\tEra\tCorrection\tPath")
    for isUltraLegacy in CorrectionlibPathDict:
        for thisera in CorrectionlibPathDict[isUltraLegacy]:
            for corrname in CorrectionlibPathDict[isUltraLegacy][thisera]:
                path = Path(CorrectionlibPathDict[isUltraLegacy][thisera][corrname])
                if path.exists():
                    rich.print(f"[green]{isUltraLegacy}\t{thisera}\t{corrname}\t{str(path)}")
                else:
                    rich.print(f"[red]{isUltraLegacy}\t{thisera}\t{corrname}\t{str(path)}")
                

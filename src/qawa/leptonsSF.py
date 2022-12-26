import os.path
from coffea.lookup_tools import extractor, dense_lookup
import awkward as ak
import numpy as np
import uproot 


class LeptonScaleFactors:
    def __init__(self, era:str='2018', isAPV:bool=False):
        if isAPV:
            self._era = era + 'APV'
        else:
            self._era = era 
        extLepSF = extractor()

        _data_path = os.path.join(os.path.dirname(__file__), 'data/lep')

        muonSelectionTag     = "LooseWP_" + self._era
        electronSelectionTag = "GPMVA90_" + self._era

        if muonSelectionTag=="LooseWP_2016":
            mu_f=["Efficiencies_muon_generalTracks_Z_Run2016_UL_SingleMuonTriggers.root",
                  "Efficiencies_muon_generalTracks_Z_Run2016_UL_ID.root",
                  "Efficiencies_muon_generalTracks_Z_Run2016_UL_ISO.root"]
            mu_h = ["NUM_IsoMu24_or_IsoTkMu24_DEN_CutBasedIdTight_and_PFIsoTight_abseta_pt",
                    "NUM_LooseID_DEN_TrackerMuons_abseta_pt",
                    "NUM_LooseRelIso_DEN_LooseID_abseta_pt"]
        elif muonSelectionTag=="LooseWP_2016APV":
            mu_f=["Efficiencies_muon_generalTracks_Z_Run2016_UL_HIPM_SingleMuonTriggers.root",
                  "Efficiencies_muon_generalTracks_Z_Run2016_UL_HIPM_ID.root",
                  "Efficiencies_muon_generalTracks_Z_Run2016_UL_HIPM_ISO.root"]
            mu_h = ["NUM_IsoMu24_or_IsoTkMu24_DEN_CutBasedIdTight_and_PFIsoTight_abseta_pt",
                    "NUM_LooseID_DEN_TrackerMuons_abseta_pt",
                    "NUM_LooseRelIso_DEN_LooseID_abseta_pt"]
        elif muonSelectionTag=="LooseWP_2017":
            mu_f=["Efficiencies_muon_generalTracks_Z_Run2017_UL_SingleMuonTriggers.root",
                  "Efficiencies_muon_generalTracks_Z_Run2017_UL_ID.root",
                  "Efficiencies_muon_generalTracks_Z_Run2017_UL_ISO.root"]
            mu_h = ["NUM_IsoMu27_DEN_CutBasedIdTight_and_PFIsoTight_abseta_pt",
                    "NUM_LooseID_DEN_TrackerMuons_abseta_pt",
                    "NUM_LooseRelIso_DEN_LooseID_abseta_pt"]
        elif muonSelectionTag=="LooseWP_2018":
            mu_f=["Efficiencies_muon_generalTracks_Z_Run2018_UL_SingleMuonTriggers.root",
                  "Efficiencies_muon_generalTracks_Z_Run2018_UL_ID.root",
                  "Efficiencies_muon_generalTracks_Z_Run2018_UL_ISO.root"]
            mu_h = ["NUM_IsoMu24_DEN_CutBasedIdTight_and_PFIsoTight_abseta_pt",
                    "NUM_LooseID_DEN_TrackerMuons_abseta_pt",
                    "NUM_LooseRelIso_DEN_LooseID_abseta_pt"]
        else:
            print (f'wrong era: {muonSelectionTag}')


        if electronSelectionTag=="GPMVA90_2016":
            el_f = ["egammaEffi_txt_Ele_wp90iso_postVFP_EGM2D.root"]
            el_h = ["EGamma_SF2D"]
        if electronSelectionTag=="GPMVA90_2016APV":
            el_f = ["egammaEffi_txt_Ele_wp90iso_preVFP_EGM2D.root"]
            el_h = ["EGamma_SF2D"]
        elif electronSelectionTag=="GPMVA90_2017":
            el_f = ["egammaEffi_txt_EGM2D_MVA90iso_UL17.root"]
            el_h = ["EGamma_SF2D"]
        elif electronSelectionTag=="GPMVA90_2018":
            el_f = ["egammaEffi_txt_Ele_wp90iso_EGM2D.root"]
            el_h = ["EGamma_SF2D"]
        else:
            print (f'wrong era: {electronSelectionTag}')
            
        self.maps_nom = {}
        self.maps_err = {}
        for i, _fname in enumerate(mu_f):
            with uproot.open(f"{_data_path}/{_fname}") as _fn:
                _hist = _fn[mu_h[i]].to_hist()
                _hnom = _hist.values()
                _herr = np.sqrt(_hist.variances())
                tag = f"MuonTri{era}" if "Trigger" in _fname else ""
                tag = f"MuonIso{era}" if "ISO" in _fname else tag
                tag = f"MuonId{era}"  if "ID" in _fname else tag
                
                self.maps_nom[tag] = dense_lookup.dense_lookup(_hnom,[ax.edges for ax in _hist.axes])
                self.maps_err[tag] = dense_lookup.dense_lookup(_herr,[ax.edges for ax in _hist.axes])

        for i, _fname in enumerate(el_f):
            with uproot.open(f"{_data_path}/{_fname}") as _fn:
                _hist = _fn[el_h[i]].to_hist()
                _hnom = _hist.values()
                _herr = np.sqrt(_hist.variances())
                tag = f"ElectronSF{era}" if "Ele" in _fname else ""
                
                self.maps_nom[tag] = dense_lookup.dense_lookup(_hnom,[ax.edges for ax in _hist.axes])
                self.maps_err[tag] = dense_lookup.dense_lookup(_herr,[ax.edges for ax in _hist.axes])


    def muonSF(self, muons: ak.Array):
        sf_nom  = 1.0 # ak.ones_like(muons.pt)
        sf_up   = 1.0 # ak.ones_like(muons.pt)
        sf_down = 1.0 # ak.ones_like(muons.pt)
        
        for n in self.maps_nom.keys():
            if 'Muon' not in n: continue
            _nom = self.maps_nom[n](muons.pt, np.abs(muons.eta))
            _err = self.maps_err[n](muons.pt, np.abs(muons.eta))
            print(n, _nom)
            sf_nom = sf_nom * _nom 
            sf_up = sf_up * (_nom + _err)
            sf_down = sf_down * (_nom - _err)
            
        return sf_nom, sf_up, sf_down

    def electronSF(self, electrons: ak.Array):
        sf_nom  = 1.0 # ak.ones_like(muons.pt)
        sf_up   = 1.0 # ak.ones_like(muons.pt)
        sf_down = 1.0 # ak.ones_like(muons.pt)
        
        for n in self.maps_nom.keys():
            if 'Electron' not in n: continue
            _nom = self.maps_nom[n](electrons.pt, np.abs(electrons.eta))
            _err = self.maps_err[n](electrons.pt, np.abs(electrons.eta))
            print(n, _nom)
            sf_nom = sf_nom * _nom 
            sf_up = sf_up * (_nom + _err)
            sf_down = sf_down * (_nom - _err)
            
        return sf_nom, sf_up, sf_down












import os.path
from coffea.lookup_tools import extractor, dense_lookup
import awkward as ak
import numpy as np


class LeptonScaleFactors:
    def __init__(self, era:str='2018'):
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
        elif muonSelectionTag=="LooseWP_2016_APV":
            mu_f=["Efficiencies_muon_generalTracks_Z_Run2016_UL_HIPM_SingleMuonTriggers.root",
                  "Efficiencies_muon_generalTracks_Z_Run2016_UL_HIPM_ID.root",
                  "Efficiencies_muon_generalTracks_Z_Run2016_UL_HIPM_ISO.root"]
            mu_h = ["NUM_IsoMu24_or_IsoTkMu24_DEN_CutBasedIdTight_and_PFIsoTight_abseta_pt",
                    "Efficiencies_muon_generalTracks_Z_Run2016_UL_ID.root",
                    "Efficiencies_muon_generalTracks_Z_Run2016_UL_ISO.root"]
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
        if electronSelectionTag=="GPMVA90_2016_APV":
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
            
        extLepSF.add_weight_sets([f'MuonTrig_{self._era} {mu_h[0]} {_data_path}/{mu_f[0]}'])
        extLepSF.add_weight_sets([f'MuonTrig_{self._era}_stat {mu_h[0]}_stat {_data_path}/{mu_f[0]}'])
        extLepSF.add_weight_sets([f'MuonTrig_{self._era}_syst {mu_h[0]}_syst {_data_path}/{mu_f[0]}'])
        extLepSF.add_weight_sets([f'MuonID_{self._era} {mu_h[1]} {_data_path}/{mu_f[1]}'])
        extLepSF.add_weight_sets([f'MuonID_{self._era}_stat {mu_h[1]}_stat {_data_path}/{mu_f[1]}'])
        extLepSF.add_weight_sets([f'MuonID_{self._era}_syst {mu_h[1]}_syst {_data_path}/{mu_f[1]}'])
        extLepSF.add_weight_sets([f'MuonISO_{self._era} {mu_h[2]} {_data_path}/{mu_f[2]}'])
        extLepSF.add_weight_sets([f'MuonISO_{self._era}_stat {mu_h[2]}_stat {_data_path}/{mu_f[2]}'])
        extLepSF.add_weight_sets([f'MuonISO_{self._era}_syst {mu_h[2]}_syst {_data_path}/{mu_f[2]}'])
        extLepSF.add_weight_sets([f'ElecSF_{self._era} {el_h[0]} {_data_path}/{el_f[0]}'])
        extLepSF.add_weight_sets([f'ElecSF_{self._era}_er {el_h[0]}_error {_data_path}/{el_f[0]}'])
        
        extLepSF.finalize()
        self.SFevaluator = extLepSF.make_evaluator()

    def AttachMuonSF(self, muons):

        eta = np.abs(muons.eta)
        pt = muons.pt

        trig_sf = self.SFevaluator[f'MuonTrig_{self._era}'](eta,pt)
        trig_sf_err = np.sqrt(self.SFevaluator[f'MuonTrig_{self._era}_stat'](eta,pt)*self.SFevaluator[f'MuonTrig_{self._era}_stat'](eta,pt) + self.SFevaluator[f'MuonTrig_{self._era}_syst'](eta,pt)*self.SFevaluator[f'MuonTrig_{self._era}_syst'](eta,pt))
        looseid_sf = self.SFevaluator[f'MuonID_{self._era}'](eta,pt)
        looseid_sf_err = np.sqrt(self.SFevaluator[f'MuonID_{self._era}_stat'](eta,pt)*self.SFevaluator[f'MuonID_{self._era}_stat'](eta,pt) + self.SFevaluator[f'MuonID_{self._era}_syst'](eta,pt)*self.SFevaluator[f'MuonID_{self._era}_syst'](eta,pt))
        iso_sf = self.SFevaluator[f'MuonISO_{self._era}'](eta,pt)
        iso_sf_err = np.sqrt(self.SFevaluator[f'MuonISO_{self._era}_stat'](eta,pt)*self.SFevaluator[f'MuonISO_{self._era}_stat'](eta,pt) + self.SFevaluator[f'MuonISO_{self._era}_syst'](eta,pt)*self.SFevaluator[f'MuonISO_{self._era}_syst'](eta,pt))

        muon_sf_nom = trig_sf * looseid_sf * iso_sf
        muon_sf_up = (trig_sf + trig_sf_err) * (looseid_sf + looseid_sf_err) * (iso_sf + iso_sf_err)
        muon_sf_down = (trig_sf - trig_sf_err) * (looseid_sf - looseid_sf_err) * (iso_sf - iso_sf_err)

        return muon_sf_nom, muon_sf_up, muon_sf_down


    def AttachElectronSF (self, electrons):

        eta = np.abs(electrons.eta)
        pt = electrons.pt

        elec_sf = self.SFevaluator[f'ElecSF_{self._era}']((eta),pt)
        elec_sf_err = self.SFevaluator[f'ElecSF_{self._era}']((eta),pt)

        elec_sf_nom = elec_sf
        elec_sf_up = elec_sf + elec_sf_err
        elec_sf_down = elec_sf - elec_sf_err

        return elec_sf_nom, elec_sf_up, elec_sf_down


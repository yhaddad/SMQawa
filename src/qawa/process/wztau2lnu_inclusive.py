import awkward as ak
import numpy as np
import scipy.interpolate as interp
from scipy import stats as st
from scipy.special import expit
import uproot
import pickle
import hist
import yaml
import os
import re
import onnxruntime as rt
import numpy.lib.recfunctions as rf
import uproot
import pickle
import numpy as np
import pandas as pd
import math

from coffea import processor
from coffea.nanoevents.methods import candidate

from coffea.analysis_tools import Weights, PackedSelection
from coffea.lumi_tools import LumiMask

from qawa.roccor import rochester_correction

from qawa.leptonsSF import LeptonScaleFactors
from qawa.jetPU import jetPUScaleFactors
from qawa.tauSF import tauIDScaleFactors
from qawa.btag import BTVCorrector, btag_id
from qawa.jme import JMEUncertainty, update_collection
from qawa.gen_match import find_best_match
from qawa.common import pileup_weights, ewk_corrector, met_phi_xy_correction, theory_ps_weight, theory_pdf_weight, trigger_rules




def build_leptons(muons, electrons):
    # select tight/loose muons
    tight_muons_mask = (
        (muons.pt             >  20. ) &
        (np.abs(muons.eta)    <  2.4 ) &
        (np.abs(muons.dxy)    <  0.045) &
        (np.abs(muons.dz )    <  0.2 ) &
        (muons.pfRelIso04_all <= 0.15) & 
        muons.tightId
    )
    tight_muons = muons[tight_muons_mask]
    loose_muons = muons[
        ~tight_muons_mask &
        (muons.pt            >  10. ) &
        (np.abs(muons.eta)   <  2.4 ) &
        (muons.pfRelIso04_all<= 0.25) &
        muons.softId   
    ]
    # select tight/loose electron
    tight_electrons_mask = (
        (electrons.pt           > 20.) &
        ((np.abs(electrons.eta) < 1.4442) | ((np.abs(electrons.eta) > 1.5660) & (np.abs(electrons.eta)  < 2.5)))  &
        electrons.mvaFall17V2Iso_WP90
    )
    tight_electrons = electrons[tight_electrons_mask]
    loose_electrons = electrons[
        ~tight_electrons_mask &
        (electrons.pt           > 10. ) &
        (np.abs(electrons.eta)  < 2.5) &
        electrons.mvaFall17V2Iso_WPL
    ]
    # contruct a lepton object
    tight_leptons = ak.with_name(ak.concatenate([tight_muons, tight_electrons], axis=1), 'PtEtaPhiMCandidate')
    loose_leptons = ak.with_name(ak.concatenate([loose_muons, loose_electrons], axis=1), 'PtEtaPhiMCandidate')

    return tight_leptons, loose_leptons

def build_htaus(tau, lepton):
    #print(dir(tau))
    #print(tau.__dict__)
    
    base_selection = (
        (tau.pt         > 20 ) & 
        (np.abs(tau.eta)< 2.3 ) &
        (np.abs(tau.dz)< 0.2 ) &
        (tau.decayMode != 5   ) & 
        (tau.decayMode != 6   ) &
        (tau.idDeepTau2017v2p1VSe >= 2) &
        (tau.idDeepTau2017v2p1VSmu >= 1) &
        (tau.idDeepTau2017v2p1VSjet >= 64)
    )

    overlap_leptons = ak.any(
        tau.metric_table(lepton) <= 0.4,
        axis=2
    )
   
    return tau[base_selection & ~overlap_leptons]

def build_htaus_tight(tau, lepton):
    #print(dir(tau))
    #print(tau.__dict__)
    
    base_selection = (
        (tau.pt         > 20 ) & 
        (np.abs(tau.eta)< 2.3 ) & 
        (np.abs(tau.dz)< 0.2 ) &
        (tau.decayMode != 5   ) & 
        (tau.decayMode != 6   ) &
        (tau.idDeepTau2017v2p1VSe >= 2) &
        (tau.idDeepTau2017v2p1VSmu >= 1) &
        (tau.idDeepTau2017v2p1VSjet >= 32)
    )

    overlap_leptons = ak.any(
        tau.metric_table(lepton) <= 0.4,
        axis=2
    )
    
    return tau[base_selection & ~overlap_leptons]

def build_htaus_loose(tau, lepton):
    #print(dir(tau))
    #print(tau.__dict__)
    
    base_selection = (
        (tau.pt         > 20 ) & 
        (np.abs(tau.eta)< 2.3 ) & 
        (np.abs(tau.dz)< 0.2 ) &
        (tau.decayMode != 5   ) & 
        (tau.decayMode != 6   ) &
        (tau.idDeepTau2017v2p1VSe >= 2) &
        (tau.idDeepTau2017v2p1VSmu >= 1) &
        (tau.idDeepTau2017v2p1VSjet >= 1)
    )

    overlap_leptons = ak.any(
        tau.metric_table(lepton) <= 0.4,
        axis=2
    )
    
    return tau[base_selection & ~overlap_leptons]
    

def build_photons(photon):
    base = (
        (photon.pt          > 20. ) & 
        (np.abs(photon.eta) < 2.5 )
    )
    # MVA ID
    tight_photons = photon[base & photon.mvaID_WP90]
    loose_photons = photon[base & photon.mvaID_WP80 & ~photon.mvaID_WP90]
    
    # cut based ID
    return tight_photons, loose_photons


class wzinclusive_processor(processor.ProcessorABC):
    # EWK corrections process has to be define before hand, it has to change when we move to dask
    def __init__(self, era: str ='2018', ewk_process_name=None, run_period: str = ''): 
        self._era = era
        if 'APV' in self._era:
            self._isAPV = True
            self._era = re.findall(r'\d+', self._era)[0] 
            #print(f"[YACINE DEBUG] era={self._era} APV={self._isAPV}")
        else:
            self._isAPV = False

        
        
        
        jec_tag = ''
        jer_tag = ''
        if len(run_period)==0:
            if self._era == '2016':
                if self._isAPV:
                    jec_tag = 'Summer19UL16APV_V7_MC'
                    jer_tag = 'Summer20UL16APV_JRV3_MC'
                else:
                    jec_tag = 'Summer19UL16_V7_MC'
                    jer_tag = 'Summer20UL16_JRV3_MC'
            elif self._era == '2017':
                jec_tag = 'Summer19UL17_V5_MC'
                jer_tag = 'Summer19UL17_JRV2_MC'
            elif self._era == '2018':
                jec_tag = 'Summer19UL18_V5_MC'
                jer_tag = 'Summer19UL18_JRV2_MC'
            else:
                print('error')
        else:
            if self._era == '2016':
                if self._isAPV:
                    if run_period in ['B', 'C', 'D']:
                        jec_tag = 'Summer19UL16APV_RunBCD_V7_DATA'
                    else:
                        jec_tag = 'Summer19UL16APV_RunEF_V7_DATA'
                else:
                    jec_tag = 'Summer19UL16_RunFGH_V7_DATA'
            elif self._era == '2017':
                jec_tag = f'Summer19UL17_Run{run_period}_V5_DATA'
            elif self._era == '2018':
                jec_tag = f'Summer19UL18_Run{run_period}_V5_DATA'
            else:
                print('error')
        
        self.btag_wp = 'M'
        self.jetPU_wp = 'M'
        self.tauIDvsjet_wp = 'VTight' #Medium is working
        self.tauIDvse_wp = 'VVLoose'
        self.tauIDvsmu_wp = 'VLoose'
        self.zmass = 91.1873 # GeV 
        self._btag = BTVCorrector(era=self._era, wp=self.btag_wp, isAPV=self._isAPV)
        self._jmeu = JMEUncertainty(jec_tag, jer_tag, era=self._era, is_mc=(len(run_period)==0))
        self._purw = pileup_weights(era=self._era)
        self._leSF = LeptonScaleFactors(era=self._era, isAPV=self._isAPV)
        self._jpSF = jetPUScaleFactors(era=self._era, wp=self.jetPU_wp, isAPV=self._isAPV)
        self._tauID= tauIDScaleFactors(era=self._era, vsjet_wp=self.tauIDvsjet_wp,vse_wp=self.tauIDvse_wp, vsmu_wp=self.tauIDvsmu_wp, isAPV=self._isAPV)

        _data_path = 'qawa/data'
        _data_path = os.path.join(os.path.dirname(__file__), '../data')
        self._json = {
            '2018': LumiMask(f'{_data_path}/json/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt'),
            '2017': LumiMask(f'{_data_path}/json/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt'),
            '2016': LumiMask(f'{_data_path}/json/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt'),
        }
        with open(f'{_data_path}/{self._era}-trigger-rules.yaml') as ftrig:
            self._triggers = yaml.load(ftrig, Loader=yaml.FullLoader)
            
        with open(f'{_data_path}/eft-names.dat') as eft_file:
            self._eftnames = [n.strip() for n in eft_file.readlines()]

        with uproot.open(f'{_data_path}/trigger_sf/histo_triggerEff_sel0_{self._era}.root') as _fn:
            _hvalue = np.dstack([_fn[_hn].values() for _hn in _fn.keys()] + [np.ones((7,7))])
            _herror = np.dstack([np.sqrt(_fn[_hn].variances()) for _hn in _fn.keys()] + [np.zeros((7,7))])
            self.trig_sf_map = np.stack([_hvalue, _herror], axis=-1)

        self.ewk_process_name = ewk_process_name
        if self.ewk_process_name is not None:
            self.ewk_corr = ewk_corrector(process=ewk_process_name)
       

        self.build_histos = lambda: {
            'dilep_mt': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(100, 0, 1000, name="dilep_mt", label=r"$M_{T}^{\ell\ell}$ (GeV)"),
                hist.storage.Weight()
            ), 
	        'dilep_pt': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(60, 0, 600, name="dilep_pt", label=r"$p_{T}^{\ell\ell}$ (GeV)"),
                hist.storage.Weight()
            ), 
	        'dilep_m': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(60, 0, 120, name="dilep_m", label=r"$M_{\ell\ell}$ (GeV)"),
                hist.storage.Weight()
            ), 
            'met_pt': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(60, 0, 600, name="met_pt", label=r"$p_{T}^{miss}$ (GeV)"),
                hist.storage.Weight()
            ),
            'm_T': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(60, 0, 600, name="m_T", label=r"$m_{T}$ (GeV)"),
                hist.storage.Weight()
            ),
            'm_T_WZ': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(60, 0, 600, name="m_T_WZ", label=r"$m_{T}^{WZ}$ (GeV)"),
                hist.storage.Weight()
            ),
            'mT_WZ': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(60, 0, 600, name="mT_WZ", label=r"$p4m_{T}^{WZ}$ (GeV)"),
                hist.storage.Weight()
            ),
            'emu_mT_WZ': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(60, 0, 600, name="emu_mT_WZ", label=r"$emu_{mT}^{WZ}$ (GeV)"),
                hist.storage.Weight()
            ),
            'tau_pt': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(60, 0, 600, name="tau_pt", label=r"$p_{T}^{tau}$ (GeV)"),
                hist.storage.Weight()
            ),
            'taus_eta': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, -5, 5, name="taus_eta", label=r"$\eta(\tau)$"),
                hist.storage.Weight()
            ),
            'taus_phi': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, 0, np.pi, name="taus_phi", label=r"$\phi(\tau)$"),
                hist.storage.Weight()
            ),
            'delta_tau_met_phi': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, 0, np.pi, name="delta_tau_met_phi", label=r"$\Delta \phi(\tau, p_{T}^{miss})$"),
                hist.storage.Weight()
            ),
            'tau_pt_loose': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(60, 0, 600, name="tau_pt_loose", label=r"$p_{T}^{tau_loose}$ (GeV)"),
                hist.storage.Weight()
            ),
            'bjets': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(5, 0, 5, name="bjets", label=r"$N_{b-jet}$ ($p_{T}>30$ GeV)"),
                hist.storage.Weight()
            ),
            'deep_tau_jet': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(100, 0, 1, name="deep_tau_jet", label=r"$deep_{\tau}^{jet}$"),
                hist.storage.Weight()
            ),
            'deep_tau_e': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(100, 0, 1, name="deep_tau_e", label=r"$deep_{\tau}^{e}$"),
                hist.storage.Weight()
            ),
            'deep_tau_mu': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(100, 0, 1, name="deep_tau_mu", label=r"$deep_{\tau}^{mu}$"),
                hist.storage.Weight()
            ),
            'dphi_met_ll': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, 0, np.pi, name="dphi_met_ll", label=r"$\Delta \phi(\ell\ell,p_{T}^{miss})$"),
                hist.storage.Weight()
            ),
            'dphi_jet_met': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, 0, np.pi, name="dphi_jet_met", label=r"$\Delta \phi(j,p_{T}^{miss})$"),
                hist.storage.Weight()
            ),
            'dilep_dphi_tau': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, 0, np.pi, name="dilep_dphi_tau", label=r"$\Delta \phi(\ell\ell,\tau)$"),
                hist.storage.Weight()
            ),
            'dilep_dphi': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, 0, np.pi, name="dilep_dphi", label=r"$\Delta \phi(\ell\ell)$"),
                hist.storage.Weight()
            ),
            'baseweight': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(400, -100, 100, name="baseweight", label=r"baseweight"),
                hist.storage.Weight()
            ),
            'delta_R': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, 0, np.pi, name="delta_R", label=r"$\Delta R$"),
                hist.storage.Weight()
            ),
            'delta_R_jet_dilep': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, 0, np.pi, name="delta_R_jet_dilep", label=r"$\Delta R(\ell\ell,j)$"),
                hist.storage.Weight()
            ),
            'delta_R_jet_tau': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, 0, np.pi, name="delta_R_jet_tau", label=r"$\Delta R(\tau,j)$"),
                hist.storage.Weight()
            ),
            'dilep_dR': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, 0, np.pi, name="dilep_dR", label=r"$\Delta R(\ell\ell)$"),
                hist.storage.Weight()
            ),
            "min_dphi_met_j": hist.Hist( 
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, 0, np.pi, name="min_dphi_met_j", label=r"$\min\Delta\phi(p_{T}^{miss},j)$"),
                hist.storage.Weight()
            ),
        }

    
    def _add_trigger_sf(self, weights, lead_lep, subl_lep):
        mask_BB = ak.fill_none((lead_lep.eta <= 1.5) & (subl_lep.eta <= 1.5), False)
        mask_EB = ak.fill_none((lead_lep.eta >= 1.5) & (subl_lep.eta <= 1.5), False)
        mask_BE = ak.fill_none((lead_lep.eta <= 1.5) & (subl_lep.eta >= 1.5), False)
        mask_EE = ak.fill_none((lead_lep.eta >= 1.5) & (subl_lep.eta >= 1.5), False)

        mask_mm = ak.fill_none((np.abs(lead_lep.pdgId)==13) & (np.abs(subl_lep.pdgId)==13), False)
        mask_ee = ak.fill_none((np.abs(lead_lep.pdgId)==11) & (np.abs(subl_lep.pdgId)==11), False)
       
        mask_me = (~mask_mm & ~mask_ee) & (np.abs(lead_lep.pdgId) == 13)
        mask_em = (~mask_mm & ~mask_ee) & (np.abs(lead_lep.pdgId) == 11)

        lept_pt_bins = [20, 25, 30, 35, 40, 50, 60, 100000]
        lep_1_bin = np.digitize(lead_lep.pt.to_numpy(), lept_pt_bins) - 1
        lep_2_bin = np.digitize(subl_lep.pt.to_numpy(), lept_pt_bins) - 1
        trigg_bin = np.select([
            (mask_ee & mask_BB).to_numpy(),
            (mask_ee & mask_BE).to_numpy(),
            (mask_ee & mask_EB).to_numpy(),
            (mask_ee & mask_EE).to_numpy(),

            (mask_em & mask_BB).to_numpy(),
            (mask_em & mask_BE).to_numpy(),
            (mask_em & mask_EB).to_numpy(),
            (mask_em & mask_EE).to_numpy(),

            (mask_me & mask_BB).to_numpy(),
            (mask_me & mask_BE).to_numpy(),
            (mask_me & mask_EB).to_numpy(),
            (mask_me & mask_EE).to_numpy(),

            (mask_mm & mask_BB).to_numpy(),
            (mask_mm & mask_BE).to_numpy(),
            (mask_mm & mask_EB).to_numpy(),
            (mask_mm & mask_EE).to_numpy()
        ], np.arange(0,16), 16)

        # this is to avoid cases were two 
        # leptons are not in the event
        lep_1_bin[lep_1_bin>6] = -1
        lep_2_bin[lep_2_bin>6] = -1
        center_value = self.trig_sf_map[lep_1_bin,lep_2_bin,trigg_bin,0]
        errors_value = self.trig_sf_map[lep_1_bin,lep_2_bin,trigg_bin,1]
        
        weights.add(
            'triggerSF', 
            center_value, 
            center_value + errors_value,
            center_value - errors_value
        )


    def process_shift(self, event, shift_name:str=''):
        _data_path = os.path.join(os.path.dirname(__file__), 'data/')
        dataset = event.metadata['dataset']
        is_data = event.metadata.get("is_data")
        selection = PackedSelection()
        weights = Weights(len(event), storeIndividual=True)
        
        histos = self.build_histos()
        
        if is_data:
            selection.add('lumimask', self._json[self._era](event.run, event.luminosityBlock))
            selection.add('triggers', trigger_rules(event, self._triggers, self._era))
        else:
            selection.add('lumimask', np.ones(len(event), dtype='bool'))
            selection.add('triggers', np.ones(len(event), dtype='bool'))
        
        # MET filters
        if is_data:
            selection.add(
                'metfilter',
                #event.Flag.METFilters &
                event.Flag.globalSuperTightHalo2016Filter & 
                event.Flag.HBHENoiseFilter &
                event.Flag.HBHENoiseIsoFilter & 
                event.Flag.EcalDeadCellTriggerPrimitiveFilter &
                event.Flag.goodVertices &
                event.Flag.eeBadScFilter &
                event.Flag.globalTightHalo2016Filter &
                event.Flag.BadChargedCandidateFilter & 
                event.Flag.BadPFMuonFilter
            )
        else:
            selection.add(
                'metfilter',
                #event.Flag.METFilters &
                event.Flag.globalSuperTightHalo2016Filter & 
                event.Flag.HBHENoiseFilter &
                event.Flag.HBHENoiseIsoFilter & 
                event.Flag.EcalDeadCellTriggerPrimitiveFilter & 
                event.Flag.goodVertices &
                event.Flag.eeBadScFilter &
                event.Flag.globalTightHalo2016Filter &
                event.Flag.BadChargedCandidateFilter & 
                event.Flag.BadPFMuonFilter
            )


        tight_lep, loose_lep = build_leptons(
            event.Muon,
            event.Electron
        )
        
        had_taus = build_htaus(event.Tau, tight_lep)
        had_taus_loose = build_htaus_loose(event.Tau, tight_lep)
        had_taus_tight = build_htaus_tight(event.Tau, tight_lep)
        ntight_lep = ak.num(tight_lep)
        nloose_lep = ak.num(loose_lep)
        nhtaus_lep = ak.num(had_taus)
        lead_tau = ak.firsts(had_taus)

        tau_pt = lead_tau.pt
        nhtaus_lep_loose = ak.num(had_taus_loose)
        nhtaus_lep_tight = ak.num(had_taus_tight)
        lead_tau_loose = ak.firsts(had_taus_loose)
        taus_phi = lead_tau.phi
        taus_eta = lead_tau.eta

        deep_tau_e = lead_tau.rawDeepTau2017v2p1VSe
        deep_tau_mu = lead_tau.rawDeepTau2017v2p1VSmu
        deep_tau_jet = lead_tau.rawDeepTau2017v2p1VSjet

        # print("deep_tau_e", deep_tau_e)
        # print("deep_tau_mu", deep_tau_mu)
        # print("deep_tau_jet", deep_tau_jet)

        
        jets = event.Jet
        overlap_leptons = ak.any(
            jets.metric_table(tight_lep) <= 0.4,
            axis=2
        )
        overlap_taus = ak.any(
            jets.metric_table(had_taus) <= 0.4,
            axis=2
        )
        
        jet_mask = (
            ~overlap_leptons & 
            ~overlap_taus &
            (jets.pt>30.0) & 
            (np.abs(jets.eta) < 4.7) & 
            (jets.jetId >= 6)& # tight JetID 7(2016) and 6(2017/8)
            (jets.puId >= 6)  # medium puID https://twiki.cern.ch/twiki/bin/viewauth/CMS/PileupJetIDUL
        )
        
        jet_btag = (
                event.Jet.btagDeepFlavB > btag_id(
                    self.btag_wp, 
                    self._era + 'APV' if self._isAPV else self._era
                )
        )
        
        good_jets = jets[~jet_btag & jet_mask]
        good_bjet = jets[jet_btag & jet_mask & (np.abs(jets.eta)<2.4)]
        
        ngood_jets  = ak.num(good_jets)
        ngood_bjets = ak.num(good_bjet)
        
        event['ngood_bjets'] = ngood_bjets
        event['ngood_jets']  = ngood_jets
       
        # lepton quantities
        def z_lepton_pair(leptons):
            pair = ak.combinations(leptons, 2, axis=1, fields=['l1', 'l2'])
            #lep_3 = ak.combinations(leptons, 3, axis=1, fields=['l1', 'l2', 't'])
            mass = (pair.l1 + pair.l2).mass
            #mass_llt = (pair.l1 + pair.l2 + pair.t).mass
            cand = ak.local_index(mass, axis=1) == ak.argmin(np.abs(mass - self.zmass), axis=1)

            extra_lepton = leptons[(
                ~ak.any(leptons.metric_table(pair[cand].l1) <= 0.01, axis=2) & 
                ~ak.any(leptons.metric_table(pair[cand].l2) <= 0.01, axis=2) )
            ]
            return pair[cand], extra_lepton, cand
        
        dilep, extra_lep, z_cand_mask = z_lepton_pair(tight_lep)
        
        lead_lep = ak.firsts(ak.where(dilep.l1.pt >  dilep.l2.pt, dilep.l1, dilep.l2),axis=1)
        subl_lep = ak.firsts(ak.where(dilep.l1.pt <= dilep.l2.pt, dilep.l1, dilep.l2),axis=1)
        
        dilep_p4 = (lead_lep + subl_lep)
        dilep_m  = dilep_p4.mass
        dilep_pt = dilep_p4.pt
        lead_tau = ak.firsts(had_taus)
        tau_pt = lead_tau.pt
        lead_tau_loose = ak.firsts(had_taus_loose)
        tau_pt_loose = lead_tau_loose.pt

        # high level observables
        p4_met = ak.zip(
            {
                "pt": event.MET.pt,
                "eta": ak.zeros_like(event.MET.pt),
                "phi": event.MET.phi,
                "mass": ak.zeros_like(event.MET.pt),
                "charge": ak.zeros_like(event.MET.pt),
            },
            with_name="PtEtaPhiMCandidate",
            behavior=candidate.behavior,
        )

        emu_met = ak.firsts(extra_lep, axis=1) + p4_met
        reco_met_pt = ak.where(ntight_lep==2, p4_met.pt, emu_met.pt)
        reco_met_phi = ak.where(ntight_lep==2, p4_met.phi, emu_met.phi)
        #print("reco_met = ", reco_met_pt)
        #print("reco_phi", reco_met_phi)

	
	    # this definition is not correct as it doesn't include the mass of the second Z
        dilep_et_ll = np.sqrt(dilep_pt**2 + dilep_m**2)
        dilep_et_met = np.sqrt(reco_met_pt**2 + self.zmass**2)
        dilep_mt = ak.where(
                ntight_lep==3,
                np.sqrt((dilep_et_ll + dilep_et_met)**2 - ((dilep_p4.pvec + emu_met.pvec).pt)**2),
                np.sqrt((dilep_et_ll + dilep_et_met)**2 - ((dilep_p4.pvec +  p4_met.pvec).pt)**2)
	    )
	
        dilep_dphi = lead_lep.delta_phi(subl_lep)
        dilep_deta = np.abs(lead_lep.eta - subl_lep.eta)
        dilep_dR   = lead_lep.delta_r(subl_lep)

        delta_R = ak.where(ntight_lep==2, dilep_p4.delta_r(lead_tau), dilep_p4.delta_r(lead_tau))
        dilep_dphi_met  = ak.where(ntight_lep==2, dilep_p4.delta_phi(p4_met), dilep_p4.delta_phi(emu_met))
        #scalar_balance = ak.where(ntight_lep==3, emu_met.pt/dilep_p4.pt, p4_met.pt/dilep_p4.pt)
        delta_tau_met_phi = ak.where(ntight_lep==2, lead_tau.delta_phi(p4_met), lead_tau.delta_phi(emu_met))
        dilep_dphi_tau = ak.where(ntight_lep==2, dilep_p4.delta_phi(lead_tau), dilep_p4.delta_phi(lead_tau))
        delta_tau_loose_met_phi = ak.where(ntight_lep==2, lead_tau_loose.delta_phi(p4_met), lead_tau_loose.delta_phi(emu_met))
        dilep_dphi_tau_loose = ak.where(ntight_lep==2, dilep_p4.delta_phi(lead_tau_loose), dilep_p4.delta_phi(lead_tau_loose))
        m_T = np.sqrt((2*tau_pt*reco_met_pt)*(1-np.cos(delta_tau_met_phi)))
        dilep_tau_pt = (dilep_pt+tau_pt)
        dilep_loose_tau_pt = (dilep_pt+tau_pt_loose)
        dilep_tau = dilep_p4+lead_tau
        dilep_loose_tau = dilep_p4+lead_tau_loose
        dilep_tau_met_dphi = ak.where(ntight_lep==2, dilep_tau.delta_phi(p4_met), dilep_tau.delta_phi(emu_met))
        dilep_loose_tau_met_dphi = ak.where(ntight_lep==2, dilep_loose_tau.delta_phi(p4_met), dilep_loose_tau.delta_phi(emu_met))
        m_T_WZ = np.sqrt((2*dilep_tau_pt*reco_met_pt)*(1-np.cos(dilep_tau_met_dphi)))
        emu_mT_WZ = np.sqrt((2*dilep_loose_tau_pt*reco_met_pt)*(1-np.cos(dilep_loose_tau_met_dphi)))

        #mT_WZ = (dilep_p4 + lead_tau + p4_met).mt
        #mass_T = ak.to_num(m_T)
        #print("m_T = ", mass_T)

        
        # 2jet and vbs related variables
        lead_jet = ak.firsts(good_jets)
        subl_jet = ak.firsts(good_jets[lead_jet.delta_r(good_jets)>0.01])
        third_jet = ak.firsts(good_jets[(lead_jet.delta_r(good_jets)>0.01) & (subl_jet.delta_r(good_jets)>0.01)])
        delta_R_jet_dilep = ak.where(ntight_lep==2, dilep_p4.delta_r(lead_jet), dilep_p4.delta_r(lead_jet))
        delta_R_jet_tau = ak.where(ntight_lep==2, lead_tau.delta_r(lead_jet), lead_tau.delta_r(lead_jet))
        dphi_jet_met = ak.where(ntight_lep==2, lead_jet.delta_phi(p4_met), lead_jet.delta_phi(emu_met))
        
        dijet_mass = (lead_jet + subl_jet).mass
        dijet_deta = np.abs(lead_jet.eta - subl_jet.eta)
        event['dijet_mass'] = dijet_mass
        event['dijet_deta'] = dijet_deta 
        #dijet_zep1 = np.abs(2*lead_lep.eta - (lead_jet.eta + subl_jet.eta))/dijet_deta
        #dijet_zep2 = np.abs(2*subl_lep.eta - (lead_jet.eta + subl_jet.eta))/dijet_deta
        
        min_dphi_met_j = ak.min(np.abs(
            ak.where(
                ntight_lep==3, 
                good_jets.delta_phi(emu_met), 
                good_jets.delta_phi(p4_met)
            )
        ), axis=1)

        event['min_dphi_met_j'] = min_dphi_met_j
        
        # define basic selection
        selection.add(
            "require-ossf",
            (ntight_lep==2) & (nloose_lep==0) &
            (ak.firsts(tight_lep).pt>25) &
            ak.fill_none((lead_lep.pdgId + subl_lep.pdgId)==0, False)
        )
        
        selection.add(
            "require-osof",
            (ntight_lep==2) & (nloose_lep==0) &
            (ak.firsts(tight_lep).pt>25) &
           ak.fill_none(np.abs(lead_lep.pdgId) != np.abs(subl_lep.pdgId), False)
        )
        
        selection.add(
            "require-2lep",
            (ntight_lep==2) & (nloose_lep==0) &
            (ak.firsts(tight_lep).pt>25) &
            ak.fill_none((lead_lep.pdgId + subl_lep.pdgId)==0, False)
        )

        selection.add(
            "require-3lep",
            (ntight_lep==3) & (nloose_lep==0) &
            (ak.firsts(tight_lep).pt>25) &
            ak.fill_none((lead_lep.pdgId + subl_lep.pdgId)==0, False)
        )
        
        selection.add(
            "require-4lep",
            (ntight_lep>=2) & (nloose_lep + ntight_lep)==4 &
            (ak.firsts(tight_lep).pt>25) &
            ak.fill_none((lead_lep.pdgId + subl_lep.pdgId)==0, False)
        )

        selection.add('met_pt', ak.fill_none((reco_met_pt > 30), False))
        selection.add('low_met_pt', ak.fill_none((reco_met_pt < 20) & (reco_met_pt > 0), False))
        selection.add('dilep_m'   , ak.fill_none(np.abs(dilep_m - self.zmass) < 15, False))
        selection.add('mT_40', ak.fill_none(m_T > 40, False))
        selection.add('dilep_pt', ak.fill_none(dilep_pt > 30, False))
        selection.add("dilep_dphi_met", ak.fill_none(np.abs(dilep_dphi_met)>1.0, False))
        selection.add("dilep_dphi_tau", ak.fill_none(np.abs(dilep_dphi_tau)>1.0, False))
        selection.add("delta_tau_met_phi", ak.fill_none(np.abs(delta_tau_met_phi)>1.0, False))
        selection.add("dilep_dphi_tau_loose", ak.fill_none(np.abs(dilep_dphi_tau_loose)>1.0, False))
        selection.add("delta_tau_loose_met_phi", ak.fill_none(np.abs(delta_tau_loose_met_phi)>1.0, False))
        selection.add(
            "min_dphi_met_j",
            ak.where(
                ngood_jets <= 1, 
                ak.fill_none(np.abs(min_dphi_met_j)>0.25, False), 
                ak.fill_none(np.abs(min_dphi_met_j)>0.5, False), 
            )
        )
        # jet demography
        selection.add('0njets' , ngood_jets  == 0 )
        selection.add('1njets' , ngood_jets  <= 1 )
        selection.add('1njets_only' , ngood_jets  == 1 )
        selection.add('2njets' , ngood_jets  <= 2 )
        selection.add('3njets' , ngood_jets  >= 3 )
        selection.add('1nbjets', ngood_bjets >= 1 )
        selection.add('1nhtaus', nhtaus_lep  == 1 )
        selection.add('1nhtaus_loose', nhtaus_lep_loose  == 1 )
        selection.add('1nhtaus_tight', nhtaus_lep_tight  == 1 )
        
        selection.add('dijet_deta', ak.fill_none(dijet_deta > 2.5, False))
        selection.add('dijet_mass_400' , ak.fill_none(dijet_mass >  400, False))
        selection.add('m_T_cut' , ak.fill_none((m_T > 50) & (m_T < 70), False))
        selection.add('dijet_mass_800' , ak.fill_none(dijet_mass >  800, False))

        # Define all variables for the BDT
        event['met_pt'  ] = reco_met_pt
        event['m_T'  ] = m_T
        event['m_T_WZ'  ] = m_T_WZ
        event['emu_mT_WZ'  ] = emu_mT_WZ
        event['met_phi' ] = reco_met_phi
        event['dilep_mt'] = dilep_mt
        event['dilep_m'] = dilep_m
        event['dilep_pt'] = dilep_pt
        event['dilep_dphi'] = dilep_dphi
        event['njets'   ] = ngood_jets
        event['bjets'   ] = ngood_bjets
        event['dphi_met_ll'] = dilep_dphi_met
        event['dilep_dphi_tau'] = dilep_dphi_tau
        event['dijet_mass'] = dijet_mass
        event['dijet_deta'] = dijet_deta
        event['min_dphi_met_j'] = min_dphi_met_j
        event['tau_pt'] = tau_pt
        event['taus_phi'] = taus_phi
        event['taus_eta'] = taus_eta
        event['delta_R'] = delta_R
        event['dilep_dR'] = dilep_dR
        event['delta_tau_met_phi'] = delta_tau_met_phi
        event['tau_pt_loose'] = tau_pt_loose
        event['leading_lep_pt'  ] = lead_lep.pt
        event['leading_lep_eta' ] = lead_lep.eta
        event['leading_lep_phi' ] = lead_lep.phi
        event['trailing_lep_pt' ] = subl_lep.pt
        event['trailing_lep_eta'] = subl_lep.eta
        event['trailing_lep_phi'] = subl_lep.phi       
        event['lead_jet_pt'  ] = lead_jet.pt
        event['lead_jet_eta' ] = lead_jet.eta
        event['lead_jet_phi' ] = lead_jet.phi
        event['trail_jet_pt' ] = subl_jet.pt
        event['trail_jet_eta'] = subl_jet.eta
        event['trail_jet_phi'] = subl_jet.phi
        event['third_jet_pt' ] = third_jet.pt
        event['third_jet_eta'] = third_jet.eta
        event['third_jet_phi'] = third_jet.phi
        event['deep_tau_jet'] = deep_tau_jet
        event['deep_tau_e'] = deep_tau_e
        event['deep_tau_mu'] = deep_tau_mu
        event['delta_R_jet_tau'] = delta_R_jet_tau
        event['delta_R_jet_dilep'] = delta_R_jet_dilep
        event['dphi_jet_met'] = dphi_jet_met
        


        # Now adding weights
        if not is_data:
            weights.add('genweight', event.genWeight)
            self._btag.append_btag_sf(jets, weights)
            self._jpSF.append_jetPU_sf(jets, weights)
            self._purw.append_pileup_weight(weights, event.Pileup.nPU)
            self._tauID.append_tauID_sf(had_taus, weights)
            self._add_trigger_sf(weights, lead_lep, subl_lep)
            
            weights.add (
                    'LeptonSF', 
                    lead_lep.SF*subl_lep.SF, 
                    lead_lep.SF_up*subl_lep.SF_up, 
                    lead_lep.SF_down*subl_lep.SF_down
            )
            _ones = np.ones(len(weights.weight()))
            
            if self.ewk_process_name:
                self.ewk_corr.get_weight(
                        event.GenPart,
                        event.Generator.x1,
                        event.Generator.x2,
                        weights
                )
            else:
                weights.add("kEW", _ones, _ones, _ones)
            
            if "PSWeight" in event.fields:
                theory_ps_weight(weights, event.PSWeight)
            else:
                theory_ps_weight(weights, None)
            
            if "LHEPdfWeight" in event.fields:
                theory_pdf_weight(weights, event.LHEPdfWeight)
            else:
                theory_pdf_weight(weights, None)

            if ('LHEScaleWeight' in event.fields) and (len(event.LHEScaleWeight[0]) > 0):
                if len(event.LHEScaleWeight[0]) == 9:
                    weights.add('QCDScale0w'  , _ones, event.LHEScaleWeight[:, 1], event.LHEScaleWeight[:, 7])
                    weights.add('QCDScale1w'  , _ones, event.LHEScaleWeight[:, 3], event.LHEScaleWeight[:, 5])
                    weights.add('QCDScale2w'  , _ones, event.LHEScaleWeight[:, 0], event.LHEScaleWeight[:, 8])
                elif len(event.LHEScaleWeight[0]) == 8:
                    weights.add('QCDScale0w'  , _ones, event.LHEScaleWeight[:, 1], event.LHEScaleWeight[:, 6])
                    weights.add('QCDScale1w'  , _ones, event.LHEScaleWeight[:, 3], event.LHEScaleWeight[:, 4])
                    weights.add('QCDScale2w'  , _ones, event.LHEScaleWeight[:, 0], event.LHEScaleWeight[:, 7])
                elif len(event.LHEScaleWeight[0]) == 18:
                    weights.add('QCDScale0w'  , _ones, event.LHEScaleWeight[:, 2], event.LHEScaleWeight[:, 14])
                    weights.add('QCDScale1w'  , _ones, event.LHEScaleWeight[:, 6], event.LHEScaleWeight[:, 10])
                    weights.add('QCDScale2w'  , _ones, event.LHEScaleWeight[:, 0], event.LHEScaleWeight[:, 16])
                else:
                    print("WARNING: QCD scale variation type not recongnised ... ")
                
            if 'LHEReweightingWeight' in event.fields and 'aQGC' in dataset:
                for i in range(1057):
                    weights.add(f"eft_{self._eftnames[i]}", event.LHEReweightingWeight[:, i])
            
            # 2017 Prefiring correction weight
            if 'L1PreFiringWeight' in event.fields:
                weights.add("prefiring_weight", event.L1PreFiringWeight.Nom, event.L1PreFiringWeight.Dn, event.L1PreFiringWeight.Up)

        # selections (delta_tau_met_phi cut is removed from SR)

        common_sel = ['triggers', 'lumimask', 'metfilter']
        channels = {
            "inc-SR0": common_sel + [
		    'require-ossf', 'require-2lep', 'dilep_m', 'dilep_dphi_met', '0njets', '~1nbjets', '1nhtaus', 'met_pt', 'dilep_pt'
        ],
            "inc-SR1": common_sel + [
            'require-ossf', 'require-2lep', 'dilep_m', 'dilep_dphi_met', '1njets_only', '~1nbjets', '1nhtaus' ,'met_pt', 'dilep_pt'
        ],
            "inc-SR_new": common_sel + [
            'require-ossf', 'require-2lep', 'dilep_m', 'dilep_dphi_met', '1njets', '~1nbjets', '1nhtaus', 'met_pt', 'dilep_pt'
        ],
            "inc-DY0": common_sel + [
		    'require-ossf', 'require-2lep', 'dilep_m', 'dilep_dphi_met', 'met_pt','dilep_pt', '0njets',  '~1nhtaus', '1nhtaus_loose', '~1nbjets', '~1nhtaus_tight'
        ],
            "inc-DY1": common_sel + [
            'require-ossf', 'require-2lep', 'dilep_m', 'dilep_dphi_met', 'met_pt','dilep_pt', '1njets_only', '~1nhtaus', '1nhtaus_loose', '~1nbjets', '~1nhtaus_tight'
	    ],
            "inc-EM": common_sel + [
		    'require-osof', 'dilep_m', 'dilep_pt', 'dilep_dphi_met', '1nhtaus', 'delta_tau_met_phi', 'met_pt'
        ],

            "inc-B": common_sel + [
            'require-ossf', 'require-2lep', 'dilep_m', 'dilep_pt', 'delta_tau_met_phi', '1nhtaus_loose', '~1nbjets', 'met_pt', '~1njets'
        ],

            "inc-C": common_sel + [
            'require-ossf', 'require-2lep', 'dilep_m', 'dilep_pt', 'dilep_dphi_met', 'delta_tau_met_phi', 'low_met_pt', '1nhtaus_loose', '~1nbjets', '~1njets'
        ],

            "inc-D": common_sel + [
            'require-ossf', 'require-2lep', 'dilep_m', 'dilep_pt', 'dilep_dphi_met', 'delta_tau_met_phi', 'low_met_pt', '1nhtaus', '~1nbjets', '~1njets'
        ],
        }

        if shift_name is None:
            systematics = [None] + list(weights.variations)
        else:
            systematics = [shift_name]
            
        def _format_variable(variable, cut):
            if cut is None:
                vv = ak.to_numpy(ak.fill_none(variable, np.nan))
                if np.isnan(np.any(vv)):
                    print(" - vv with nan:", vv)
                return ak.to_numpy(ak.fill_none(variable, np.nan))
            else:
                vv = ak.to_numpy(ak.fill_none(variable[cut], np.nan))
                if np.isnan(np.any(vv)):
                    print(" - vv with nan:", vv)
                return ak.to_numpy(ak.fill_none(variable[cut], np.nan))
        
        def _histogram_filler(ch, syst, var, _weight=None):
            sel_ = channels[ch]
            sel_args_ = {
                s.replace('~',''): (False if '~' in s else True) for s in sel_ if var not in s
            }
            cut =  selection.require(**sel_args_)

            systname = 'nominal' if syst is None else syst
            
            if _weight is None: 
                if syst in weights.variations:
                    weight = weights.weight(modifier=syst)[cut]
                else:
                    weight = weights.weight()[cut]
            else:
                weight = weights.weight()[cut] * _weight[cut]
            #baseweight = weight
            #modification for adding weight variable
            if var == 'baseweight':
                var_values = ak.to_numpy(weight)  # Store weight values directly
            else:
                var_values = _format_variable(event[var], cut)

            #modification ends here

            vv = ak.to_numpy(ak.fill_none(weight, np.nan))
            if np.isnan(np.any(vv)):
                print(f" - {syst} weight nan/inf:", vv[np.isnan(vv)], vv[np.isinf(vv)])
            #print("var " ,var)
            histos[var].fill(
                **{
                    "channel": ch, 
                    "systematic": systname, 
                    var: var_values, 
                    "weight": ak.nan_to_num(weight,nan=1.0, posinf=1.0, neginf=1.0) if var != 'baseweight' else None 
                }
            )
            
        for ch in channels:
            for sys in systematics:
                _histogram_filler(ch, sys, 'met_pt')
                _histogram_filler(ch, sys, 'm_T')
                _histogram_filler(ch, sys, 'm_T_WZ')
                _histogram_filler(ch, sys, 'emu_mT_WZ')
                _histogram_filler(ch, sys, 'dilep_mt')
                _histogram_filler(ch, sys, 'dilep_pt')
                _histogram_filler(ch, sys, 'dilep_m')
                _histogram_filler(ch, sys, 'tau_pt')
                _histogram_filler(ch, sys, 'taus_phi')
                _histogram_filler(ch, sys, 'taus_eta')
                _histogram_filler(ch, sys, 'delta_R')
                _histogram_filler(ch, sys, 'delta_R_jet_tau')
                _histogram_filler(ch, sys, 'delta_R_jet_dilep')
                _histogram_filler(ch, sys, 'dilep_dR')
                _histogram_filler(ch, sys, 'delta_tau_met_phi')
                _histogram_filler(ch, sys, 'tau_pt_loose')
                _histogram_filler(ch, sys, 'dphi_met_ll')
                _histogram_filler(ch, sys, 'dilep_dphi_tau')
                _histogram_filler(ch, sys, 'dphi_jet_met')
                _histogram_filler(ch, sys, 'dilep_dphi')
                _histogram_filler(ch, sys, 'deep_tau_jet')
                _histogram_filler(ch, sys, 'deep_tau_e')
                _histogram_filler(ch, sys, 'deep_tau_mu')
                
                
        return {dataset: histos}
        
    def process(self, event: processor.LazyDataFrame):
        dataset_name = event.metadata['dataset']
        is_data = event.metadata.get("is_data")
        
        # x-y met shit corrections
        # for the moment I am replacing the met with the corrected met 
        # before doing the JES/JER corrections
        
        run = event.run 
        npv = event.PV.npvs
        met = event.MET
        
        met = met_phi_xy_correction(
            event.MET, run, npv, 
            is_mc=not is_data, 
            era=self._era
        )
        event = ak.with_field(event, met, 'MET')

        # JES/JER corrections
        rho = event.fixedGridRhoFastjetAll
        cache = event.caches[0]
        if is_data: 
            softjet_gen_pt = None
        else:
            softjet_gen_pt = find_best_match(event.CorrT1METJet,event.GenJet)
        
        softjets_shift_L123 = self._jmeu.corrected_jets_L123(event.CorrT1METJet, rho, cache, softjet_gen_pt)
        softjets_shift_L1 = self._jmeu.corrected_jets_L1(event.CorrT1METJet, rho, cache, softjet_gen_pt)
        
        jets_shift_L123 = self._jmeu.corrected_jets_L123(event.Jet, rho, cache)
        jets_shift_L1 = self._jmeu.corrected_jets_L1(event.Jet, rho, cache)

        jets_col_shift_L123 = ak.concatenate([jets_shift_L123, softjets_shift_L123],axis=1)
        jets_col_shift_L1 = ak.concatenate([jets_shift_L1, softjets_shift_L1],axis=1)
        
        raw_met = event.RawMET
        met_to_correct = event.MET
        met_to_correct["pt"] = raw_met.pt
        met_to_correct["phi"] = raw_met.phi
        jets = self._jmeu.corrected_jets_jer(event.Jet, event.fixedGridRhoFastjetAll, event.caches[0])
        met = self._jmeu.corrected_met(met_to_correct, jets_col_shift_L123, jets_col_shift_L1, event.fixedGridRhoFastjetAll, event.caches[0])
        
        event = ak.with_field(event, jets, 'Jet')
        event = ak.with_field(event, met, 'MET')

    
        if is_data:
            
            # HEM15/16 issue
            if self._era == "2018":
                _runid = (event.run >= 319077)
                jets = event.Jet
                j_mask = ak.where((jets.phi > -1.57) & (jets.phi < -0.87) &
                                  (jets.eta > -2.50) & (jets.eta <  1.30) & 
                                  _runid, 0.8, 1)
                # met = event.MET
                # event['met_pt'] = met.pt
                # event['met_phi'] = met.phi            
                jets['pt']   = j_mask * jets.pt
                jets['mass'] = j_mask * jets.mass

                event = ak.with_field(event, jets, 'Jet')
                
            return self.process_shift(event, None)
		
        # Adding scale factors to Muon and Electron fields
        muon = event.Muon 
        electron = event.Electron
        muonSF_nom, muonSF_up, muonSF_down = self._leSF.muonSF(muon)
        elecSF_nom, elecSF_up, elecSF_down = self._leSF.electronSF(electron)
        
        muon['SF'] = muonSF_nom
        muon['SF_up'] = muonSF_up
        muon['SF_down'] = muonSF_down

        electron['SF'] = elecSF_nom
        electron['SF_up'] = elecSF_up
        electron['SF_down'] = elecSF_down

        event = ak.with_field(event, muon, 'Muon')
        event = ak.with_field(event, electron, 'Electron')

        # # # JES/JER corrections
        # jets = self._jmeu.corrected_jets(event.Jet, event.fixedGridRhoFastjetAll, event.caches[0])
        # met  = self._jmeu.corrected_met(event.MET, jets, event.fixedGridRhoFastjetAll, event.caches[0])
         
        # Apply rochester_correction
        muon=event.Muon
        muonEnUp=event.Muon
        muonEnDown=event.Muon
        muon_pt,muon_pt_roccorUp,muon_pt_roccorDown=rochester_correction(is_data).apply_rochester_correction (muon)
        
        muon['pt'] = muon_pt
        muonEnUp['pt'] = muon_pt_roccorUp
        muonEnDown['pt'] = muon_pt_roccorDown 
        event = ak.with_field(event, muon, 'Muon')
        
        # Electron corrections
        electronEnUp=event.Electron
        electronEnDown=event.Electron

        electronEnUp  ['pt'] = event.Electron['pt'] + event.Electron.energyErr/np.cosh(event.Electron.eta)
        electronEnDown['pt'] = event.Electron['pt'] - event.Electron.energyErr/np.cosh(event.Electron.eta)	


        #Tau corrections
        if not is_data :
            tau=event.Tau
            tauEnUp = event.Tau 
            tauEnDown = event.Tau 
            tau_pt,tau_pt_EnUp,tau_pt_EnDown,tau_mass,tau_mass_EnUp,tau_mass_EnDown=self._tauID.tau_energy_scale_correction(tau)

            tau['pt'] = tau_pt
            tau['mass'] = tau_mass
            tauEnUp['pt'] = tau_pt_EnUp
            tauEnUp['mass'] = tau_mass_EnUp
            tauEnDown['pt'] = tau_pt_EnDown
            tauEnDown['mass'] = tau_mass_EnDown
            event = ak.with_field(event, tau, 'Tau')


	
        # define all the shifts
        shifts = [
            # Jets
            ({"Jet": jets                             , "MET": met                               }, None                  ),
            ({"Jet": jets.JES_Total.up                , "MET": met.JES_Total.up                  }, "JESUp"               ),
            ({"Jet": jets.JES_Total.down              , "MET": met.JES_Total.down                }, "JESDown"             ),
            ({"Jet": jets.JES_Absolute.up             , "MET": met.JES_Absolute.up               }, "JES_AbsoluteUp"      ),
            ({"Jet": jets.JES_Absolute.down           , "MET": met.JES_Absolute.down             }, "JES_AbsoluteDown"    ),
            ({"Jet": jets.JES_BBEC1.up                , "MET": met.JES_BBEC1.up                  }, "JES_BBEC1Up"         ),
            ({"Jet": jets.JES_BBEC1.down              , "MET": met.JES_BBEC1.down                }, "JES_BBEC1Down"       ),
            ({"Jet": jets.JES_EC2.up                  , "MET": met.JES_EC2.up                    }, "JES_EC2Up"           ),
            ({"Jet": jets.JES_EC2.down                , "MET": met.JES_EC2.down                  }, "JES_EC2Down"         ),
            ({"Jet": jets.JES_FlavorQCD.up            , "MET": met.JES_FlavorQCD.up              }, "JES_FlavorQCDUp"     ),
            ({"Jet": jets.JES_FlavorQCD.down          , "MET": met.JES_FlavorQCD.down            }, "JES_FlavorQCDDown"   ),
            ({"Jet": jets.JES_HF.up                   , "MET": met.JES_HF.up                     }, "JES_HFUp"            ),
            ({"Jet": jets.JES_HF.down                 , "MET": met.JES_HF.down                   }, "JES_HFDown"          ),
            ({"Jet": jets.JES_RelativeBal.up          , "MET": met.JES_RelativeBal.up            }, "JES_RelativeBalUp"   ),
            ({"Jet": jets.JES_RelativeBal.down        , "MET": met.JES_RelativeBal.down          }, "JES_RelativeBalDown" ),
            ({"Jet": jets.JER.up                      , "MET": met.JER.up                        }, "JERUp"               ),
            ({"Jet": jets.JER.down                    , "MET": met.JER.down                      }, "JERDown"             ),
            ({"Jet": jets                             , "MET": met.MET_UnclusteredEnergy.up      }, "UESUp"               ),
            ({"Jet": jets                             , "MET": met.MET_UnclusteredEnergy.down    }, "UESDown"             ), 
            # year dependent systematics
            ({"Jet": getattr(jets,f'JES_BBEC1_{self._era}').up     , "MET": getattr(met,f'JES_BBEC1_{self._era}').up      }, f"JES_BBEC1{self._era}Up"  ),
            ({"Jet": getattr(jets,f'JES_BBEC1_{self._era}').down   , "MET": getattr(met,f'JES_BBEC1_{self._era}').down    }, f"JES_BBEC1{self._era}Down"),
            ({"Jet": getattr(jets,f'JES_Absolute_{self._era}').up  , "MET": getattr(met,f'JES_Absolute_{self._era}').up   }, f"JES_Absolute{self._era}Up"  ),
            ({"Jet": getattr(jets,f'JES_Absolute_{self._era}').down, "MET": getattr(met,f'JES_Absolute_{self._era}').down }, f"JES_Absolute{self._era}Down"),
            ({"Jet": getattr(jets,f'JES_EC2_{self._era}').up       , "MET": getattr(met,f'JES_EC2_{self._era}').up        }, f"JES_EC2{self._era}Up"  ),
            ({"Jet": getattr(jets,f'JES_EC2_{self._era}').down     , "MET": getattr(met,f'JES_EC2_{self._era}').down      }, f"JES_EC2{self._era}Down"),
            ({"Jet": getattr(jets,f'JES_HF_{self._era}').up        , "MET": getattr(met,f'JES_HF_{self._era}').up         }, f"JES_HF{self._era}Up"  ),
            ({"Jet": getattr(jets,f'JES_HF_{self._era}').down      , "MET": getattr(met,f'JES_HF_{self._era}').down       }, f"JES_HF{self._era}Down"),
            ({"Jet": getattr(jets,f'JES_RelativeSample_{self._era}').up  , "MET": getattr(met,f'JES_RelativeSample_{self._era}').up   }, f"JES_RelativeSample{self._era}Up"  ),
            ({"Jet": getattr(jets,f'JES_RelativeSample_{self._era}').down, "MET": getattr(met,f'JES_RelativeSample_{self._era}').down }, f"JES_RelativeSample{self._era}Down"),

            
            # Electrons + MET shift (FIXME: shift to be added)
            ({"Electron": electronEnUp  }, "ElectronEnUp"  ),
            ({"Electron": electronEnDown}, "ElectronEnDown"),
            # Muon + MET shifts
            ({"Muon": muonEnUp  }, "MuonRocUp"),
            ({"Muon": muonEnDown}, "MuonRocDown"),
            # Tau + MET shifts
            ({"Tau": tauEnUp  }, "TauEnUp"),
            ({"Tau": tauEnDown}, "TauEnDown"),

        ]
        
        shifts = [
            self.process_shift(
                update_collection(event, collections), 
                name
            ) for collections, name in shifts
        ]
        return processor.accumulate(shifts)
    

    
    def postprocess(self, accumulator):
        return accumulator


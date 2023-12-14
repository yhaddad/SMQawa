import awkward as ak
import numpy as np
import scipy.interpolate as interp
from scipy import stats as st
import uproot
import pickle
import hist
import yaml
import os
import re

from coffea import processor
from coffea.processor import dict_accumulator, column_accumulator
from coffea.nanoevents.methods import candidate

from coffea.analysis_tools import Weights, PackedSelection
from coffea.lumi_tools import LumiMask

from qawa.applyGNN import applyGNN

from qawa.roccor import rochester_correction
from qawa.leptonsSF import LeptonScaleFactors
from qawa.jetPU import jetPUScaleFactors
from qawa.tauSF import tauIDScaleFactors
from qawa.btag import BTVCorrector, btag_id
from qawa.jme import JMEUncertainty, update_collection
from qawa.common import pileup_weights, ewk_corrector, met_phi_xy_correction, theory_ps_weight, theory_pdf_weight, trigger_rules

import matplotlib.pyplot as plt


def build_leptons(muons, electrons):
    # select tight/loose muons
    tight_muons_mask = (
        (muons.pt             >  20. ) &
        (np.abs(muons.eta)    <  2.4 ) &
        (np.abs(muons.dxy)    <  0.02) &
        (np.abs(muons.dz )    <  0.1 ) &
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
        (np.abs(electrons.eta)  < 2.5) &
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
    base = (
        (tau.pt         > 20 ) & 
        (np.abs(tau.eta)< 2.3 ) & 
        (tau.decayMode != 5   ) & 
        (tau.decayMode != 6   ) &
        (tau.idDeepTau2017v2p1VSe >= 2) &
        (tau.idDeepTau2017v2p1VSmu >= 1) &
        (tau.idDeepTau2017v2p1VSjet >= 16)
    )
    overlap_leptons = ak.any(
        tau.metric_table(lepton) <= 0.4,
        axis=2
    )
    return tau[base & ~overlap_leptons]

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



class coffea_gnn_input(processor.ProcessorABC):
	# EWK corrections process has to be define before hand, it has to change when we move to dask
	def __init__(self, era: str ='2018', ewk_process_name=None):
		self._era = era
		if 'APV' in self._era:
			self._isAPV = True
			self._era = re.findall(r'\d+', self._era)[0]
		else:
			self._isAPV = False

		jec_tag = ''
		jer_tag = ''
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
		print (jer_tag)

		self.btag_wp = 'M'
		self.jetPU_wp = 'M'
		self.tauIDvsjet_wp = 'Medium'
		self.tauIDvse_wp = 'VVLoose'
		self.tauIDvsmu_wp = 'VLoose'
		self.zmass = 91.1873 # GeV 
		self._btag = BTVCorrector(era=self._era, wp=self.btag_wp, isAPV=self._isAPV)
		self._jmeu = JMEUncertainty(jec_tag, jer_tag, self._era)
		self._purw = pileup_weights(era=self._era)
		self._leSF = LeptonScaleFactors(era=self._era, isAPV=self._isAPV)
		self._jpSF = jetPUScaleFactors(era=self._era, wp=self.jetPU_wp, isAPV=self._isAPV)
		self._tauID= tauIDScaleFactors(era=self._era, vsjet_wp=self.tauIDvsjet_wp,vse_wp=self.tauIDvse_wp, vsmu_wp=self.tauIDvsmu_wp, isAPV=self._isAPV)

		_data_path = 'qawa/data'
		_data_path = os.path.join(os.path.dirname(__file__), '../data')
		self._data_path = _data_path
		self._json = {
			'2018': LumiMask(f'{_data_path}/json/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt'),
			'2017': LumiMask(f'{_data_path}/json/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt'),
			'2016': LumiMask(f'{_data_path}/json/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt'),
		}
		with open(f'{_data_path}/{self._era}-trigger-rules.yaml') as ftrig:
			self._triggers = yaml.load(ftrig, Loader=yaml.FullLoader)
        
		with uproot.open(f'{_data_path}/trigger_sf/histo_triggerEff_sel0_{self._era}.root') as _fn:
			_hvalue = np.dstack([_fn[_hn].values() for _hn in _fn.keys()] + [np.ones((7,7))])
			_herror = np.dstack([np.sqrt(_fn[_hn].variances()) for _hn in _fn.keys()] + [np.zeros((7,7))])
			self.trig_sf_map = np.stack([_hvalue, _herror], axis=-1)

		self.ewk_process_name = ewk_process_name
		if self.ewk_process_name is not None:
			self.ewk_corr = ewk_corrector(process=ewk_process_name)

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
	def get_process_category (self, dataset_name):
		with open(f'{self._data_path}/processCategory.yaml',"r") as stream:
			readyaml=yaml.safe_load(stream)
			return readyaml[dataset_name][0]

	def process(self, event: processor.LazyDataFrame):
		dataset_name = event.metadata['dataset']
		is_data = event.metadata.get("is_data")
		# selection = PackedSelection()
		weights = Weights(len(event), storeIndividual=True)
		event['proc']=[self.get_process_category(dataset_name)]*len(event)

		if is_data:
			print ("error: we don't need data for gnn input")

		run = event.run 
		npv = event.PV.npvs
		met = event.MET
        
		met = met_phi_xy_correction(
			event.MET, run, npv, 
			is_mc=not is_data, 
			era=self._era
		)
		event = ak.with_field(event, met, 'MET')
		
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

		# JES/JER corrections
		#jets = self._jmeu.corrected_jets(event.Jet, event.fixedGridRhoFastjetAll, event.caches[0])
		#met  = self._jmeu.corrected_met(event.MET, jets, event.fixedGridRhoFastjetAll, event.caches[0])
         
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
        
		# if is_data:
		# 	selection.add('lumimask', self._json[self._era](event.run, event.luminosityBlock))
		# 	selection.add('triggers', trigger_rules(event, self._triggers, self._era))
		# else:
		# 	selection.add('lumimask', np.ones(len(event), dtype='bool'))
		# 	selection.add('triggers', np.ones(len(event), dtype='bool'))

		# MET filter

		event['metfilter']= (event.Flag.METFilters &
							event.Flag.HBHENoiseFilter &
							event.Flag.HBHENoiseIsoFilter & 
							event.Flag.EcalDeadCellTriggerPrimitiveFilter & 
							event.Flag.goodVertices &
							event.Flag.eeBadScFilter &
							event.Flag.globalTightHalo2016Filter &
							event.Flag.BadChargedCandidateFilter & 
							event.Flag.BadPFMuonFilter)


		tight_lep, loose_lep = build_leptons(
			event.Muon,
			event.Electron
		)

		had_taus = build_htaus(event.Tau, tight_lep)
        
		ntight_lep = ak.num(tight_lep)
		nloose_lep = ak.num(loose_lep)
		nhtaus_lep = ak.num(had_taus)
        



		jets = event.Jet
		overlap_leptons = ak.any(
			jets.metric_table(tight_lep) <= 0.4,
			axis=2
		)
        
		jet_mask = (
			~overlap_leptons & 
			(jets.pt>30.0) & 
			(np.abs(jets.eta) < 4.7) & 
			(jets.jetId >= 6) & # tight JetID 7(2016) and 6(2017/8)
			(jets.puId >= 6) # medium puID https://twiki.cern.ch/twiki/bin/viewauth/CMS/PileupJetIDUL
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
        
		event['ngood_bjets'] = ak.fill_none(ngood_bjets,-99)
		event['ngood_jets']  = ak.fill_none(ngood_jets,-99)
		# lepton quantities
		def z_lepton_pair(leptons):
			pair = ak.combinations(leptons, 2, axis=1, fields=['l1', 'l2'])
			mass = (pair.l1 + pair.l2).mass
			cand = ak.local_index(mass, axis=1) == ak.argmin(np.abs(mass - self.zmass), axis=1)

			extra_lepton = leptons[(
				~ak.any(leptons.metric_table(pair[cand].l1) <= 0.01, axis=2) & 
				~ak.any(leptons.metric_table(pair[cand].l2) <= 0.01, axis=2) )
			]
			return pair[cand], extra_lepton, cand

		dilep, extra_lep, z_cand_mask = z_lepton_pair(tight_lep)
        
		lead_lep = ak.firsts(ak.where(dilep.l1.pt >  dilep.l2.pt, dilep.l1, dilep.l2),axis=1)
		subl_lep = ak.firsts(ak.where(dilep.l1.pt <= dilep.l2.pt, dilep.l1, dilep.l2),axis=1)

		event['require-3lep'] = (ntight_lep==3) & (nloose_lep==0) & (ak.firsts(tight_lep).pt>25) & ak.fill_none((lead_lep.pdgId + subl_lep.pdgId)==0, False)
        
		dilep_p4 = (lead_lep + subl_lep)
		dilep_m  = dilep_p4.mass
		dilep_pt = dilep_p4.pt
		dilep_eta= dilep_p4.eta
		dilep_phi= dilep_p4.phi
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
        
		# this definition is not correct as it doesn't include the mass of the second Z
		dilep_et_ll = np.sqrt(dilep_pt**2 + dilep_m**2)
		dilep_et_met = np.sqrt(reco_met_pt**2 + self.zmass**2)
		dilep_mt = ak.where(
				ntight_lep==3,
				np.sqrt((dilep_et_ll + dilep_et_met)**2 - (dilep_p4.pvec + emu_met.pvec).p2),
				np.sqrt((dilep_et_ll + dilep_et_met)**2 - (dilep_p4.pvec +  p4_met.pvec).p2)
		)
	
		# dilep_dphi = lead_lep.delta_phi(subl_lep)
		# dilep_deta = np.abs(lead_lep.eta - subl_lep.eta)
		# dilep_dR   = lead_lep.delta_r(subl_lep)
		dilep_dphi_met  = ak.where(ntight_lep==2, dilep_p4.delta_phi(p4_met), dilep_p4.delta_phi(emu_met))
		#scalar_balance = ak.where(ntight_lep==3, emu_met.pt/dilep_p4.pt, p4_met.pt/dilep_p4.pt)
        
        
		# 2jet and vbs related variables
		lead_jet = ak.firsts(good_jets)
		subl_jet = ak.firsts(good_jets[lead_jet.delta_r(good_jets)>0.01])
		third_jet = ak.firsts(good_jets[(lead_jet.delta_r(good_jets)>0.01) & (subl_jet.delta_r(good_jets)>0.01)])
        
		dijet_mass = (lead_jet + subl_jet).mass
		dijet_deta = np.abs(lead_jet.eta - subl_jet.eta)
		event['dijet_mass'] = ak.fill_none(dijet_mass,-99)
		event['dijet_deta'] = ak.fill_none(dijet_deta,-99)
        #dijet_zep1 = np.abs(2*lead_lep.eta - (lead_jet.eta + subl_jet.eta))/dijet_deta
        #dijet_zep2 = np.abs(2*subl_lep.eta - (lead_jet.eta + subl_jet.eta))/dijet_deta
        
		min_dphi_met_j = ak.min(np.abs(
			ak.where(
				ntight_lep==3, 
				jets.delta_phi(emu_met), 
				jets.delta_phi(p4_met)
			)
		), axis=1)

		event["require-ossf"] =ak.fill_none((ntight_lep==2) & (nloose_lep==0) &
								(ak.firsts(tight_lep).pt>25) &
								(lead_lep.pdgId + subl_lep.pdgId)==0, False)

		# Define all variables for the GNN
		event['nhtaus'] = ak.fill_none(nhtaus_lep,-99)
		event['dilep_eta'] = ak.fill_none(dilep_eta,-99)
		event['dilep_phi'] = ak.fill_none(dilep_phi,-99)
		event['dilep_dphi_met']=ak.fill_none(dilep_dphi_met,-99)

		event['met_pt'  ] = ak.fill_none(reco_met_pt,-99)
		event['met_phi' ] = ak.fill_none(reco_met_phi,-99)
		event['dilep_mt'] = ak.fill_none(dilep_mt,-99)
		event['dilep_m'] = ak.fill_none(dilep_m,-99)
		event['dilep_pt'] = ak.fill_none(dilep_pt,-99)
		event['njets'   ] = ak.fill_none(ngood_jets,-99)
		event['bjets'   ] = ak.fill_none(ngood_bjets,-99)
		event['dphi_met_ll'] = ak.fill_none(dilep_dphi_met,-99)
		event['dijet_mass'] = ak.fill_none(dijet_mass,-99)
		event['dijet_deta'] = ak.fill_none(dijet_deta,-99)
		event['min_dphi_met_j'] = ak.fill_none(min_dphi_met_j,-99)

		event['leading_lep_pt'  ] = ak.fill_none(lead_lep.pt,-99)
		event['leading_lep_eta' ] = ak.fill_none(lead_lep.eta,-99)
		event['leading_lep_phi' ] = ak.fill_none(lead_lep.phi,-99)
		event['trailing_lep_pt' ] = ak.fill_none(subl_lep.pt,-99)
		event['trailing_lep_eta'] = ak.fill_none(subl_lep.eta,-99)
		event['trailing_lep_phi'] = ak.fill_none(subl_lep.phi,-99)
                
		event['lead_jet_pt'  ] = ak.fill_none(lead_jet.pt,-99)
		event['lead_jet_eta' ] = ak.fill_none(lead_jet.eta,-99)
		event['lead_jet_phi' ] = ak.fill_none(lead_jet.phi,-99)
		event['trail_jet_pt' ] = ak.fill_none(subl_jet.pt,-99)
		event['trail_jet_eta'] = ak.fill_none(subl_jet.eta,-99)
		event['trail_jet_phi'] = ak.fill_none(subl_jet.phi,-99)
		event['third_jet_pt' ] = ak.fill_none(third_jet.pt,-99)
		event['third_jet_eta'] = ak.fill_none(third_jet.eta,-99)
		event['third_jet_phi'] = ak.fill_none(third_jet.phi,-99)

		gnn_score = applyGNN(event).get_nnscore()
		event['gnn_score'] = ak.fill_none(gnn_score,-99)
		# Now adding weights

		weights.add('genweight', event.genWeight)
		genweight = event.genWeight
		event['genweight'] = ak.fill_none(genweight,-99)
		# event_genweight = ak.where(genweight > 0, 1*np.ones_like(genweight), -1*np.ones_like(genweight))
		# weights.add('genweight', event_genweight)
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
			if 'aQGC' in dataset_name:
				weights.add('QCDScale0w'  , _ones, event.LHEScaleWeight[:, 1], event.LHEScaleWeight[:, 6])
				weights.add('QCDScale1w'  , _ones, event.LHEScaleWeight[:, 3], event.LHEScaleWeight[:, 4])
				weights.add('QCDScale2w'  , _ones, event.LHEScaleWeight[:, 0], event.LHEScaleWeight[:, 7])

			else:
				weights.add('QCDScale0w'  , _ones, event.LHEScaleWeight[:, 1], event.LHEScaleWeight[:, 7])
				weights.add('QCDScale1w'  , _ones, event.LHEScaleWeight[:, 3], event.LHEScaleWeight[:, 5])
				weights.add('QCDScale2w'  , _ones, event.LHEScaleWeight[:, 0], event.LHEScaleWeight[:, 8])

                
		if 'LHEReweightingWeight' in event.fields and 'aQGC' in dataset_name:
			for i in range(1057):
				weights.add(f"eft_{self._eftnames[i]}", event.LHEReweightingWeight[:, i])
		
		# 2017 Prefiring correction weight
		if 'L1PreFiringWeight' in event.fields:
  			weights.add(
					"prefiring_weight",
					event.L1PreFiringWeight.Nom,
					event.L1PreFiringWeight.Dn,
					event.L1PreFiringWeight.Up
					)
		final_weight = weights.weight()
		event["final_weight"] = final_weight

		vv = ak.to_numpy(ak.fill_none(final_weight, np.nan))
		if np.isnan(np.any(vv)):
			print("weight nan:", vv)


		dict_accumulator = processor.dict_accumulator({
						'met_pt': column_accumulator(np.zeros(shape=(0,))),
						'met_phi': column_accumulator(np.zeros(shape=(0,))),
						'lead_jet_pt': column_accumulator(np.zeros(shape=(0,))),
						'lead_jet_eta': column_accumulator(np.zeros(shape=(0,))),
						'lead_jet_phi': column_accumulator(np.zeros(shape=(0,))),
						'trail_jet_pt': column_accumulator(np.zeros(shape=(0,))),
						'trail_jet_eta': column_accumulator(np.zeros(shape=(0,))),
						'trail_jet_phi': column_accumulator(np.zeros(shape=(0,))),
						'third_jet_pt': column_accumulator(np.zeros(shape=(0,))),
						'third_jet_eta': column_accumulator(np.zeros(shape=(0,))),
						'third_jet_phi': column_accumulator(np.zeros(shape=(0,))),
						'leading_lep_pt': column_accumulator(np.zeros(shape=(0,))),
						'leading_lep_eta': column_accumulator(np.zeros(shape=(0,))),
						'leading_lep_phi': column_accumulator(np.zeros(shape=(0,))),
						'trailing_lep_pt': column_accumulator(np.zeros(shape=(0,))),
						'trailing_lep_eta': column_accumulator(np.zeros(shape=(0,))),
						'trailing_lep_phi': column_accumulator(np.zeros(shape=(0,))),
						'ngood_bjets': column_accumulator(np.zeros(shape=(0,))),
						'ngood_jets': column_accumulator(np.zeros(shape=(0,))),
						'dilep_pt': column_accumulator(np.zeros(shape=(0,))),
						'dilep_m': column_accumulator(np.zeros(shape=(0,))),
						'nhtaus': column_accumulator(np.zeros(shape=(0,))),
						'dijet_mass': column_accumulator(np.zeros(shape=(0,))),
						'dijet_deta': column_accumulator(np.zeros(shape=(0,))),
						'dilep_eta': column_accumulator(np.zeros(shape=(0,))),
						'dilep_phi': column_accumulator(np.zeros(shape=(0,))),
						'dilep_dphi_met': column_accumulator(np.zeros(shape=(0,))),
						'min_dphi_met_j': column_accumulator(np.zeros(shape=(0,))),
						'ossf' : column_accumulator(np.zeros(shape=(0,))),
						'final_weight' : column_accumulator(np.zeros(shape=(0,))),
						'metfilter': column_accumulator(np.zeros(shape=(0,))),
						'proc': column_accumulator(np.zeros(shape=(0,))),
						'gnn_score': column_accumulator(np.zeros(shape=(0,))),
						'require-3lep': column_accumulator(np.zeros(shape=(0,))),
						'genweight': column_accumulator(np.zeros(shape=(0,))),
						})

		gnn_input_acc = dict_accumulator.identity()
		gnn_input_acc['met_pt'] += column_accumulator(ak.to_numpy(event["met_pt"]))
		gnn_input_acc['met_phi'] += column_accumulator(ak.to_numpy(event["met_phi"]))
		gnn_input_acc['lead_jet_pt'] += column_accumulator(ak.to_numpy(event["lead_jet_pt"]))
		gnn_input_acc['lead_jet_eta'] += column_accumulator(ak.to_numpy(event["lead_jet_eta"]))
		gnn_input_acc['lead_jet_phi'] += column_accumulator(ak.to_numpy(event["lead_jet_phi"]))
		gnn_input_acc['trail_jet_pt'] += column_accumulator(ak.to_numpy(event["trail_jet_pt"]))
		gnn_input_acc['trail_jet_eta'] += column_accumulator(ak.to_numpy(event["trail_jet_eta"]))
		gnn_input_acc['trail_jet_phi'] += column_accumulator(ak.to_numpy(event["trail_jet_phi"]))
		gnn_input_acc['third_jet_pt'] += column_accumulator(ak.to_numpy(event["third_jet_pt"]))
		gnn_input_acc['third_jet_eta'] += column_accumulator(ak.to_numpy(event["third_jet_eta"]))
		gnn_input_acc['third_jet_phi'] += column_accumulator(ak.to_numpy(event["third_jet_phi"]))
		gnn_input_acc['leading_lep_pt'] += column_accumulator(ak.to_numpy(event["leading_lep_pt"]))
		gnn_input_acc['leading_lep_eta'] += column_accumulator(ak.to_numpy(event["leading_lep_eta"]))
		gnn_input_acc['leading_lep_phi'] += column_accumulator(ak.to_numpy(event["leading_lep_phi"]))
		gnn_input_acc['trailing_lep_pt'] += column_accumulator(ak.to_numpy(event["trailing_lep_pt"]))
		gnn_input_acc['trailing_lep_eta'] += column_accumulator(ak.to_numpy(event["trailing_lep_eta"]))
		gnn_input_acc['trailing_lep_phi'] += column_accumulator(ak.to_numpy(event["trailing_lep_phi"]))
		gnn_input_acc['ngood_bjets'] += column_accumulator(ak.to_numpy(event["ngood_bjets"]))
		gnn_input_acc['ngood_jets'] += column_accumulator(ak.to_numpy(event["ngood_jets"]))
		gnn_input_acc['dilep_pt'] += column_accumulator(ak.to_numpy(event["dilep_pt"]))
		gnn_input_acc['dilep_m'] += column_accumulator(ak.to_numpy(event["dilep_m"]))
		gnn_input_acc['nhtaus'] += column_accumulator(ak.to_numpy(event["nhtaus"]))
		gnn_input_acc['dijet_mass'] += column_accumulator(ak.to_numpy(event["dijet_mass"]))
		gnn_input_acc['dijet_deta'] += column_accumulator(ak.to_numpy(event["dijet_deta"]))
		gnn_input_acc['dilep_eta'] += column_accumulator(ak.to_numpy(event["dilep_eta"]))
		gnn_input_acc['dilep_phi'] += column_accumulator(ak.to_numpy(event["dilep_phi"]))
		gnn_input_acc['dilep_dphi_met'] += column_accumulator(ak.to_numpy(event["dilep_dphi_met"]))
		gnn_input_acc['min_dphi_met_j'] += column_accumulator(ak.to_numpy(event["min_dphi_met_j"]))
		gnn_input_acc['ossf'] += column_accumulator(ak.to_numpy(event["require-ossf"]))
		gnn_input_acc['final_weight'] += column_accumulator(ak.to_numpy(event["final_weight"]))
		gnn_input_acc['metfilter'] += column_accumulator(ak.to_numpy(event["metfilter"]))
		gnn_input_acc['gnn_score'] += column_accumulator(ak.to_numpy(event["gnn_score"]))
		gnn_input_acc['proc'] += column_accumulator(ak.to_numpy(event["proc"]))
		gnn_input_acc['require-3lep'] += column_accumulator(ak.to_numpy(event['require-3lep']))
		gnn_input_acc['genweight'] += column_accumulator(ak.to_numpy(event['genweight']))

#		print(gnn_input_acc)

		return {dataset_name: gnn_input_acc}

	def postprocess(self, accumulator):
		return accumulator



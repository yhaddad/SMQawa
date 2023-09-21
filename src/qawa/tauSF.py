import correctionlib
import pickle
import hist
import gzip
import os

from coffea.analysis_tools import Weights

import awkward as ak
import numpy as np
import uproot

from correctionlib import _core

class tauIDScaleFactors:
	def __init__(self, era:str='2018', vsjet_wp:str='Medium', vse_wp:str='VVLoose', vsmu_wp:str='VLoose', isAPV=False):
		self._era = era
		self.vsjet_wp = vsjet_wp
		self.vse_wp = vse_wp
		self.vsmu_wp = vsmu_wp
		self.isAPV = isAPV
		_data_path = os.path.join(os.path.dirname(__file__), 'data')
		# Load CorrectionSet
		fname = f"{_data_path}/tau_ID_SF/{era+'_APV' if isAPV else era}/tau.json.gz"
		with gzip.open(fname,'rt') as file:
			data = file.read().strip()
			cset = correctionlib.CorrectionSet.from_string(data)

		#Load Correction Objects :
		self.corr_vsjet = cset["DeepTau2017v2p1VSjet"]
		self.corr_vse = cset["DeepTau2017v2p1VSe"]
		self.corr_vsmu = cset["DeepTau2017v2p1VSmu"]

		
	def getSF(self, tau, syst="nom"):

		ntaus = ak.num(tau)
		taus  = ak.flatten(tau)
		tau_eta = np.array(taus.eta)
		tau_pt  = np.array(taus.pt)
		tau_dm  = np.array(taus.decayMode)
		tau_genmatch = np.array(taus.genPartFlav)
		
		# DeepTau2017v2p1VSjet
		sf_vsjet = self.corr_vsjet.evaluate(tau_pt,tau_dm,tau_genmatch, self.vsjet_wp, self.vse_wp, syst,"pt")
		sf_vsjet = ak.fill_none(sf_vsjet, 1.)
		sf_vsjet = ak.unflatten(sf_vsjet, ntaus)
		sf_vsjet = ak.prod(sf_vsjet, axis=-1)

		# DeepTau2017v2p1VSe
		sf_vse = self.corr_vse.evaluate(tau_eta,tau_genmatch,self.vse_wp,syst)
		sf_vse = ak.fill_none(sf_vse, 1.)
		sf_vse = ak.unflatten(sf_vse, ntaus)
		sf_vse = ak.prod(sf_vse, axis=-1)

		# DeepTau2017v2p1VSmu
		sf_vsmu = self.corr_vsmu.evaluate(tau_eta,tau_genmatch,self.vsmu_wp,syst)
		sf_vsmu = ak.fill_none(sf_vsmu, 1.)
		sf_vsmu = ak.unflatten(sf_vsmu, ntaus)
		sf_vsmu = ak.prod(sf_vsmu, axis=-1)


		return  sf_vsjet, sf_vse, sf_vsmu

	def append_tauID_sf(self, taus: ak.Array, weights: Weights):
	 	
		taus = taus[(taus.pt >= 20) & (np.abs(taus.eta) <= 2.3) & (np.abs(taus.dz) < 0.2)]
	 	
		sf_vsjet_nom,  sf_vse_nom,  sf_vsmu_nom  = self.getSF(taus, syst="nom" )
		sf_vsjet_up,   sf_vse_up,   sf_vsmu_up   = self.getSF(taus, syst="up"  )
		sf_vsjet_down, sf_vse_down, sf_vsmu_down = self.getSF(taus, syst="down")

		weights.add('tauIDvsjet_sf', sf_vsjet_nom, sf_vsjet_up, sf_vsjet_down)
		weights.add('tauIDvse_sf'  , sf_vse_nom,   sf_vse_up,   sf_vse_down  )
		weights.add('tauIDvsmu_sf' , sf_vsmu_nom,  sf_vsmu_up,  sf_vsmu_down )

		return sf_vsjet_nom, sf_vsjet_up, sf_vsjet_down, sf_vse_nom, sf_vse_up, sf_vse_down, sf_vsmu_nom, sf_vsmu_up, sf_vsmu_down




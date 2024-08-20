from coffea import processor
import pickle as pkl
import awkward as ak
import numpy as np
import os

from coffea.analysis_tools import Weights



class dataDrivenDYRatio:
	def __init__(self,dilep_pt,met_pt, isDY=False, era:str='2018', ddtype:str='onlySR'):
		self._isDY =isDY
		self._era = era
		self.met_pt = met_pt
		self.dilep_pt = dilep_pt
		_data_path = os.path.join(os.path.dirname(__file__), 'data')



		if era == '2018' or era=='2017':
			if ddtype == 'MC':
				print(f"DDMC_Ratio_ptmiss_binned_at_{'160' if era=='2017' else '200'}_{era}_DYSR.pkl")
				with open(f"{_data_path}/ddr/DDMC_Ratio_ptmiss_binned_at_{'160' if era=='2017' else '200'}_{era}_DYSR.pkl", 'rb') as pkl_file:
					ddmc_ratio = pkl.load(pkl_file)
			else:
				print(f"{_data_path}/ddr/DDMC_Ratio_ptmiss_binned_at_{'160' if era=='2017' else '200'}_{era}_{ddtype}.pkl")
				with open(f"{_data_path}/ddr/DDMC_Ratio_ptmiss_binned_at_{'160' if era=='2017' else '200'}_{era}_{ddtype}.pkl", 'rb') as pkl_file:
					ddmc_ratio = pkl.load(pkl_file)


		elif '2016' in era:
			if ddtype == 'MC':
				print(f"{_data_path}/ddr/DDMC_Ratio_ptmiss_binned_at_200_2016_DYSR.pkl")
				with open(f"{_data_path}/ddr/DDMC_Ratio_ptmiss_binned_at_200_2016_DYSR.pkl", 'rb') as pkl_file:
					ddmc_ratio = pkl.load(pkl_file)
			else:
				print(f"{_data_path}/ddr/DDMC_Ratio_ptmiss_binned_at_200_2016_{ddtype}.pkl")
				with open(f"{_data_path}/ddr/DDMC_Ratio_ptmiss_binned_at_200_2016_{ddtype}.pkl", 'rb') as pkl_file:
					ddmc_ratio = pkl.load(pkl_file)
		self.ddmc_ratio = ddmc_ratio

	def ddr_add_weight(self, weights:Weights):
		
		ddmc_ratio = self.ddmc_ratio
		met_pt=self.met_pt
		dilep_pt=self.dilep_pt

		if not self._isDY :
			dd_weights = np.ones(len(met_pt))
#			print("I'm not DY")
		else:
			dd_weights = np.zeros(len(met_pt))
			for i in range(len(ddmc_ratio['dilep_pt_bin'])-1):
				for j in range(len(ddmc_ratio['met_pt_bin'])-1):
					weights_i= np.array(list(map(lambda x: ddmc_ratio['ratio_mapping'][i][j] if x else 0, 
									(met_pt > ddmc_ratio['met_pt_bin'][j]) 
									& (met_pt < ddmc_ratio['met_pt_bin'][j+1])
									& (dilep_pt > ddmc_ratio['dilep_pt_bin'][i])
									& (dilep_pt < ddmc_ratio['dilep_pt_bin'][i+1]))))
					dd_weights = dd_weights+ weights_i		
		dd_weights[dd_weights==0]=1
		dd_weights_Up   = dd_weights.copy()
		dd_weights_Down = dd_weights.copy()
		weights.add('dataDrivenDYRatio', dd_weights,dd_weights_Up,dd_weights_Down)














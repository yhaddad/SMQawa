from coffea import processor
import pickle as pkl
import awkward as ak
import numpy as np
import os

from coffea.analysis_tools import Weights


def compare_and_replace_arrays(A, B):
	# Ensure both arrays are of the same shape
	if len(A) != len(B) or any(len(row) != len(B[0]) for row in A):
		raise ValueError("Arrays must have the same shape")

	# Create a new array to store the result
	result_array = [[0 for _ in range(len(A[0]))] for _ in range(len(A))]

	# Iterate over the elements of the arrays
	for i in range(len(A)):
		for j in range(len(A[0])):
			# Compare and replace if necessary
			if A[i][j] < 0:
				result_array[i][j] = B[i][j]
			else:
				result_array[i][j] = A[i][j]

	return result_array


class dataDrivenDYRatio:
	def __init__(self,dilep_pt,met_pt, isDY=False, era:str='2018'):
		self._isDY =isDY
		self._era = era
		self.met_pt = met_pt
		self.dilep_pt = dilep_pt
		_data_path = os.path.join(os.path.dirname(__file__), 'data')



		if era == '2018' or era=='2017':
			with open(f"{_data_path}/ddr/DDMC_Ratio_ptmiss_binned_at_{'160' if era=='2017' else '200'}_{era}.pkl", 'rb') as pkl_file:
				ddmc_ratio = pkl.load(pkl_file)

#			print("Loaded Dictionary:", ddmc_ratio)

		elif '2016' in era:
			# with open(f"{_data_path}/ddr/DDMC_Ratio_ptmiss_binned_at_200_2016.pkl", 'rb') as pkl_file:
			#  	ddmc_ratio_2016 = pkl.load(pkl_file)
			with open(f"{_data_path}/ddr/DDMC_Ratio_ptmiss_binned_at_200_2016APV.pkl", 'rb') as pkl_file:
				ddmc_ratio = pkl.load(pkl_file)
			# if era == '2016':
			# 	ddmc_ratio_2016['ratio_mapping'] = compare_and_replace_arrays(ddmc_ratio_2016['ratio_mapping'], ddmc_ratio_2016APV['ratio_mapping'])
			# 	ddmc_ratio = ddmc_ratio_2016
			# elif era=='2016APV':
			# 	ddmc_ratio_2016APV['ratio_mapping'] = compare_and_replace_arrays(ddmc_ratio_2016APV['ratio_mapping'], ddmc_ratio_2016['ratio_mapping'])
			# 	ddmc_ratio = ddmc_ratio_2016APV
		self.ddmc_ratio = ddmc_ratio

	def ddr_add_weight(self, weights):
		
		ddmc_ratio = self.ddmc_ratio
		met_pt=self.met_pt
		#print(met_pt,'met_pt')
		dilep_pt=self.dilep_pt
		#print(dilep_pt,'dilep_pt')

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
					#print(sum(weights_i))
					dd_weights = dd_weights+ weights_i
		#print(sum(weights_i),'sumweight')			
		dd_weights[dd_weights==0]=1

		# for i in dd_weights:
		# 	print(i)
		weights.add('dataDrivenDYRatio', dd_weights)














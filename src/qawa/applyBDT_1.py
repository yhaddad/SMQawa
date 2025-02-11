from coffea import processor
import onnxruntime as rt
import numpy.lib.recfunctions as rf
import awkward as ak
import uproot
import pickle

import numpy as np
import pandas as pd
import math
import os
from xgboost.sklearn import XGBClassifier
import xgboost as xgb
from sklearn.model_selection  import train_test_split
from sklearn.model_selection  import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split

class applyGNN:
	 def __init__(self, event: processor.LazyDataFrame):
		_data_path = os.path.join(os.path.dirname(__file__), 'data/')
		df1 = pd.read_parquet(f'{_data_path}/BDTmodel/df_2018_WZ_inclusive_HT.parquet', engine='pyarrow')
		df2 = pd.read_parquet(f'{_data_path}/BDTmodel/df_2016_WZ_inclusive_HT.parquet', engine='pyarrow')
		df3 = pd.read_parquet(f'{_data_path}/BDTmodel/df_2017_WZ_inclusive_HT.parquet', engine='pyarrow')
		df= result_row_concat = pd.concat([df1, df2,df3], axis=0)

		sig=df[df['proc'] =='WZ_bkg']
		bkg = df[df['proc'] !='WZ_bkg']

		a = ['met_pt', 'met_phi', 'leading_lep_pt', 'leading_lep_eta',
		       'leading_lep_phi', 'trailing_lep_pt', 'trailing_lep_eta',
		       'trailing_lep_phi', 'ngood_bjets', 'dilep_pt', 'dilep_m',
		       'nhtaus', 'nhtaus_phi', 'tau_pt', 'delta_tau_met_phi',
		       'm_T', 'dilep_eta', 'dilep_phi',
		       'dilep_dphi_met', 'min_dphi_met_j', 'ossf', 'final_weight', 'metfilter']

		sig = sig[sig['final_weight']>0]
		sig_w = sig[sig['final_weight']>0].final_weight

		bkg = bkg[bkg['final_weight']>1e-100]
		bkg_w = bkg[bkg['final_weight']>1e-100].final_weight


		a = ['met_pt', 'met_phi', 'leading_lep_pt', 'leading_lep_eta',
		       'leading_lep_phi', 'trailing_lep_pt', 'trailing_lep_eta',
		       'trailing_lep_phi', 'dilep_pt', 'dilep_m', 'nhtaus_phi', 'tau_pt', 'delta_tau_met_phi',
		       'm_T', 'dilep_eta', 'dilep_phi',
		       'dilep_dphi_met']

		sig = sig[a]
		bkg = bkg[a]

		test_size = 0.2                                   #test_size : all_data, defult 0.2
		#The proportion of the dataset to include in the test split. Here, 20% of the data will be used for testing, and the remaining 80% will be used for training.
		X = pd.concat([sig.dropna(), bkg.dropna()]) #dropna() : drop Nan data; mix data and background; X feature set
		y = np.concatenate((np.ones(sig.shape[0]),np.zeros(bkg.shape[0])))  #.shape[0] number of rows; .shape[1] number of column; y sample tag
		weight = pd.concat((sig_w, bkg_w))

		X_train, X_test, y_train, y_test, weight_train, weight_test = train_test_split(X, y, weight, test_size=0.2, random_state=42)

		def normalize_weights(weights, classes):
		    weights_ = np.copy(weights)
		    for cl in np.unique(classes):
		        weights_[classes==cl] = 1+weights_[classes==cl]/np.sum(weights_[classes==cl])
		    return weights_


		w_train = normalize_weights(weight_train, y_train)
		w_test = normalize_weights(weight_test, y_test)

		# w_train = weight_train
		# w_test = weight_test

		dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train, enable_categorical=True)
		dtest = xgb.DMatrix(X_test, label=y_test, weight=w_test, enable_categorical=True)
		param = {
		    'eval_metric': 'auc',             # evaluation metric
		    #'eta': 0.1,                        # learning rate
		    'learning_rate': 0.1,
		    'max_depth': 5,                    # maximum depth of trees
		    'subsample': 0.8,                  # subsample ratio of the training instances
		    'colsample_bytree': 0.8,         # subsample ratio of columns when constructing each tree
		    'objective':'binary:logistic',
		    'seed': 80                         # random seed
		}

		num_round = 400

		watchlist = [(dtest,'eval'), (dtrain,'train')]
		evals_result = {}


		bst = xgb.train(param, dtrain, num_round, watchlist, evals_result=evals_result,early_stopping_rounds=15)



		pred_train = bst.predict(dtrain)
		pred_test = bst.predict(dtest)

		fpr_train, tpr_train, thr_train=roc_curve(y_train, pred_train)
		auc_train=roc_auc_score(y_train, pred_train)

		fpr_test, tpr_test, thr_test=roc_curve(y_test, pred_test)
		auc_test=roc_auc_score(y_test, pred_test)


		bst.save_model(f'{_data_path}/model_bdt.json')

		#y_pred1 = bst.predict(dtest)
		#print(y_pred1)
		# Create separate arrays for signal and background predictions
		#signal_pred = y_pred1[y_test==1]
		#background_pred = y_pred1[y_test==0]

		#bdt_cut = []
		#for i in signal_pred :
		#    bdt_cut.append(i)
		#for i in background_pred :
		#    bdt_cut.append(i)

		#print(len(bdt_cut))

		#BDT = np.array(bdt_cut)

		#applying cut on BDT

		#BDT = BDT[BDT>0.3]

	def get_bdtscore(self):
		Variable_bdt = ['met_pt', 'lead_jet_pt', 'lead_jet_eta', 'lead_jet_phi', 'trail_jet_pt', 'trail_jet_eta', 'trail_jet_phi', 'leading_lep_pt', 'leading_lep_eta', 'trailing_lep_pt', 'trailing_lep_eta', 'dijet_mass', 'dijet_deta', 'dilep_phi', 'dilep_dphi_met', 'min_dphi_met_j','ngood_jets']
		bdt_score = bst.predict(Variable_bdt)

		return bdt_score
		




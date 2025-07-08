from coffea import processor
from qawa.globalSignificance import globalSignificance
import tensorflow as tf
import awkward as ak
import numpy as np
import pickle as pkl
import warnings
import os

class applyGNN:
    def __init__(self, event: processor.LazyDataFrame):
        
        _data_path = os.path.join(os.path.dirname(__file__), 'data/')
        model = f'{_data_path}/GNNmodel/BestUnifiedModel.keras'
        with open(f'{_data_path}/GNNmodel/BestModelScaler.pkl', 'rb') as pkl_file:
            self.modelScaler = pkl.load(pkl_file)
        
        self.event = event

        self.features = ['lead_jet_pt','lead_jet_eta','lead_jet_phi',
            'trail_jet_pt','trail_jet_eta','trail_jet_phi',
            'third_jet_pt','third_jet_eta','third_jet_phi',
            'leading_lep_pt','leading_lep_eta','leading_lep_phi',
            'trailing_lep_pt','trailing_lep_eta','trailing_lep_phi',
            'met_pt','met_phi','dijet_mass','dijet_deta','dilep_m','ngood_jets']

        
        nn_model = tf.keras.models.load_model(model, custom_objects={'globalSignificance': globalSignificance})

        self.model = nn_model
        
    def conversion_GNN(self):
        
        inputs = self.event[self.features]

        df_inputs=ak.to_pandas(inputs)
        df_inputs.loc[df_inputs['third_jet_pt'] == -99, 'third_jet_pt'] = 30
        df_inputs.loc[df_inputs['third_jet_phi'] == -99, 'third_jet_phi'] = 0
        df_inputs.loc[df_inputs['third_jet_eta'] == -99, 'third_jet_eta'] = 0
        df_inputs.loc[df_inputs['ngood_jets'] > 2, 'ngood_jets'] = 3

        warnings.filterwarnings("ignore", message="X has feature names, but MinMaxScaler was fitted without feature names")

        df_inputs = self.modelScaler.transform(df_inputs)
        
        return df_inputs
    
    
    def get_nnscore(self):

        return self.model.predict(self.conversion_GNN(), verbose=0)[:,0]

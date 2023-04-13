from coffea import processor
import onnxruntime as rt
import numpy.lib.recfunctions as rf
import awkward as ak
import numpy as np
import uproot
import os

class applyGNN:
    def __init__(self, event: processor.LazyDataFrame):
        
        _data_path = os.path.join(os.path.dirname(__file__), 'data/')
        model_2j = f'{_data_path}/GNNmodel/exactly2JetsNoBjets.onnx'
        model_3j = f'{_data_path}/GNNmodel/atLeast3JetsNoBjets.onnx'
        
        self.event = event
        self.feature_2j = [
            'lead_jet_pt','lead_jet_eta','lead_jet_phi',
            'trail_jet_pt','trail_jet_eta','trail_jet_phi',
            'leading_lep_pt','leading_lep_eta','leading_lep_phi',
            'trailing_lep_pt','trailing_lep_eta','trailing_lep_phi',
            'met_pt','met_phi','ngood_bjets',
        ]
        self.feature_3j = [
            'lead_jet_pt','lead_jet_eta','lead_jet_phi',
            'trail_jet_pt','trail_jet_eta','trail_jet_phi',
            'third_jet_pt','third_jet_eta','third_jet_phi',
            'leading_lep_pt','leading_lep_eta','leading_lep_phi',
            'trailing_lep_pt','trailing_lep_eta','trailing_lep_phi',
            'met_pt','met_phi','ngood_bjets','ngood_jets',
        ]
        self.model_2j=model_2j
        self.model_3j=model_3j
        
    def conversion_GNN(self):
        
        nn2j_inputs = self.event[self.feature_2j]
        nn3j_inputs = self.event[self.feature_3j]

        df_allNoeta=ak.to_pandas(nn2j_inputs)
        df_allEta=ak.to_pandas(nn3j_inputs)
        featuresNumber2Jets = 4 # the number of rows in the object matrix for the 2 jets events
        featuresNumber3Jets = 5 # the number of rows in the objects matrix for the 3 jets events
        objectsNumber3Jets  = 6 # the number of columns in the objects matrix for the 3 jets network
        jetsNumber3Jets     = 3 # the number of jets under consideration for the at least 3 jets sample
        objectsNumber2Jets  = 5 # the number of columns in the objects matrix for the 2 jets network
        jetsNumber2Jets     = 2 # the number of jets under consideration for the exactly 2 jets sample
        atLeast3Jets = np.zeros(shape=(len(  df_allEta),objectsNumber3Jets,featuresNumber3Jets),dtype=np.float32) # the objects matrix for the at least 3 jets events
        exactly2Jets = np.zeros(shape=(len(df_allNoeta),objectsNumber2Jets,featuresNumber2Jets),dtype=np.float32) # the objects matrix for the exactly 2 jets events

        #loop to initialize the pt, eta, phi and quark-gluon discriminant of jets, leptons and the Met for exactly 2 jets events
        for j in range(objectsNumber2Jets):
            if j < jetsNumber2Jets:
                exactly2Jets[:,j,0] = df_allNoeta.iloc[:,3*j]
                exactly2Jets[:,j,0] = df_allNoeta.iloc[:,3*j]
                exactly2Jets[:,j,1] = df_allNoeta.iloc[:,3*j+1]
                exactly2Jets[:,j,2] = df_allNoeta.iloc[:,3*j+2]
                # the line below is an initialization of a global feature
                exactly2Jets[:,j,3] = df_allNoeta.iloc[:,len(df_allNoeta.columns)-1] # initialization of ngood_bjets variable
            # in this elif conditional I initialize the quark-gluon discriminant of the leptons as -1.0 to force an indeterminacy (leptons are fundamental particles which are neither provenient of quark nor gluons)
            elif (jetsNumber2Jets <= j) & (j <(objectsNumber2Jets-1)):
                exactly2Jets[:,j,0] = df_allNoeta.iloc[:,j*3]
                exactly2Jets[:,j,1] = df_allNoeta.iloc[:,j*3+1]
                exactly2Jets[:,j,2] = df_allNoeta.iloc[:,j*3+2]
                # the line below is an initialization of a global feature
                exactly2Jets[:,j,3] = df_allNoeta.iloc[:,len(df_allNoeta.columns)-1] # initialization of ngood_bjets variable
            # in this elif conditional I initialize the quark-gluon discriminant of the neutrino as -1.0 to force an indeterminacy (neutrinos are neither provenient of quark nor gluons)
            else:
                exactly2Jets[:,j,0] = df_allNoeta.iloc[:,j*3]
                exactly2Jets[:,j,1] = 0.0
                exactly2Jets[:,j,2] = df_allNoeta.iloc[:,j*3+1]
                # The line below is an initialization of a global feature
                exactly2Jets[:,j,3] = df_allNoeta.iloc[:,len(df_allNoeta.columns)-1] # initialization of ngood_bjets variable
            #loop to initialize the pt, eta, phi and quark-gluon discriminant of jets, leptons and the Met for at least 3 jets events
        for j in range(objectsNumber3Jets):
            if j < jetsNumber3Jets:
                atLeast3Jets[:,j,0] = df_allEta.iloc[:,3*j]
                atLeast3Jets[:,j,1] = df_allEta.iloc[:,3*j+1]
                atLeast3Jets[:,j,2] = df_allEta.iloc[:,3*j+2]
                # the line beow are global features initialization lines
                atLeast3Jets[:,j,3] = df_allEta.iloc[:,len(df_allEta.columns)-1]
                atLeast3Jets[:,j,4] = df_allEta.iloc[:,len(df_allEta.columns)-2]
                # in this elif conditional I initialize the quark-gluon discriminant of the leptons as -1.0 to force an indeterminacy (leptons are fundamental particles which are neither provenient of quark nor gluons)
            elif (jetsNumber3Jets <= j) & (j <(objectsNumber3Jets-1)):
                atLeast3Jets[:,j,0] = df_allEta.iloc[:,j*3]
                atLeast3Jets[:,j,1] = df_allEta.iloc[:,j*3+1]
                atLeast3Jets[:,j,2] = df_allEta.iloc[:,j*3+2]
                # The two lines below are global features initialization lines
                atLeast3Jets[:,j,3] = df_allEta.iloc[:,len(df_allEta.columns)-1]
                atLeast3Jets[:,j,4] = df_allEta.iloc[:,len(df_allEta.columns)-2]
                # in this elif conditional I initialize the quark-gluon discriminant of the neutrino as -1.0 to force an indeterminacy (neutrinos are neither provenient of quark nor gluons)
            else:
                atLeast3Jets[:,j,0] = df_allEta.iloc[:,j*3]
                atLeast3Jets[:,j,1] = 0.0
                atLeast3Jets[:,j,2] = df_allEta.iloc[:,j*3+1]
                atLeast3Jets[:,j,3] = df_allEta.iloc[:,len(df_allEta.columns)-1]
                atLeast3Jets[:,j,4] = df_allEta.iloc[:,len(df_allEta.columns)-2]

        return exactly2Jets, atLeast3Jets
    
    
    def get_nnscore(self):
        if len(self.event) == 0:
            return -1
        nn2j_inputs,nn3j_inputs=self.conversion_GNN()
        
        sess_2j = rt.InferenceSession(
             self.model_2j,
             providers=rt.get_available_providers()
        )
        
        nn2j_score = sess_2j.run([sess_2j.get_outputs()[0].name], {
            sess_2j.get_inputs()[0].name: nn2j_inputs.astype(np.float32), 
            sess_2j.get_inputs()[1].name: -np.ones(len(nn2j_inputs)).astype(np.float32),
            sess_2j.get_inputs()[2].name: -np.ones(len(nn2j_inputs)).astype(np.float32),
        })[0].flatten()
        
        
        sess_3j = rt.InferenceSession(
            self.model_3j, 
            providers=rt.get_available_providers()
        )
        
        nn3j_score = sess_3j.run([sess_3j.get_outputs()[0].name], {
            sess_3j.get_inputs()[0].name: nn3j_inputs.astype(np.float32), 
            sess_3j.get_inputs()[1].name: -np.ones(len(nn3j_inputs)).astype(np.float32),
            sess_3j.get_inputs()[2].name: -np.ones(len(nn3j_inputs)).astype(np.float32),
        })[0].flatten()
        
        del sess_2j
        del sess_3j

        return np.where(
            self.event.ngood_jets == 2,
            nn2j_score,
            nn3j_score
        )

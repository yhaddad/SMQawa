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

class applyBDT:
    def __init__(self, event: processor.LazyDataFrame):
        _data_path = os.path.join(os.path.dirname(__file__), 'data/')
        model_bdt = f'{_data_path}/BDTmodel/bdtmodel_vbs_uW.onnx'
        self.event = event
        self.feature = ['met_pt', 'dilep_pt', 'dijet_mass', 'dijet_deta', 'lead_jet_pt', 'trail_jet_pt']

        self.model_bdt = model_bdt

    def conversion_BDT(self):
        BDT_inputs = self.event[self.feature]
        bdt_inputs = ak.to_pandas(BDT_inputs)
        bdt_inputs = bdt_inputs.astype(np.float32).values

        return bdt_inputs

    
    def get_bdtscore(self):
        BDT_inputs = self.conversion_BDT()
        sess = rt.InferenceSession(self.model_bdt,providers=['AzureExecutionProvider', 'CPUExecutionProvider'])
        input_name = sess.get_inputs()[0].name

        label_name_1 = sess.get_outputs()[1].name
        pred_onx_1 = sess.run([label_name_1], {input_name: BDT_inputs})[0].flatten()
        # print(pred_onx_1)
        # array = np.array(pred_onx_1)
        # filename = 'array_DY.txt'
        # np.savetxt(filename, array, delimiter=',', fmt='%.6f')
        BDT_score = pred_onx_1[1::2]

        return BDT_score

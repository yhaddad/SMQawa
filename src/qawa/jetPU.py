import correctionlib
import pickle
import hist
import gzip
import os
from hist.intervals import clopper_pearson_interval
from correctionlib import _core

from coffea.analysis_tools import Weights

import awkward as ak
import numpy as np
import uproot



class jetPUScaleFactors:
    def __init__(self, era:str='2018', wp:str='M', isAPV=False):
        self._era = era
        self._wp  = wp
        self.isAPV = isAPV
        _data_path = os.path.join(os.path.dirname(__file__), 'data')
        with gzip.open(
                f"{_data_path}/jet_PU_SF/jmar_{era+'_APV' if isAPV else era}_UL.json.gz", 
                'rt') as ifile:
            data = ifile.read().strip()
            #self.evaluator = _core.CorrectionSet.from_string(data)
            self.evaluator = correctionlib.CorrectionSet.from_string(data)


    def getSF(self, jet, syst="nom"):
        njets = ak.num(jet)
        jets  = ak.flatten(jet)
        jet_eta = np.array(jets.eta)
        jet_pt  = np.array(jets.pt)
        jet_sf = self.evaluator["PUJetID_eff"].evaluate(
            jet_eta,
            jet_pt,
            syst, self._wp
            )
        jet_sf = ak.fill_none(jet_sf, 1.)
        sf = ak.unflatten(jet_sf, njets)
        sf = ak.prod(sf, axis=-1)
        return sf
    


    def append_jetPU_sf(self, jets: ak.Array, weights: Weights):
        jets = jets[(jets.pt <= 50) & (np.abs(jets.eta) <= 5) & (jets.pt >= 20)]
        
        sf_nom  = self.getSF(jets, 'nom')
        sf_up   = self.getSF(jets, 'up')
        sf_down = self.getSF(jets, 'down')
        weights.add('jetPUid_sf'  , sf_nom, sf_up, sf_down)



        return sf_nom, sf_up, sf_down

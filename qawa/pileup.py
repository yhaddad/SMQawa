import os
import uproot
import awkward as ak

from coffea.lookup_tools import dense_lookup
from coffea.analysis_tools import Weights

class pileup_corrector:
    def __init__(self, do_syst:bool=True, era:str='2018'):
        _data_path = os.path.join(os.path.dirname(__file__), 'data')
        _files = {
            "Nom" : f"{_data_path}/PU/PileupHistogram-goldenJSON-13tev-{era}-69200ub-99bins.root",
            "Up"  : f"{_data_path}/PU/PileupHistogram-goldenJSON-13tev-{era}-72400ub-99bins.root",
            "Down": f"{_data_path}/PU/PileupHistogram-goldenJSON-13tev-{era}-66000ub-99bins.root"
        }
        
        hist_norm = lambda x: np.divide(x, x.sum())
        
        simu_pu = None 
        with uproot.open(f'{_data_path}/PU/mcPileupUL{era}.root') as ifile:
            simu_pu = ifile['pu_mc'].values()
            simu_pu = np.insert(simu_pu,0,0.)
            simu_pu = simu_pu[:-2]
        
        mask = simu_pu > 0
        
        self.corrections = {}
        for var,pfile in _files.items():
            with uproot.open(pfile) as ifile:
                data_pu = hist_norm(ifile["pileup"].values())
                edges = ifile["pileup"].axis().edges()
                
                corr = np.divide(data_pu, simu_pu, where=mask)
                pileup_corr = dense_lookup.dense_lookup(corr, edges)
                self.corrections['puWeight' if 'Nom' in var else f'puWeight{var}'] = pileup_corr
        
        

    def add_pileup_weight(self, weights:Weights, nPU: ak.Array):
        weights.add(
            'pileup_weight',
            self.corrections[f'puWeight'    ](nPU),
            self.corrections[f'puWeightUp'  ](nPU),
            self.corrections[f'puWeightDown'](nPU),
        )
        return weights

#compiled = pileup_correction(events)
#add_pileup_weight(weights, events.Pileup.nPU)

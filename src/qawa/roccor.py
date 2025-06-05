from coffea import lookup_tools
from coffea.lookup_tools import extractor, txt_converters, rochester_lookup

import awkward as ak
import numpy as np
import uproot
import os

class rochester_correction:
    def __init__(self, is_data=False, era:str='2018'):
        self._era = era
        self._is_mc = not is_data
        
        _data_path = os.path.join(os.path.dirname(__file__), 'data/')
        if self._era == '2016APV':
            rochester_data = txt_converters.convert_rochester_file(f"{_data_path}/roccor/RoccoR2016aUL.txt", loaduncs=True)
        elif self._era == '2016':
            rochester_data = txt_converters.convert_rochester_file(f"{_data_path}/roccor/RoccoR2016bUL.txt", loaduncs=True)
        elif self._era == '2017':
            rochester_data = txt_converters.convert_rochester_file(f"{_data_path}/roccor/RoccoR2017UL.txt", loaduncs=True)
        elif self._era == '2018':
            rochester_data = txt_converters.convert_rochester_file(f"{_data_path}/roccor/RoccoR2018UL.txt", loaduncs=True)
        else:
            print('lost in time? either too far in the past or in the future ...')
        
        self.rochester = rochester_lookup.rochester_lookup(rochester_data)
        
    def apply_rochester_correction (self, muons):
        if self._is_mc:
            hasgen = ~np.isnan(ak.fill_none(muons.matched_gen.pt, np.nan))
            mc_rand = np.random.rand(*ak.to_numpy(ak.flatten(muons.pt)).shape)
            mc_rand = ak.unflatten(mc_rand, ak.num(muons.pt, axis=1))
            corrections = np.array(ak.flatten(ak.ones_like(muons.pt)))
            errors = np.array(ak.flatten(ak.ones_like(muons.pt)))
            
            mc_kspread = self.rochester.kSpreadMC(muons.charge[hasgen],muons.pt[hasgen],muons.eta[hasgen],muons.phi[hasgen],muons.matched_gen.pt[hasgen])
            mc_ksmear = self.rochester.kSmearMC(muons.charge[~hasgen],muons.pt[~hasgen],muons.eta[~hasgen],muons.phi[~hasgen],muons.nTrackerLayers[~hasgen],mc_rand[~hasgen])
            errspread = self.rochester.kSpreadMCerror(muons.charge[hasgen],muons.pt[hasgen],muons.eta[hasgen],muons.phi[hasgen],muons.matched_gen.pt[hasgen])
            errsmear = self.rochester.kSmearMCerror(muons.charge[~hasgen],muons.pt[~hasgen],muons.eta[~hasgen],muons.phi[~hasgen],muons.nTrackerLayers[~hasgen],mc_rand[~hasgen])
            hasgen_flat = np.array(ak.flatten(hasgen))
            corrections[hasgen_flat] = np.array(ak.flatten(mc_kspread))
            corrections[~hasgen_flat] = np.array(ak.flatten(mc_ksmear))
            errors[hasgen_flat] = np.array(ak.flatten(errspread))
            errors[~hasgen_flat] = np.array(ak.flatten(errsmear))
            corrections = ak.unflatten(corrections, ak.num(muons.pt, axis=1))
            errors = ak.unflatten(errors, ak.num(muons.pt, axis=1))
        else:
            corrections = self.rochester.kScaleDT(muons.charge, muons.pt, muons.eta, muons.phi)
            errors = self.rochester.kScaleDTerror(muons.charge, muons.pt, muons.eta, muons.phi)
    
        pt_nom = np.where(muons.pt<=200,muons.pt*corrections,muons.pt)
        pt_err = np.where(muons.pt<=200,muons.pt*errors,np.zeros_like(muons.pt))
        # pt_nom = muons.pt * corrections
        # pt_err = muons.pt * errors
    
        return pt_nom, pt_nom + pt_err, pt_nom - pt_err

from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.lookup_tools import extractor
from coffea.jetmet_tools import FactorizedJetCorrector
from coffea.jetmet_tools import JetResolution
from coffea.jetmet_tools import JECStack
from coffea.jetmet_tools import JetCorrectionUncertainty
from coffea.jetmet_tools import JetResolutionScaleFactor
from coffea.jetmet_tools import CorrectedJetsFactory
from coffea.jetmet_tools import CorrectedMETFactory

from coffea.lookup_tools import dense_lookup
import awkward as ak
import numpy as np
import os


jec_name_map = {
    'JetPt': 'pt',
    'JetMass': 'mass',
    'JetEta': 'eta',
    'JetA': 'area',
    'ptRaw': 'pt_raw',
    'massRaw': 'mass_raw',
    'Rho': 'rho',
    'METpt': 'pt',
    'METphi': 'phi',
    'JetPhi': 'phi',
    'UnClusteredEnergyDeltaX': 'MetUnclustEnUpDeltaX',
    'UnClusteredEnergyDeltaY': 'MetUnclustEnUpDeltaY',
}

def update_collection(event, coll):
    out = event
    for name, value in coll.items():
        out = ak.with_field(out, value, name)
    return out

def add_jme_variables(jets, events_rho, pt_gen=None):
    jets['pt_raw'  ] = (1 - jets.rawFactor) * jets.pt 
    jets['mass_raw'] = (1 - jets.rawFactor) * jets.mass 
    if hasattr(jets, 'matched_gen'):
        jets['pt_gen'  ] = ak.values_astype(ak.fill_none(jets.matched_gen.pt, 0), np.float32)
    elif pt_gen is not None:
        jets['pt_gen'] = ak.values_astype(ak.fill_none(pt_gen, 0), np.float32)
    else:
        jets['pt_gen'] = ak.Array(np.zeros(len(jets), dtype=np.float32))
    jets['rho'     ] = ak.broadcast_arrays(events_rho, jets.pt)[0]
    return jets

class JMEUncertainty:
    def __init__(
        self,
        jec_tag: str = 'Summer19UL18_V5_MC',
        jer_tag: str = 'Summer19UL18_JRV2_MC',
        era: str = "2018",
        is_mc: bool = True
    ):
        _data_path = os.path.join(os.path.dirname(__file__), 'data/jme/')
        extract_L123 = extractor()
        extract_L1 = extractor()
        extract_JER = extractor()
        
        correction_list_L123 = [
            # Jet Energy Correction
            f'* * {_data_path}/{era}/{jec_tag}_L1FastJet_AK4PFchs.jec.txt',
            f'* * {_data_path}/{era}/{jec_tag}_L2L3Residual_AK4PFchs.jec.txt',
            f'* * {_data_path}/{era}/{jec_tag}_L2Relative_AK4PFchs.jec.txt',
            f'* * {_data_path}/{era}/{jec_tag}_L3Absolute_AK4PFchs.jec.txt',
        ]
        
        correction_list_L1 = [
            # Jet Energy Correction
            f'* * {_data_path}/{era}/{jec_tag}_L1FastJet_AK4PFchs.jec.txt',
        ]
        correction_list_JER = []
        if is_mc:
            common_files = [    
                # Jet Energy Resolution
                f'* * {_data_path}/{era}/RegroupedV2_{jec_tag}_UncertaintySources_AK4PFchs.junc.txt',
                f'* * {_data_path}/{era}/{jer_tag}_PtResolution_AK4PFchs.jr.txt',
                f'* * {_data_path}/{era}/{jer_tag}_SF_AK4PFchs.jersf.txt',
            ]
            correction_list_L123 += common_files
            correction_list_JER += common_files
        jec_name_map.update({'ptGenJet': 'pt_gen'})

        extract_L1.add_weight_sets(correction_list_L1)
        extract_L1.finalize()
        evaluator_L1 = extract_L1.make_evaluator()
        jec_inputs_L1 = {
            name: evaluator_L1[name] for name in dir(evaluator_L1)
        }
        self.jec_stack_L1 = JECStack(jec_inputs_L1)
        self.jec_factory_L1 = CorrectedJetsFactory(jec_name_map, self.jec_stack_L1)
        
        extract_JER.add_weight_sets(correction_list_JER)
        extract_JER.finalize()
        evaluator_JER = extract_JER.make_evaluator()
        jec_inputs_JER = {
            name: evaluator_JER[name] for name in dir(evaluator_JER)
        }
        self.jec_stack_JER = JECStack(jec_inputs_JER)
        self.jec_factory_JER = CorrectedJetsFactory(jec_name_map, self.jec_stack_JER)
        
        extract_L123.add_weight_sets(correction_list_L123)
        extract_L123.finalize()
        evaluator_L123 = extract_L123.make_evaluator()
        jec_inputs_L123 = {
            name: evaluator_L123[name] for name in dir(evaluator_L123)
        }
        self.jec_stack_L123 = JECStack(jec_inputs_L123)
        self.jec_factory_L123 = CorrectedJetsFactory(jec_name_map, self.jec_stack_L123)
        self.met_factory = CorrectedMETFactory(jec_name_map)

    
    def corrected_jets_L123(self, jets, event_rho, lazy_cache, pt_gen=None):
        jet_pt_L123 = self.jec_factory_L123.build(
            add_jme_variables(jets, event_rho, pt_gen),
            lazy_cache 
        )
        emFraction = jet_pt_L123.chEmEF + jet_pt_L123.neEmEF
        mask_jec = (jet_pt_L123['pt'] > 15) & (emFraction <= 0.9)
        selected_jets_L123 = ak.mask(jet_pt_L123, mask_jec)
        selected_jets_L123['pt'] = selected_jets_L123['pt'] * (1 - jets.muonSubtrFactor)
        return selected_jets_L123
    
    def corrected_jets_L1(self, jets, event_rho, lazy_cache, pt_gen=None):
        jet_pt_L1 = self.jec_factory_L1.build(
            add_jme_variables(jets, event_rho, pt_gen),
            lazy_cache 
        )
        jet_pt_L1['pt'] = jet_pt_L1['pt'] * (1 - jets.muonSubtrFactor)
        return jet_pt_L1
    
    def corrected_jets_jer(self, jets, event_rho, lazy_cache):
        jets = add_jme_variables(jets, event_rho)
        jets['pt_raw'  ] = jets.pt 
        jets['mass_raw'] = jets.mass 
        return self.jec_factory_JER.build(
            jets,
            lazy_cache 
        )
        
    def corrected_met(self, met, jets_L123, jets_L1, event_rho, lazy_cache):
        jets_L123 = add_jme_variables(jets_L123, event_rho)
        jets_L123['pt'      ] = -jets_L123.pt 
        jets_L123['pt_raw'  ] = -jets_L1.pt 
        
        return self.met_factory.build(
            met,
            jets_L123,
            lazy_cache=lazy_cache 
        )

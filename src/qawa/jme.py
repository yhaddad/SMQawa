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


def add_jme_variables(jets, events_rho):
    jets['pt_raw'  ] = (1 - jets.rawFactor) * jets.pt
    jets['mass_raw'] = (1 - jets.rawFactor) * jets.mass
    if 'matched_gen' in jets.fields:
        jets['pt_gen'  ] = ak.values_astype(ak.fill_none(jets.matched_gen.pt, 0), np.float32)
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
        extract = extractor()

        correction_list = [
            # Jet Energy Correction
            f'* * {_data_path}/{era}/{jec_tag}_L1FastJet_AK4PFchs.jec.txt',
            f'* * {_data_path}/{era}/{jec_tag}_L2L3Residual_AK4PFchs.jec.txt',
            f'* * {_data_path}/{era}/{jec_tag}_L2Relative_AK4PFchs.jec.txt',
            f'* * {_data_path}/{era}/{jec_tag}_L3Absolute_AK4PFchs.jec.txt',
        ]
        if is_mc:
            correction_list += [    
                # Jet Energy Resolution
                f'* * {_data_path}/{era}/RegroupedV2_{jec_tag}_UncertaintySources_AK4PFchs.junc.txt',
                f'* * {_data_path}/{era}/{jer_tag}_PtResolution_AK4PFchs.jr.txt',
                f'* * {_data_path}/{era}/{jer_tag}_SF_AK4PFchs.jersf.txt',
            ]
            jec_name_map.update({'ptGenJet': 'pt_gen'})

        extract.add_weight_sets(correction_list)
        extract.finalize()
        evaluator = extract.make_evaluator()
        jec_inputs = {
            name: evaluator[name] for name in dir(evaluator)
        }
        self.jec_stack = JECStack(jec_inputs)
        self.jec_factory = CorrectedJetsFactory(jec_name_map, self.jec_stack)
        self.met_factory = CorrectedMETFactory(jec_name_map)

    def corrected_jets(self, jets, event_rho, lazy_cache):
        return self.jec_factory.build(
            add_jme_variables(jets, event_rho),
            lazy_cache #events.caches[0]
        )

    def corrected_met(self, met, jets, event_rho, lazy_cache):
        return self.met_factory.build(
            met,
            add_jme_variables(jets, event_rho),
            lazy_cache=lazy_cache # events.caches[0]
        )

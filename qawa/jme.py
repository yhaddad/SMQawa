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


jec_name_map = {
    'JetPt': 'pt',
    'JetMass': 'mass',
    'JetEta': 'eta',
    'JetA': 'area',
    'ptGenJet': 'pt_gen',
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
    jets['pt_gen'  ] = ak.values_astype(ak.fill_none(jets.matched_gen.pt, 0), np.float32)
    jets['rho'     ] = ak.broadcast_arrays(events_rho, jets.pt)[0]
    return jets

class JMEUncertainty:
    def __init__(
        self, 
        jec_tag: str = 'Summer19UL18_V5_MC',
        jer_tag: str = 'Summer19UL18_JRV2_MC',
    ):
        extract = extractor()
        extract.add_weight_sets([
            # Jet Energy Correction
            f'* * data/{jec_tag}/{jec_tag}_L1FastJet_AK4PFchs.jec.txt',
            f'* * data/{jec_tag}/{jec_tag}_L2L3Residual_AK4PFchs.jec.txt',
            f'* * data/{jec_tag}/{jec_tag}_L2Relative_AK4PFchs.jec.txt',
            f'* * data/{jec_tag}/{jec_tag}_L3Absolute_AK4PFchs.jec.txt',
            f'* * data/{jec_tag}/RegroupedV2_{jec_tag}_UncertaintySources_AK4PFchs.junc.txt',
            # Jet Energy Resolution
            f'* * data/{jec_tag}/{jer_tag}_PtResolution_AK4PFchs.jr.txt',
            f'* * data/{jec_tag}/{jer_tag}_SF_AK4PFchs.jersf.txt',
        ])
        
        extract.finalize()
        evaluator = extract.make_evaluator()
        jec_inputs = {
            name: evaluator[name] for name in dir(evaluator)
        }
        for n in jec_inputs.keys(): print(n)
        self.jec_stack = JECStack(jec_inputs)
        self.jec_factory = CorrectedJetsFactory(jec_name_map, self.jec_stack)
        self.met_factory = CorrectedMETFactory(jec_name_map)
        
    def corrected_jets(self, jets, event_rho):
        return self.jec_factory.build(
            add_jme_variables(jets, event_rho), 
            events.caches[0]
        )
    
    def corrected_met(self, met, jets, event_rho):
        return self.met_factory.build(
            met, 
            add_jme_variables(jets, event_rho),
            lazy_cache=events.caches[0]
        )


import awkward as ak
import numpy as np
import uproot
import hist

from coffea import processor
from coffea import nanoevents
from coffea.nanoevents.methods import candidate

from coffea.analysis_tools import Weights, PackedSelection
from coffea.lumi_tools import LumiMask


from qawa.btag   import BTVCorrector, btag_id
from qawa.jme    import JMEUncertainty, update_collection
from qawa.common import pileup_weights


def build_leptons(muons, electrons):
    # select tight/loose muons
    tight_muons_mask = (
        (muons.pt             >  20. ) &
        (np.abs(muons.eta)    <  2.4 ) &
        (np.abs(muons.dxy)    <  0.02) &
        (np.abs(muons.dz )    <  0.1 ) &
        (muons.pfRelIso04_all <= 0.15) & 
        muons.tightId
    )
    tight_muons = muons[tight_muons_mask]
    loose_muons = muons[
        ~tight_muons_mask &
        (muons.pt            >  7.  ) &
        (np.abs(muons.eta)   <  2.4 ) &
        (muons.pfRelIso04_all<= 0.15) &
        muons.looseId   
    ]
    # select tight/loose electron
    tight_electrons_mask = (
        (electrons.pt           > 20.) &
        (np.abs(electrons.eta)  < 2.5) &
        electrons.mvaFall17V2Iso_WP90
    )
    tight_electrons = electrons[tight_electrons_mask]
    loose_electrons = electrons[
        ~tight_electrons_mask &
        (electrons.pt           > 7. ) &
        (np.abs(electrons.eta)  < 2.5) &
        electrons.mvaFall17V2Iso_WPL
    ]
    # contruct a lepton object
    tight_leptons = ak.with_name(ak.concatenate([tight_muons, tight_electrons], axis=1), 'PtEtaPhiMCandidate')
    loose_leptons = ak.with_name(ak.concatenate([loose_muons, loose_electrons], axis=1), 'PtEtaPhiMCandidate')

    return tight_leptons, loose_leptons


def build_photons(photons, jets):
    base = (
        (photon.pt          > 20. ) & 
        (np.abs(photon.eta) < 2.5 )
    )
    tight_photons = photons[selction & photon.mvaID_WP90]
    loose_photons = photons[selction & photon.mvaID_WP80 & ~photon.mvaID_WP90]


class zz2l2nu_inclusive(processor.ProcessorABC):
    def __init__(self,
            model_2j: str = 'nn2j-mandalorian.onnx', 
            model_3j: str = 'nn3j-mandalorian.onnx', 
            era: str = '2018'
        ):
        self._era = era
        
        jec_tag = ''
        jer_tag = ''
        if self._era == '2018':
            jec_tag = 'Summer19UL18_V5_MC'
            jer_tag = 'Summer19UL18_JRV2_MC'
        elif:
            jec_tag = 'Summer19UL17_V5_MC'
            jer_tag = 'Summer19UL17_JRV2_MC'
        else:
            print('error')
        
        self.zmass = 91.1873 # GeV 
        self._btag = BTVCorrector(era=era)
        self._jmeu = JMEUncertainty(jec_tag, jer_tag)
        self._purw = pileup_weights(era=self._era)
        
        _data_path = 'qawa/data/json'
        self._json = {
            '2017': LumiMask(f'{_data_path}/{era}/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt'),
            '2018': LumiMask(f'{_data_path}/{era}/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt'),
        }
        
        self.build_histos = lambda: {
            'MT': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(20, 0, 500, name="MT", label="Reco $M_{T}$ (GeV)"),
                hist.storage.Weight()
            ), 
            'MET': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(20, 0, 500, name="MET", label="Reco $M_{T}$ (GeV)"),
                hist.storage.Weight()
            )
            
        }
        
    def process_shift(self, event, shift_name:str=''):
        dataset = event.metadata['dataset']
        is_data = event.metadata.get("is_data")
        selection = PackedSelection()
        weights = Weights(len(events), storeIndividual=True)
        
        histos = self.build_histos()
        
        if is_data:
            selection.add('lumimask', self._json[self._era](event.run, event.luminosityBlock))
        else:
            selection.add('lumimask', np.ones(len(event), dtype='bool'))
        
        tight_lep, loose_lep = build_leptons(
            event.Muon,
            event.Electron
        )
        
        ntight_lep = ak.num(tight_lep)
        nloose_lep = ak.num(loose_lep)
        
        jets = event.Jet
        overlap_leptons = ak.any(
            jets.metric_table(tight_lep) <= 0.4,
            axis=2
        )
        
        jet_mask = (
            ~overlap_leptons & 
            (jets.pt>30.0) & 
            (np.abs(jets.eta) < 4.7) & 
            (jets.jetId >= 6) # tight JetID 7(2016) and 6(2017/8)
        )

        jet_btag = (event.Jet.btagDeepFlavB > btag_id('L', self._era))
        
        ngood_jets  = ak.num(jets[~jet_btag & jet_mask])
        ngood_bjets = ak.num(jets[jet_btag & jet_mask & (np.abs(jets.eta)<2.4)])
        
        selection.add('0bjet', ngood_bjets ==0 )
        selection.add('1bjet', ngood_bjets >=0 ) # at least
        selection.add('0jet' , ngood_jets  ==0 )
        selection.add('1jet' , ngood_jets  ==1 )
        selection.add('2jet' , ngood_jets  >=2 ) # at least
        
        # 2L quantities
        def z_lepton_pair(leptons):
            pair = ak.combinations(leptons, 2, axis=1, fields=['l1', 'l2'])
            mass = (pair.l1 + pair.l2).mass
            cand = ak.local_index(mass, axis=1) == ak.argmin(np.abs(mass - self.zmass), axis=1)

            extra_lepton = leptons[(
                ~ak.any(leptons.metric_table(pair[cand].l1) <= 0.01, axis=2) & 
                ~ak.any(leptons.metric_table(pair[cand].l2) <= 0.01, axis=2) )
            ]
            return pair[cand], extra_lepton, cand
        
        dilep, extra_lep, z_cand_mask = z_lepton_pair(tight_lep)
        
        lead_lep = ak.firsts(ak.where(dilep.l1.pt >  dilep.l2.pt, dilep.l1, dilep.l2),axis=1)
        subl_lep = ak.firsts(ak.where(dilep.l1.pt <= dilep.l2.pt, dilep.l1, dilep.l2),axis=1)
        
        dilep_p4 = (lead_lep + subl_lep)
        dilep_m  = dilep_p4.mass
        dilep_pt = dilep_p4.pt
        delta_phi_ll_met = dilep_p4.delta_phi(event.MET)
        dilep_mt = np.sqrt(2 * dilep_pt * event.MET.pt * (1 - np.cos(delta_phi_ll_met)))
        
        # high level observables
        p4_met = ak.zip(
            {
                "pt": events.MET.pt,
                "eta": ak.zeros_like(events.MET.pt),
                "phi": events.MET.phi,
                "mass": ak.zeros_like(events.MET.pt),
                "charge": ak.zeros_like(events.MET.pt),
            },
            with_name="PtEtaPhiMCandidate",
            behavior=candidate.behavior,
        )
        
        emu_met = ak.firsts(extra_lep, axis=1) + p4_met
        
        dphi_ll = lead_lep.delta_phi(subl_lep)
        deta_ll = np.abs(lead_lep.eta - subl_lep.eta)
        dR_ll   = dilep.l1.delta_r(dilep.l2)
        dphi_met_ll = ak.where(ntight_lep==3, dilep_p4.delta_phi(p4_met), dilep_p4.delta_phi(emu_met))
        vector_balance = ak.where(ntight_lep==3, (p4_met - dilep_p4).pt/dilep_p4.pt, (emu_met - dilep_p4).pt/dilep_p4.pt)
        scalar_balance = ak.where(ntight_lep==3, p4_met.pt/dilep_p4.pt, emu_met.pt/dilep_p4.pt)

        # build selections
        selection.add('2leptons', (ntight_lep==2) & (nloose_lep==0) & (ak.firsts(tight_lep).pt>25))
        selection.add('3leptons', (ntight_lep==3) & (nloose_lep==0) & (ak.firsts(tight_lep).pt>25))
        selection.add('4leptons', ((ntight_lep + nloose_lep) == 4 ) & (ak.firsts(tight_lep).pt>25))
        
        selection.add('OSSF', ak.fill_none((lead_lep.pdgId + subl_lep.pdgId)==0, False))
        selection.add('OF'  , ak.fill_none(np.abs(lead_lep.pdgId) != np.abs(subl_lep.pdgId), False))
        
        # kinematic selections
        selection.add('dilep_pt' , dilep_pt> 55 )
        selection.add('is_zmass' , np.abs(dilep_m - self.zmass) < 15)
        selection.add('met_cut'  , np.where(ngood_jets<2, p4_met.pt>100, p4_met.pt>120))
        selection.add('dphimetll', np.abs(dphi_met_ll) > 1.0 )
        selection.add('emu_met', emu_met.pt > 70)
        
        # Now adding weights
        if not is_data:
            weights.add('genweight', events.genWeight)
            self._btag.append_btag_sf(jets, weights)
            self._purw.append_pileup_weight(weights, events.Pileup.nPU)
        
        # selections
        common_sel = ['2leptons', 'OSSF', 'is_zmass', 'met_cut', 'dphimetll', 'dilep_pt',]
        channels = {
            # inclusive categories
            'catSR_0J': common_sel + ['0jet'],
            'catSR_1J': common_sel + ['1jet'], 
            'catSR_2J': common_sel + ['2jet'], 
            'cat3L': ['2leptons', 'OSSF', 'is_zmass', 'emu_met', 'dphimetll', 'dilep_pt'],
            'catEM': ['2leptons', 'OF', 'is_zmass', 'met_cut', 'dphimetll', 'dilep_pt'],
            # VBS categories
            # 'catSR_0J_VBS': common_sel + ['0jet', 'vbs'],
            # 'catSR_1J_VBS': common_sel + ['1jet', 'vbs'], 
            # 'catSR_2J_VBS': common_sel + ['2jet', 'vbs'], 
            # 'cat3L_VBS'   : ['2leptons', 'OSSF', 'is_zmass', 'met_cut', 'dphimetll', 'dilep_pt'],
            # 'catEM_VBS'   : ['2leptons', 'OSSF', 'is_zmass', 'met_cut', 'dphimetll', 'dilep_pt'],
            # 'catTT_VBS'   : ['2leptons', 'OSSF', 'is_zmass', 'met_cut', 'dphimetll', 'dilep_pt'],
            # 'catNR_VBS'   : ['2leptons', 'OSSF', 'is_zmass', 'met_cut', 'dphimetll', 'dilep_pt'],
            # 'preselection': ['2leptons', 'OSSF', 'is_zmass'], 
            'no_selection': []
        }
        
        if shift_name is None:
            systematics = [None] + list(weights.variations)
        else:
            systematics = [shift_name]
            
        def format_variable(variable, cut):
            if cut is None:
                return ak.to_numpy(ak.fill_none(variable, np.nan))
            else:
                return ak.to_numpy(ak.fill_none(variable[cut], np.nan))
        
        def histogram_filler(ch, syst, _weight=None):
            sel_ = channels[ch]
            cut =  selection.all(*sel_)
            
            systname = 'nominal' if syst is None else syst
            
            if _weight is None: 
                if syst in weights.variations:
                    weight = weights.weight(modifier=syst)[cut]
                else:
                    weight = weights.weight()[cut]
            else:
                weight = weights.weight()[cut] * _weight[cut]
            
            histos['MT'].fill(
                channel=ch,
                systematic=systname,
                MT=format_variable(dilep_mt, cut),
                weight=weight,
            )
            histos['MET'].fill(
                channel=ch,
                systematic=systname,
                MET=format_variable(dilep_mt, cut), 
                weight=weight,
            )
                
            
        for ch in channels:
            cut = selection.all(*channels[ch])
            for sys in systematics:
                histogram_filler(ch, sys)
            # if shift_name is None and 'LHEWeight' in event.fields:
            #     for c in events.LHEWeight[1:]:
            #         histogram_filler(ch, f'LHEWeight_{c}', events.LHEWeight[c])
                
        return {dataset: histos}
        
    def process(self, event: processor.LazyDataFrame):
        dataset_name = event.metadata['dataset']
        is_data = event.metadata.get("is_data")
        
        if is_data:
            return self.process_shift(event, None)
        
        jets = self._jmeu.corrected_jets(event.Jet, event.fixedGridRhoFastjetAll, event.caches[0])
        met  = self._jmeu.corrected_met(event.MET, jets, event.fixedGridRhoFastjetAll, event.caches[0])
        
        shifts = [
            ({"Jet": jets               , "MET": met                }, None     ),
            ({"Jet": jets.JES_Total.up  , "MET": met.JES_Total.up   }, "JESUp"  ),
            ({"Jet": jets.JES_Total.down, "MET": met.JES_Total.down }, "JESDown"),
            ({"Jet": jets.JER.up        , "MET": met.JER.up         }, "JERUp"  ),
            ({"Jet": jets.JER.down      , "MET": met.JER.down       }, "JERDown"),
            ({"Jet": jets, "MET": met.MET_UnclusteredEnergy.up      }, "UESUp"  ),
            ({"Jet": jets, "MET": met.MET_UnclusteredEnergy.down    }, "UESDown"),
            ({"Jet": jets, "MET": met, "Electron": Electron.})
        ]
        
        shifts = [
            self.process_shift(
                update_collection(event, collections), 
                name
            ) for collections, name in shifts
        ]
        return processor.accumulate(shifts)
    
    def postprocess(self, accumulator):
        return accumulator


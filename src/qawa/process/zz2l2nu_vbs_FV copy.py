import awkward as ak
import numpy as np
import scipy.interpolate as interp
from scipy import stats as st
import uproot
import pickle
import hist
import yaml
import copy
import os
import re
import gzip
from coffea import processor
from coffea import nanoevents
from coffea.nanoevents.methods import candidate
from coffea.nanoevents.methods import nanoaod
from coffea.analysis_tools import Weights, PackedSelection
from coffea.lumi_tools import LumiMask

from qawa.roccor import rochester_correction
from qawa.leptonsSF import LeptonScaleFactors
from qawa.jetPU import jetPUScaleFactors
from qawa.tauSF import tauIDScaleFactors
from qawa.btag import BTVCorrector, btag_id
from qawa.jme_gh import JMEUncertainty, update_collection
from qawa.gen_match import delta_r2, find_best_match
from qawa.ddr import dataDrivenDYRatio
from qawa.common import pileup_weights, ewk_corrector, met_phi_xy_correction, theory_ps_weight, theory_pdf_weight, trigger_rules

def build_leptons(muons, electrons):
    # select tight/loose muons
    tight_muons_mask = (
        (muons.pt             >  30. ) &
        (np.abs(muons.eta)    <  2.5 ) 
    )
    tight_muons = muons[tight_muons_mask]
    # select tight/loose electron
    tight_electrons_mask = (
        (electrons.pt           > 30.) &
        (np.abs(electrons.eta)  < 2.5) 
    )
    tight_electrons = electrons[tight_electrons_mask]
    # contruct a lepton object
    tight_leptons = ak.with_name(ak.concatenate([tight_muons, tight_electrons], axis=1), 'PtEtaPhiMCandidate')

    tight_sorted_index = ak.argsort(tight_leptons.pt,ascending=False)

    tight_leptons = tight_leptons[tight_sorted_index]

    return tight_leptons


class zzinc_processor(processor.ProcessorABC):
    # EWK corrections process has to be define before hand, it has to change when we move to dask
    def __init__(self, era: str ='2018', isDY=False, dd='SR', ewk_process_name=None, run_period: str = ''): 
        self._era = era
        self._isDY = isDY
        self._ddtype = dd
        if 'APV' in self._era:
            self._isAPV = True
            self._era = re.findall(r'\d+', self._era)[0] 
        else:
            self._isAPV = False
        
        jec_tag = ''
        jer_tag = ''
        if len(run_period)==0:
            if self._era == '2016':
                if self._isAPV:
                    jec_tag = 'Summer19UL16APV_V7_MC'
                    jer_tag = 'Summer20UL16APV_JRV3_MC'
                else:
                    jec_tag = 'Summer19UL16_V7_MC'
                    jer_tag = 'Summer20UL16_JRV3_MC'
            elif self._era == '2017':
                jec_tag = 'Summer19UL17_V5_MC'
                jer_tag = 'Summer19UL17_JRV2_MC'
            elif self._era == '2018':
                jec_tag = 'Summer19UL18_V5_MC'
                jer_tag = 'Summer19UL18_JRV2_MC'
            else:
                print('error')
        else:
            if self._era == '2016':
                if self._isAPV:
                    if run_period in ['B', 'C', 'D']:
                        jec_tag = 'Summer19UL16APV_RunBCD_V7_DATA'
                    else:
                        jec_tag = 'Summer19UL16APV_RunEF_V7_DATA'
                else:
                    jec_tag = 'Summer19UL16_RunFGH_V7_DATA'
            elif self._era == '2017':
                jec_tag = f'Summer19UL17_Run{run_period}_V5_DATA'
            elif self._era == '2018':
                jec_tag = f'Summer19UL18_Run{run_period}_V5_DATA'
            else:
                print('error')
        
        self.btag_wp = 'M'
        self.jetPU_wp = 'M'
        self.tauIDvsjet_wp = 'Medium'
        self.tauIDvse_wp = 'VVLoose'
        self.tauIDvsmu_wp = 'VLoose'
        self.zmass = 91.1873 # GeV 
        self._btag = BTVCorrector(era=self._era, wp=self.btag_wp, isAPV=self._isAPV)
        self._jmeu = JMEUncertainty(jec_tag, jer_tag, era=self._era, is_mc=(len(run_period)==0))
        self._purw = pileup_weights(era=self._era)
        self._leSF = LeptonScaleFactors(era=self._era, isAPV=self._isAPV)
        self._jpSF = jetPUScaleFactors(era=self._era, wp=self.jetPU_wp, isAPV=self._isAPV)
        self._tauID= tauIDScaleFactors(era=self._era, vsjet_wp=self.tauIDvsjet_wp,vse_wp=self.tauIDvse_wp, vsmu_wp=self.tauIDvsmu_wp, isAPV=self._isAPV)
        
        _data_path = 'qawa/data'
        _data_path = os.path.join(os.path.dirname(__file__), '../data')
            
        with open(f'{_data_path}/eft-names.dat') as eft_file:
            self._eftnames = [n.strip() for n in eft_file.readlines()]

        with uproot.open(f'{_data_path}/trigger_sf/histo_triggerEff_sel0_{self._era}.root') as _fn:
            _hvalue = np.dstack([_fn[_hn].values() for _hn in _fn.keys()] + [np.ones((7,7))])
            _herror = np.dstack([np.sqrt(_fn[_hn].variances()) for _hn in _fn.keys()] + [np.zeros((7,7))])
            self.trig_sf_map = np.stack([_hvalue, _herror], axis=-1)
        

        self.ewk_process_name = ewk_process_name
        if self.ewk_process_name is not None:
            self.ewk_corr = ewk_corrector(process=ewk_process_name)

        self.build_histos = lambda: {
            'met_pt': hist.Hist(
                hist.axis.StrCategory([], name="channel"   , growth=True),
                hist.axis.StrCategory([], name="systematic", growth=True), 
                hist.axis.Regular(50, 0, 20000, name="met_pt", label=r"$p_{T}^{miss}$ (GeV)"),
                hist.storage.Weight()
            ),
        }

    

    def process_shift(self, event, shift_name:str=''):
        dataset = event.metadata['dataset']
        is_data = event.metadata.get("is_data")
        selection = PackedSelection()
        weights = Weights(len(event), storeIndividual=True)
        histos = self.build_histos()
       
        # high level observables
        met=event.GenMET
        p4_met = ak.zip(
            {
                "pt": met.pt,
                "eta": ak.zeros_like(met.pt),
                "phi": met.phi,
                "mass": ak.zeros_like(met.pt),
                "charge": ak.zeros_like(met.pt),
            },
            with_name="PtEtaPhiMCandidate",
            behavior=candidate.behavior,
        )
        gen_leptons = event.GenDressedLepton
        gen_leptons["charge"] = ak.where(gen_leptons.pdgId < 0, 1, -1)
        gen_muons = gen_leptons[abs(gen_leptons.pdgId) == 13]
        gen_electrons = gen_leptons[abs(gen_leptons.pdgId) == 11]
        tight_lep = build_leptons(
            gen_muons,
            gen_electrons
        )
        ntight_lep = ak.num(tight_lep)
        
        jets = event.GenJet

        jet_mask = (
            (jets.pt>40.0) & 
            (np.abs(jets.eta) < 4.7) 
        )
        
        
        good_jets = jets[jet_mask]
        sorted_indices = np.argsort(-good_jets.pt)
        good_jets = good_jets[sorted_indices]
        
        lead_jet = ak.firsts(good_jets)
        subl_jet = ak.firsts(good_jets[lead_jet.delta_r(good_jets)>0.01])
        third_jet = ak.firsts(good_jets[(lead_jet.delta_r(good_jets)>0.01) & (subl_jet.delta_r(good_jets)>0.01)])


        dijet_mass = (lead_jet + subl_jet).mass
        dijet_deta = np.abs(lead_jet.eta - subl_jet.eta)
        event['dijet_mass'] = dijet_mass
        event['dijet_deta'] = dijet_deta 
        
        # Define all variables for the GNN
        event['met_pt'  ] = ak.fill_none(p4_met.pt,-99)
        # def z_lepton_pair(leptons):
        #     pair = ak.combinations(leptons, 2, axis=1, fields=['l1', 'l2'])
        #     mass = (pair.l1 + pair.l2).mass
        #     cand = ak.local_index(mass, axis=1) == ak.argmin(np.abs(mass - self.zmass), axis=1)

        #     extra_lepton = leptons[(
        #         ~ak.any(leptons.metric_table(pair[cand].l1) <= 0.01, axis=2) & 
        #         ~ak.any(leptons.metric_table(pair[cand].l2) <= 0.01, axis=2) )
        #     ]
        #     return pair, extra_lepton, cand
        
        # dilep, extra_lep, z_cand_mask = z_lepton_pair(tight_lep)
        pairs     = ak.combinations(tight_lep, 2, axis=1, fields=['l1','l2'])
        pairs     = ak.pad_none(pairs, 1, axis=1)

        lead_lep = pairs.l1[:, 0]
        subl_lep = pairs.l2[:, 0]
        # lead_lep = ak.firsts(ak.where(dilep.l1.pt >  dilep.l2.pt, dilep.l1, dilep.l2),axis=1)
        # subl_lep = ak.firsts(ak.where(dilep.l1.pt <= dilep.l2.pt, dilep.l1, dilep.l2),axis=1)
        
        dilep_p4 = (lead_lep + subl_lep)
        dilep_m  = dilep_p4.mass
        dilep_pt = dilep_p4.pt
        
        # third_lep = ak.firsts(extra_lep, axis=1)
        
        selection.add('low_met_pt', ak.fill_none(p4_met.pt > 130, False)) 
        selection.add('dilep_m'   , ak.fill_none(np.abs(dilep_m - self.zmass) < 15, False))
        selection.add('dilep_pt', ak.fill_none(dilep_pt>60, False))
        selection.add('dijet_deta', ak.fill_none(dijet_deta > 2, False))
        selection.add('dijet_mass_400' , ak.fill_none(dijet_mass >  400, False))
        
        selection.add(
            "require-ossf",
            (ntight_lep==2) &
            (ak.firsts(tight_lep).pt>25) &
            ak.fill_none((lead_lep.pdgId + subl_lep.pdgId)==0, False)
        )
        # Now adding weights
        if not is_data:
            weights.add('genweight', event.genWeight)

            _ones = np.ones(len(weights.weight()))
            if self.ewk_process_name:
                self.ewk_corr.get_weight(
                        event.GenPart,
                        event.Generator.x1,
                        event.Generator.x2,
                        weights
                )
            else:
                weights.add("kEW", _ones, _ones, _ones)
            
            if "PSWeight" in event.fields:
                theory_ps_weight(weights, event.PSWeight)
            else:
                theory_ps_weight(weights, None)
            
            if "LHEPdfWeight" in event.fields:
                theory_pdf_weight(weights, event.LHEPdfWeight)
            else:
                theory_pdf_weight(weights, None)

            if ('LHEScaleWeight' in event.fields) and (len(event.LHEScaleWeight[0]) > 0):
                if len(event.LHEScaleWeight[0]) == 9:
                    weights.add('QCDScale0w'  , _ones, event.LHEScaleWeight[:, 1], event.LHEScaleWeight[:, 7])
                    weights.add('QCDScale1w'  , _ones, event.LHEScaleWeight[:, 3], event.LHEScaleWeight[:, 5])
                    weights.add('QCDScale2w'  , _ones, event.LHEScaleWeight[:, 0], event.LHEScaleWeight[:, 8])
                elif len(event.LHEScaleWeight[0]) == 8:
                    weights.add('QCDScale0w'  , _ones, event.LHEScaleWeight[:, 1], event.LHEScaleWeight[:, 6])
                    weights.add('QCDScale1w'  , _ones, event.LHEScaleWeight[:, 3], event.LHEScaleWeight[:, 4])
                    weights.add('QCDScale2w'  , _ones, event.LHEScaleWeight[:, 0], event.LHEScaleWeight[:, 7])
                elif len(event.LHEScaleWeight[0]) == 18:
                    weights.add('QCDScale0w'  , _ones, event.LHEScaleWeight[:, 2], event.LHEScaleWeight[:, 14])
                    weights.add('QCDScale1w'  , _ones, event.LHEScaleWeight[:, 6], event.LHEScaleWeight[:, 10])
                    weights.add('QCDScale2w'  , _ones, event.LHEScaleWeight[:, 0], event.LHEScaleWeight[:, 16])
                else:
                    print("WARNING: QCD scale variation type not recongnised ... ")
                
            if 'LHEReweightingWeight' in event.fields and 'aQGC' in dataset:
                for i in range(1057):
                    weights.add(f"eft_{self._eftnames[i]}", _ones, event.LHEReweightingWeight[:, i])
            # print(weights.weight())
            # 2017 Prefiring correction weight
            if 'L1PreFiringWeight' in event.fields:
                weights.add("prefiring_weight", event.L1PreFiringWeight.Nom, event.L1PreFiringWeight.Dn, event.L1PreFiringWeight.Up)

        # selections
        channels = {
            "vbs-SR-FV":["require-ossf",'low_met_pt','dilep_m','dijet_deta','dijet_mass_400'],
            "vbs-SR-FV-v1":[],
        }

            
        def _format_variable(variable, cut):
            if cut is None:
                vv = ak.to_numpy(ak.fill_none(variable, np.nan))
                if np.isnan(np.any(vv)):
                    print(" - vv with nan:", vv)
                return ak.to_numpy(ak.fill_none(variable, np.nan))
            else:
                vv = ak.to_numpy(ak.fill_none(variable[cut], np.nan))
                if np.isnan(np.any(vv)):
                    print(" - vv with nan:", vv)
                return ak.to_numpy(ak.fill_none(variable[cut], np.nan))
        
        def _histogram_filler(ch, syst, var, _weight=None):
            sel_ = channels[ch]


            sel_args_ = {
                s.replace('~',''): (False if '~' in s else True) for s in sel_ 
            }
            cut =  selection.require(**sel_args_)

            systname = 'nominal' if syst is None else syst
            if _weight is None: 
                if syst in weights.variations:
                    weight = weights.weight(modifier=syst)[cut]
                else:
                    weight = weights.weight()[cut]
            else:
                weight = weights.weight()[cut] * _weight[cut]

            vv = ak.to_numpy(ak.fill_none(weight, np.nan))
            if np.isnan(np.any(vv)):
                print(f" - {syst} weight nan/inf:", vv[np.isnan(vv)], vv[np.isinf(vv)])
            
            histos[var].fill(
                **{
                    "channel": ch, 
                    "systematic": systname, 
                    var: _format_variable(event[var], cut), 
                    "weight": ak.nan_to_num(weight,nan=1.0, posinf=1.0, neginf=1.0)
                        # ak.ones_like(weight)
                        #ak.nan_to_num(weight,nan=1.0, posinf=1.0, neginf=1.0)
                }
            )
            
            
        if shift_name is None:
            systematics = [None] + list(weights.variations)
        else:
            systematics = [shift_name]
        for ch in channels:
            for sys in systematics:
                # print(sys)
                _histogram_filler(ch, sys, 'met_pt')



                
        return {dataset: histos}
        
    def process(self, event: processor.LazyDataFrame):
        dataset_name = event.metadata['dataset']
        is_data = event.metadata.get("is_data")
         
        
            
        #JES/JER corrections
        rho = event.fixedGridRhoFastjetAll
        cache = event.caches[0]
        if is_data: 
            softjet_gen_pt = None
        else:
            softjet_gen_pt = find_best_match(event.CorrT1METJet,event.GenJet)
        
        softjets_shift_L123 = self._jmeu.corrected_jets_L123(event.CorrT1METJet, rho, cache, softjet_gen_pt)
        softjets_shift_L1 = self._jmeu.corrected_jets_L1(event.CorrT1METJet, rho, cache, softjet_gen_pt)
        
        jets_shift_L123 = self._jmeu.corrected_jets_L123(event.Jet, rho, cache)
        jets_shift_L1 = self._jmeu.corrected_jets_L1(event.Jet, rho, cache)

        jets_col_shift_L123 = ak.concatenate([jets_shift_L123, softjets_shift_L123],axis=1)
        jets_col_shift_L1 = ak.concatenate([jets_shift_L1, softjets_shift_L1],axis=1)
        
        raw_met = event.RawMET
        met_to_correct = event.MET
        met_to_correct["pt"] = raw_met.pt
        met_to_correct["phi"] = raw_met.phi
        jets = self._jmeu.corrected_jets_jer(event.Jet, event.fixedGridRhoFastjetAll, event.caches[0])
        met = self._jmeu.corrected_met(met_to_correct, jets_col_shift_L123, jets_col_shift_L1, event.fixedGridRhoFastjetAll, event.caches[0])

        event = ak.with_field(event, jets, 'Jet')
        event = ak.with_field(event, met, 'MET')
        
        # x-y met shit corrections
        # for the moment I am replacing the met with the corrected met 
        # before doing the JES/JER corrections

        run = event.run 
        npv = event.PV.npvs
        
        met = met_phi_xy_correction(
            event.MET, run, npv, 
            is_mc=not is_data, 
            era=self._era
        )
        event = ak.with_field(event, met, 'MET')

        # Adding scale factors to Muon and Electron fields
        muon = event.Muon 
        electron = event.Electron
        muonSF_nom, muonSF_up, muonSF_down = self._leSF.muonSF(muon)
        elecSF_nom, elecSF_up, elecSF_down = self._leSF.electronSF(electron)
        
        muon['SF'] = muonSF_nom
        muon['SF_up'] = muonSF_up
        muon['SF_down'] = muonSF_down

        electron['SF'] = elecSF_nom
        electron['SF_up'] = elecSF_up
        electron['SF_down'] = elecSF_down

        event = ak.with_field(event, muon, 'Muon')
        event = ak.with_field(event, electron, 'Electron')

        # Apply rochester_correction
        muon=event.Muon
        muonEnUp=event.Muon
        muonEnDown=event.Muon
        muon_pt,muon_pt_roccorUp,muon_pt_roccorDown=rochester_correction(is_data).apply_rochester_correction (muon)
        
        muon['pt'] = muon_pt
        muonEnUp['pt'] = muon_pt_roccorUp
        muonEnDown['pt'] = muon_pt_roccorDown
        event = ak.with_field(event, muon, 'Muon')
        
        # Electron corrections
        electronEnUp=event.Electron
        electronEnDown=event.Electron

        electronEnUp  ['pt'] = event.Electron['pt'] + event.Electron.energyErr/np.cosh(event.Electron.eta)
        electronEnDown['pt'] = event.Electron['pt'] - event.Electron.energyErr/np.cosh(event.Electron.eta)	

        # define all the shifts
        shifts = [
            # Jets
            ({"Jet": jets                             , "MET": met                               }, None                  ),
            ({"Jet": jets.JES_Total.up                , "MET": met.JES_Total.up                  }, "JESUp"               ),
            ({"Jet": jets.JES_Total.down              , "MET": met.JES_Total.down                }, "JESDown"             ),
            ({"Jet": jets.JES_Absolute.up             , "MET": met.JES_Absolute.up               }, "JES_AbsoluteUp"      ),
            ({"Jet": jets.JES_Absolute.down           , "MET": met.JES_Absolute.down             }, "JES_AbsoluteDown"    ),
            ({"Jet": jets.JES_BBEC1.up                , "MET": met.JES_BBEC1.up                  }, "JES_BBEC1Up"         ),
            ({"Jet": jets.JES_BBEC1.down              , "MET": met.JES_BBEC1.down                }, "JES_BBEC1Down"       ),
            ({"Jet": jets.JES_EC2.up                  , "MET": met.JES_EC2.up                    }, "JES_EC2Up"           ),
            ({"Jet": jets.JES_EC2.down                , "MET": met.JES_EC2.down                  }, "JES_EC2Down"         ),
            ({"Jet": jets.JES_FlavorQCD.up            , "MET": met.JES_FlavorQCD.up              }, "JES_FlavorQCDUp"     ),
            ({"Jet": jets.JES_FlavorQCD.down          , "MET": met.JES_FlavorQCD.down            }, "JES_FlavorQCDDown"   ),
            ({"Jet": jets.JES_HF.up                   , "MET": met.JES_HF.up                     }, "JES_HFUp"            ),
            ({"Jet": jets.JES_HF.down                 , "MET": met.JES_HF.down                   }, "JES_HFDown"          ),
            ({"Jet": jets.JES_RelativeBal.up          , "MET": met.JES_RelativeBal.up            }, "JES_RelativeBalUp"   ),
            ({"Jet": jets.JES_RelativeBal.down        , "MET": met.JES_RelativeBal.down          }, "JES_RelativeBalDown" ),
            ({"Jet": jets.JER.up                      , "MET": met.JER.up                        }, "JERUp"               ),
            ({"Jet": jets.JER.down                    , "MET": met.JER.down                      }, "JERDown"             ),
            ({"Jet": jets                             , "MET": met.MET_UnclusteredEnergy.up      }, "UESUp"               ),
            ({"Jet": jets                             , "MET": met.MET_UnclusteredEnergy.down    }, "UESDown"             ), 
            # year dependent systematics
            ({"Jet": getattr(jets,f'JES_BBEC1_{self._era}').up     , "MET": getattr(met,f'JES_BBEC1_{self._era}').up      }, f"JES_BBEC1{self._era}Up"  ),
            ({"Jet": getattr(jets,f'JES_BBEC1_{self._era}').down   , "MET": getattr(met,f'JES_BBEC1_{self._era}').down    }, f"JES_BBEC1{self._era}Down"),
            ({"Jet": getattr(jets,f'JES_Absolute_{self._era}').up  , "MET": getattr(met,f'JES_Absolute_{self._era}').up   }, f"JES_Absolute{self._era}Up"  ),
            ({"Jet": getattr(jets,f'JES_Absolute_{self._era}').down, "MET": getattr(met,f'JES_Absolute_{self._era}').down }, f"JES_Absolute{self._era}Down"),
            ({"Jet": getattr(jets,f'JES_EC2_{self._era}').up       , "MET": getattr(met,f'JES_EC2_{self._era}').up        }, f"JES_EC2{self._era}Up"  ),
            ({"Jet": getattr(jets,f'JES_EC2_{self._era}').down     , "MET": getattr(met,f'JES_EC2_{self._era}').down      }, f"JES_EC2{self._era}Down"),
            ({"Jet": getattr(jets,f'JES_HF_{self._era}').up        , "MET": getattr(met,f'JES_HF_{self._era}').up         }, f"JES_HF{self._era}Up"  ),
            ({"Jet": getattr(jets,f'JES_HF_{self._era}').down      , "MET": getattr(met,f'JES_HF_{self._era}').down       }, f"JES_HF{self._era}Down"),
            ({"Jet": getattr(jets,f'JES_RelativeSample_{self._era}').up  , "MET": getattr(met,f'JES_RelativeSample_{self._era}').up   }, f"JES_RelativeSample{self._era}Up"  ),
            ({"Jet": getattr(jets,f'JES_RelativeSample_{self._era}').down, "MET": getattr(met,f'JES_RelativeSample_{self._era}').down }, f"JES_RelativeSample{self._era}Down"),

            
            # Electrons + MET shift (FIXME: shift to be added)
            ({"Electron": electronEnUp  }, "ElectronEnUp"  ),
            ({"Electron": electronEnDown}, "ElectronEnDown"),
            # Muon + MET shifts
            ({"Muon": muonEnUp  }, "MuonRocUp"),
            ({"Muon": muonEnDown}, "MuonRocDown"),
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
# samples ={
#         "ZZTo2E2Nu_TuneCP5_DipoleRecoil_13TeV_powheg_pythia8":{
#             'files': [
#                 "/tmp/hgao/ZZTo2E2Nu-MC_0.root",
# ],
#             'metadata':{
#                 'era': "2018",
#                 'is_data': False
#             }
#         }
#     }
# out_btag1 = processor.run_uproot_job(
#     samples,
#     processor_instance=zzinc_processor(
#         era='2018',
#         ewk_process_name=None,
#         run_period=''),
#     treename='Events',
#     executor=processor.futures_executor,
#     executor_args={
#         "schema": nanoevents.NanoAODSchema,
#         "workers": 15
#     },
#     # chunksize=1000,
#     # maxchunks=3
# )

# print(out_btag1['ZZTo2E2Nu_TuneCP5_DipoleRecoil_13TeV_powheg_pythia8']['met_pt'][{'channel': 'vbs-SR-FV', 'systematic': 'nominal'}].values().sum())
# #print(out_btag1['ZZTo2E2Nu_TuneCP5_DipoleRecoil_13TeV_powheg_pythia8']['met_pt'][{'channel': 'vbs-SR-FV', 'systematic': 'JESUp'}].values().sum())


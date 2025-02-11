import yaml
import numpy as np
from collections import defaultdict
import os
import awkward as ak
import uproot
import coffea
import copy
from coffea import processor
from coffea import nanoevents
#from coffea import hist
import hist
from coffea.nanoevents import NanoEventsFactory
from coffea.nanoevents import NanoAODSchema
from coffea.nanoevents.methods import candidate
from coffea.nanoevents.methods import nanoaod
import matplotlib.pyplot as plt


class trig_processor(processor.ProcessorABC):
    def __init__(self, isMC, era):
        self.isMC = isMC
        self.era = era
        #self.hlts_lep = hlts_lep
        self.hlts_met = ['PFMET100_PFMHT100_IDTight_PFHT60',
                         'PFMET110_PFMHT110_IDTight',
                         'PFMET120_PFMHT120_IDTight',
                         'PFMET120_PFMHT120_IDTight_PFHT60',
                         'PFMET130_PFMHT130_IDTight',
                         'PFMET140_PFMHT140_IDTight',
                         'PFMET200_HBHECleaned',
                         'PFMET200_HBHE_BeamHaloCleaned',
                         'PFMET250_HBHECleaned',
                         'PFMET300_HBHECleaned',
                         'PFMETNoMu100_PFMHTNoMu100_IDTight_PFHT60',
                         'PFMETNoMu110_PFMHTNoMu110_IDTight',
                         'PFMETNoMu120_PFMHTNoMu120_IDTight',
                         'PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60',
                         'PFMETNoMu130_PFMHTNoMu130_IDTight',
                         'PFMETNoMu140_PFMHTNoMu140_IDTight',
                         'PFMETTypeOne100_PFMHT100_IDTight_PFHT60',
                         'PFMETTypeOne110_PFMHT110_IDTight',
                         'PFMETTypeOne120_PFMHT120_IDTight',
                         'PFMETTypeOne120_PFMHT120_IDTight_PFHT60',
                         'PFMETTypeOne130_PFMHT130_IDTight',
                         'PFMETTypeOne140_PFMHT140_IDTight',
                         'PFMETTypeOne200_HBHE_BeamHaloCleaned']
        
        #dataset_axis = hist.Cat("dataset", "") #coffea.hist
        dataset_axis = hist.axis.StrCategory([], name="dataset", label="dataset", growth=True) #hist.Hist
        bins = [20, 25, 30, 35, 40, 50, 60, 70]
        #lead_axis = hist.Bin("lead", "pT lead [GeV]", bins) #coffea.hist
        #trail_axis = hist.Bin("trail", "pT trail [GeV]", bins) #coffea.hist
        lead_axis = hist.axis.Variable(bins, name="lead", label="pT lead [GeV]") #hist.Hist
        trail_axis = hist.axis.Variable(bins, name="trail", label="pT trail [GeV]") #hist.Hist
        
        self._accumulator = processor.dict_accumulator({
            'h_num_MM_EE': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),
            'h_num_ME_EE': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),
            'h_num_EM_EE': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),
            'h_num_EE_EE': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),
            'h_den_MM_EE': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),
            'h_den_ME_EE': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),
            'h_den_EM_EE': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),
            'h_den_EE_EE': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),

            'h_num_MM_EB': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),
            'h_num_ME_EB': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),
            'h_num_EM_EB': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),
            'h_num_EE_EB': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),
            'h_den_MM_EB': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),
            'h_den_ME_EB': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),
            'h_den_EM_EB': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),
            'h_den_EE_EB': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),

            'h_num_MM_BE': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),
            'h_num_ME_BE': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),
            'h_num_EM_BE': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),
            'h_num_EE_BE': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),
            'h_den_MM_BE': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),
            'h_den_ME_BE': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),
            'h_den_EM_BE': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),
            'h_den_EE_BE': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),

            'h_num_MM_BB': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),
            'h_num_ME_BB': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),
            'h_num_EM_BB': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),
            'h_num_EE_BB': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),
            'h_den_MM_BB': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),
            'h_den_ME_BB': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),
            'h_den_EM_BB': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),
            'h_den_EE_BB': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),


        })

        
        _data_path = 'qawa/data'
        _data_path = os.path.join(os.path.dirname(__file__), '../data')
        # print("I am here...")
        with open(f'{_data_path}/HLT_Run3.yaml') as f_yml:
            dict_HLT = yaml.load(f_yml, Loader=yaml.FullLoader)

        hlt_ls = [_hlt.split('HLT_')[-1] for _hlt_ls in dict_HLT[str(era)].values() for _hlt in _hlt_ls]
        self.hlts_lep = hlt_ls #FIXME: add to real processor
        
    @property
    def accumulator(self):
        return self._accumulator

    
    def process(self, events):
        # print("MMM")
        # print(f"Accumulator type: {type(self._accumulator)}")
        # print(f"Accumulator keys: {self._accumulator.keys()}")
        #output = self.accumulator.identity()
        output = self.accumulator.copy()
        #output = copy.deepcopy(self.accumulator)
        # print("G")
        dataset = events.metadata["dataset"]
        # print("Hey...")
        # HLT
        hlt_avail = events.HLT.layout.keys()
        #hlt_avail = events.HLT.fields
        events_MET = self.HLT_MET(events, hlt_avail)
        events_LEP = self.HLT_LEP(events_MET, hlt_avail)
        # print("H")
        # pt arrays
        dic_pt_MET = self.get_pTs_from_events(events_MET)
        dic_pt_LEP = self.get_pTs_from_events(events_LEP)

        output['h_num_MM_EE'].fill(dataset=dataset, lead=dic_pt_LEP['pt_lead_MM_EE'], trail=dic_pt_LEP['pt_trail_MM_EE'])
        output['h_num_ME_EE'].fill(dataset=dataset, lead=dic_pt_LEP['pt_lead_ME_EE'], trail=dic_pt_LEP['pt_trail_ME_EE'])
        output['h_num_EM_EE'].fill(dataset=dataset, lead=dic_pt_LEP['pt_lead_EM_EE'], trail=dic_pt_LEP['pt_trail_EM_EE'])
        output['h_num_EE_EE'].fill(dataset=dataset, lead=dic_pt_LEP['pt_lead_EE_EE'], trail=dic_pt_LEP['pt_trail_EE_EE'])

        output['h_den_MM_EE'].fill(dataset=dataset, lead=dic_pt_MET['pt_lead_MM_EE'], trail=dic_pt_MET['pt_trail_MM_EE'])
        output['h_den_ME_EE'].fill(dataset=dataset, lead=dic_pt_MET['pt_lead_ME_EE'], trail=dic_pt_MET['pt_trail_ME_EE'])
        output['h_den_EM_EE'].fill(dataset=dataset, lead=dic_pt_MET['pt_lead_EM_EE'], trail=dic_pt_MET['pt_trail_EM_EE'])
        output['h_den_EE_EE'].fill(dataset=dataset, lead=dic_pt_MET['pt_lead_EE_EE'], trail=dic_pt_MET['pt_trail_EE_EE'])

        output['h_num_MM_EB'].fill(dataset=dataset, lead=dic_pt_LEP['pt_lead_MM_EB'], trail=dic_pt_LEP['pt_trail_MM_EB'])
        output['h_num_ME_EB'].fill(dataset=dataset, lead=dic_pt_LEP['pt_lead_ME_EB'], trail=dic_pt_LEP['pt_trail_ME_EB'])
        output['h_num_EM_EB'].fill(dataset=dataset, lead=dic_pt_LEP['pt_lead_EM_EB'], trail=dic_pt_LEP['pt_trail_EM_EB'])
        output['h_num_EE_EB'].fill(dataset=dataset, lead=dic_pt_LEP['pt_lead_EE_EB'], trail=dic_pt_LEP['pt_trail_EE_EB'])

        output['h_den_MM_EB'].fill(dataset=dataset, lead=dic_pt_MET['pt_lead_MM_EB'], trail=dic_pt_MET['pt_trail_MM_EB'])
        output['h_den_ME_EB'].fill(dataset=dataset, lead=dic_pt_MET['pt_lead_ME_EB'], trail=dic_pt_MET['pt_trail_ME_EB'])
        output['h_den_EM_EB'].fill(dataset=dataset, lead=dic_pt_MET['pt_lead_EM_EB'], trail=dic_pt_MET['pt_trail_EM_EB'])
        output['h_den_EE_EB'].fill(dataset=dataset, lead=dic_pt_MET['pt_lead_EE_EB'], trail=dic_pt_MET['pt_trail_EE_EB'])

        output['h_num_MM_BE'].fill(dataset=dataset, lead=dic_pt_LEP['pt_lead_MM_BE'], trail=dic_pt_LEP['pt_trail_MM_BE'])
        output['h_num_ME_BE'].fill(dataset=dataset, lead=dic_pt_LEP['pt_lead_ME_BE'], trail=dic_pt_LEP['pt_trail_ME_BE'])
        output['h_num_EM_BE'].fill(dataset=dataset, lead=dic_pt_LEP['pt_lead_EM_BE'], trail=dic_pt_LEP['pt_trail_EM_BE'])
        output['h_num_EE_BE'].fill(dataset=dataset, lead=dic_pt_LEP['pt_lead_EE_BE'], trail=dic_pt_LEP['pt_trail_EE_BE'])

        output['h_den_MM_BE'].fill(dataset=dataset, lead=dic_pt_MET['pt_lead_MM_BE'], trail=dic_pt_MET['pt_trail_MM_BE'])
        output['h_den_ME_BE'].fill(dataset=dataset, lead=dic_pt_MET['pt_lead_ME_BE'], trail=dic_pt_MET['pt_trail_ME_BE'])
        output['h_den_EM_BE'].fill(dataset=dataset, lead=dic_pt_MET['pt_lead_EM_BE'], trail=dic_pt_MET['pt_trail_EM_BE'])
        output['h_den_EE_BE'].fill(dataset=dataset, lead=dic_pt_MET['pt_lead_EE_BE'], trail=dic_pt_MET['pt_trail_EE_BE'])

        output['h_num_MM_BB'].fill(dataset=dataset, lead=dic_pt_LEP['pt_lead_MM_BB'], trail=dic_pt_LEP['pt_trail_MM_BB'])
        output['h_num_ME_BB'].fill(dataset=dataset, lead=dic_pt_LEP['pt_lead_ME_BB'], trail=dic_pt_LEP['pt_trail_ME_BB'])
        output['h_num_EM_BB'].fill(dataset=dataset, lead=dic_pt_LEP['pt_lead_EM_BB'], trail=dic_pt_LEP['pt_trail_EM_BB'])
        output['h_num_EE_BB'].fill(dataset=dataset, lead=dic_pt_LEP['pt_lead_EE_BB'], trail=dic_pt_LEP['pt_trail_EE_BB'])

        output['h_den_MM_BB'].fill(dataset=dataset, lead=dic_pt_MET['pt_lead_MM_BB'], trail=dic_pt_MET['pt_trail_MM_BB'])
        output['h_den_ME_BB'].fill(dataset=dataset, lead=dic_pt_MET['pt_lead_ME_BB'], trail=dic_pt_MET['pt_trail_ME_BB'])
        output['h_den_EM_BB'].fill(dataset=dataset, lead=dic_pt_MET['pt_lead_EM_BB'], trail=dic_pt_MET['pt_trail_EM_BB'])
        output['h_den_EE_BB'].fill(dataset=dataset, lead=dic_pt_MET['pt_lead_EE_BB'], trail=dic_pt_MET['pt_trail_EE_BB'])

        
        return output


    def postprocess(self, accumulator):
        return accumulator
    

    def HLT_MET(self, events, hlt_avail):
        hlt_good = [hlt for hlt in self.hlts_met if hlt in hlt_avail]
        mask_hlt = eval(' | '.join(([f'events.HLT.{hlt}' for hlt in hlt_good])))
        events_MET = events[mask_hlt]
        return events_MET


    def HLT_LEP(self, events, hlt_avail):
        hlt_good = [hlt for hlt in self.hlts_lep if hlt in hlt_avail]
        mask_hlt = eval(' | '.join(([f'events.HLT.{hlt}' for hlt in hlt_good])))
        events_LEP = events[mask_hlt]

        return events_LEP


    def get_good_muons(self, muons):
        muons = muons[abs(muons.eta) < 2.4]
        # muons.pt >= (25 if idx==0 else 20)
        muons = muons[muons.pt > 20]
        muons = muons[muons.tightId] 
        muons = muons[muons.pfRelIso04_all <= 0.15]
        muons = muons[abs(muons.dxy) < 0.045]  
        muons = muons[abs(muons.dz) < 0.2]
        #print("muon")
        return muons

    def get_good_electrons(self, electrons):
        electrons = electrons[(abs(electrons.eta) < 1.4442) | ((abs(electrons.eta) > 1.5660) & (abs(electrons.eta)  < 2.5))]
        # skip 25 GeV
        electrons = electrons[electrons.pt > 20]
        electrons = electrons[electrons.mvaIso_WP90] 
        #print("electron")
        return electrons

    def get_pTs_from_events(self, events):
        # dictionary to be returned
        dic_pt = {}

        # get leptons
        good_Ms = self.get_good_muons(events.Muon)
        good_Es = self.get_good_electrons(events.Electron)

        # at least 2 leptons
        mask_ll = ak.num(good_Ms) + ak.num(good_Es) >= 2
        good_Ls = ak.concatenate([good_Ms[mask_ll], good_Es[mask_ll]], axis=1)
        id_sort = ak.argsort(good_Ls.pt, ascending=False)
        good_Ls = good_Ls[id_sort]
        good_Ls = good_Ls[:,:2]
        #print("inside get pt")

        # pdgID
        id_prod = abs(good_Ls.pdgId[:,0] * good_Ls.pdgId[:,1])

        # ElEl
        mask_ElEl = id_prod==121
        good_ElEl = good_Ls[mask_ElEl]

        mask_0Br = good_ElEl.eta[:,0] < 1.5
        mask_1Br = good_ElEl.eta[:,1] < 1.5
        mask_0Ec = good_ElEl.eta[:,0] > 1.5
        mask_1Ec = good_ElEl.eta[:,1] > 1.5

        mask_BrBr = mask_0Br & mask_1Br
        mask_BrEc = mask_0Br & mask_1Ec
        mask_EcBr = mask_0Ec & mask_1Br
        mask_EcEc = mask_0Ec & mask_1Ec

        dic_pt['pt_lead_EE_BB'] = good_ElEl[mask_BrBr].pt[:,0]
        dic_pt['pt_trail_EE_BB'] = good_ElEl[mask_BrBr].pt[:,1]
        dic_pt['pt_lead_EE_BE'] = good_ElEl[mask_BrEc].pt[:,0]
        dic_pt['pt_trail_EE_BE'] = good_ElEl[mask_BrEc].pt[:,1]
        dic_pt['pt_lead_EE_EB'] = good_ElEl[mask_EcBr].pt[:,0]
        dic_pt['pt_trail_EE_EB'] = good_ElEl[mask_EcBr].pt[:,1]
        dic_pt['pt_lead_EE_EE'] = good_ElEl[mask_EcEc].pt[:,0]
        dic_pt['pt_trail_EE_EE'] = good_ElEl[mask_EcEc].pt[:,1]

        # MuMu
        mask_MuMu = id_prod==169
        good_MuMu = good_Ls[mask_MuMu]

        mask_0Br = good_MuMu.eta[:,0] < 1.5
        mask_1Br = good_MuMu.eta[:,1] < 1.5
        mask_0Ec = good_MuMu.eta[:,0] > 1.5
        mask_1Ec = good_MuMu.eta[:,1] > 1.5

        mask_BrBr = mask_0Br & mask_1Br
        mask_BrEc = mask_0Br & mask_1Ec
        mask_EcBr = mask_0Ec & mask_1Br
        mask_EcEc = mask_0Ec & mask_1Ec

        dic_pt['pt_lead_MM_BB'] = good_MuMu[mask_BrBr].pt[:,0]
        dic_pt['pt_trail_MM_BB'] = good_MuMu[mask_BrBr].pt[:,1]
        dic_pt['pt_lead_MM_BE'] = good_MuMu[mask_BrEc].pt[:,0]
        dic_pt['pt_trail_MM_BE'] = good_MuMu[mask_BrEc].pt[:,1]
        dic_pt['pt_lead_MM_EB'] = good_MuMu[mask_EcBr].pt[:,0]
        dic_pt['pt_trail_MM_EB'] = good_MuMu[mask_EcBr].pt[:,1]
        dic_pt['pt_lead_MM_EE'] = good_MuMu[mask_EcEc].pt[:,0]
        dic_pt['pt_trail_MM_EE'] = good_MuMu[mask_EcEc].pt[:,1]

        # Mix
        mask_mix = id_prod==143
        good_mix = good_Ls[mask_mix]

        # ElMu
        mask_ElMu = abs(good_mix.pdgId[:,0])==11
        good_ElMu = good_mix[mask_ElMu]

        mask_0Br = good_ElMu.eta[:,0] < 1.5
        mask_1Br = good_ElMu.eta[:,1] < 1.5
        mask_0Ec = good_ElMu.eta[:,0] > 1.5
        mask_1Ec = good_ElMu.eta[:,1] > 1.5

        mask_BrBr = mask_0Br & mask_1Br
        mask_BrEc = mask_0Br & mask_1Ec
        mask_EcBr = mask_0Ec & mask_1Br
        mask_EcEc = mask_0Ec & mask_1Ec

        dic_pt['pt_lead_EM_BB'] = good_ElMu[mask_BrBr].pt[:,0]
        dic_pt['pt_trail_EM_BB'] = good_ElMu[mask_BrBr].pt[:,1]
        dic_pt['pt_lead_EM_BE'] = good_ElMu[mask_BrEc].pt[:,0]
        dic_pt['pt_trail_EM_BE'] = good_ElMu[mask_BrEc].pt[:,1]
        dic_pt['pt_lead_EM_EB'] = good_ElMu[mask_EcBr].pt[:,0]
        dic_pt['pt_trail_EM_EB'] = good_ElMu[mask_EcBr].pt[:,1]
        dic_pt['pt_lead_EM_EE'] = good_ElMu[mask_EcEc].pt[:,0]
        dic_pt['pt_trail_EM_EE'] = good_ElMu[mask_EcEc].pt[:,1]

        # MuEl
        mask_MuEl = abs(good_mix.pdgId[:,0])==13
        good_MuEl = good_mix[mask_MuEl]

        mask_0Br = good_MuEl.eta[:,0] < 1.5
        mask_1Br = good_MuEl.eta[:,1] < 1.5
        mask_0Ec = good_MuEl.eta[:,0] > 1.5
        mask_1Ec = good_MuEl.eta[:,1] > 1.5

        mask_BrBr = mask_0Br & mask_1Br
        mask_BrEc = mask_0Br & mask_1Ec
        mask_EcBr = mask_0Ec & mask_1Br
        mask_EcEc = mask_0Ec & mask_1Ec

        dic_pt['pt_lead_ME_BB'] = good_MuEl[mask_BrBr].pt[:,0]
        dic_pt['pt_trail_ME_BB'] = good_MuEl[mask_BrBr].pt[:,1]
        dic_pt['pt_lead_ME_BE'] = good_MuEl[mask_BrEc].pt[:,0]
        dic_pt['pt_trail_ME_BE'] = good_MuEl[mask_BrEc].pt[:,1]
        dic_pt['pt_lead_ME_EB'] = good_MuEl[mask_EcBr].pt[:,0]
        dic_pt['pt_trail_ME_EB'] = good_MuEl[mask_EcBr].pt[:,1]
        dic_pt['pt_lead_ME_EE'] = good_MuEl[mask_EcEc].pt[:,0]
        dic_pt['pt_trail_ME_EE'] = good_MuEl[mask_EcEc].pt[:,1]

        return dic_pt
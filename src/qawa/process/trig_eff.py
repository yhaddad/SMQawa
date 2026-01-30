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
from coffea.nanoevents.methods import nanoaod
from coffea.nanoevents.methods import candidate
ak.behavior.update(candidate.behavior)
import matplotlib.pyplot as plt


class trig_processor(processor.ProcessorABC):
    def __init__(self, isMC, era):
        self.isMC = isMC
        self.era = era
        #self.hlts_lep = hlts_lep
        self.hlts_met = ['PFMET110_PFMHT110_IDTight',
                         'PFMET120_PFMHT120_IDTight',
                         'PFMET120_PFMHT120_IDTight_PFHT60',
                         'PFMET130_PFMHT130_IDTight',
                         'PFMET140_PFMHT140_IDTight',
                         'PFMET200_BeamHaloCleaned',
                         'PFMETTypeOne200_BeamHaloCleaned',
                         'PFMETNoMu110_PFMHTNoMu110_IDTight',
                         'PFMETNoMu120_PFMHTNoMu120_IDTight',
                         'PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60',
                         'PFMETNoMu130_PFMHTNoMu130_IDTight',
                         'PFMETNoMu140_PFMHTNoMu140_IDTight',
                         'PFMETTypeOne140_PFMHT140_IDTight']

        #dataset_axis = hist.Cat("dataset", "") #coffea.hist
        dataset_axis = hist.axis.StrCategory([], name="dataset", label="dataset", growth=True) #hist.Hist
        bins = [20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 60.0, 70.0]
        # print(bins)
        #lead_axis = hist.Bin("lead", "pT lead [GeV]", bins) #coffea.hist
        #trail_axis = hist.Bin("trail", "pT trail [GeV]", bins) #coffea.hist
        lead_axis = hist.axis.Variable(bins, name="lead", label="pT lead [GeV]") #hist.Hist
        trail_axis = hist.axis.Variable(bins, name="trail", label="pT trail [GeV]") #hist.Hist
        dR_axis = hist.axis.Regular(224, 0, 5.6, name="dR", label="$\Delta~R(l, l)$")
        
        self._accumulator = processor.dict_accumulator({

            # 'h_MM_EE_no_trig': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),
            # 'h_MM_BB_no_trig': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),
            # 'h_MM_BE_no_trig': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),
            # 'h_MM_EB_no_trig': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),
            # 'h_EE_EE_no_trig': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),
            # 'h_EE_BB_no_trig': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),

            'h_num_MM': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),
            'h_num_ME': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),
            'h_num_EM': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),
            'h_num_EE': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),
            'h_den_MM': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),
            'h_den_ME': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),
            'h_den_EM': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),
            'h_den_EE': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),


            # 'h_num_MM_deltaR': hist.Hist(dataset_axis, lead_axis, trail_axis, dR_axis, storage=hist.storage.Weight()),
            # 'h_num_ME_deltaR': hist.Hist(dataset_axis, lead_axis, trail_axis, dR_axis, storage=hist.storage.Weight()),
            # 'h_num_EM_deltaR': hist.Hist(dataset_axis, lead_axis, trail_axis, dR_axis, storage=hist.storage.Weight()),
            # 'h_num_EE_deltaR': hist.Hist(dataset_axis, lead_axis, trail_axis, dR_axis, storage=hist.storage.Weight()),
            # 'h_den_MM_deltaR': hist.Hist(dataset_axis, lead_axis, trail_axis, dR_axis, storage=hist.storage.Weight()),
            # 'h_den_ME_deltaR': hist.Hist(dataset_axis, lead_axis, trail_axis, dR_axis, storage=hist.storage.Weight()),
            # 'h_den_EM_deltaR': hist.Hist(dataset_axis, lead_axis, trail_axis, dR_axis, storage=hist.storage.Weight()),
            # 'h_den_EE_deltaR': hist.Hist(dataset_axis, lead_axis, trail_axis, dR_axis, storage=hist.storage.Weight()),


            # different detector regions

            # 'h_num_MM_EE': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),
            # 'h_num_ME_EE': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),
            # 'h_num_EM_EE': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),
            # 'h_num_EE_EE': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),
            # 'h_den_MM_EE': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),
            # 'h_den_ME_EE': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),
            # 'h_den_EM_EE': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),
            # 'h_den_EE_EE': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),

            # 'h_num_MM_EB': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),
            # 'h_num_ME_EB': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),
            # 'h_num_EM_EB': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),
            # 'h_num_EE_EB': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),
            # 'h_den_MM_EB': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),
            # 'h_den_ME_EB': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),
            # 'h_den_EM_EB': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),
            # 'h_den_EE_EB': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),

            # 'h_num_MM_BE': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),
            # 'h_num_ME_BE': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),
            # 'h_num_EM_BE': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),
            # 'h_num_EE_BE': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),
            # 'h_den_MM_BE': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),
            # 'h_den_ME_BE': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),
            # 'h_den_EM_BE': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),
            # 'h_den_EE_BE': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),

            # 'h_num_MM_BB': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),
            # 'h_num_ME_BB': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),
            # 'h_num_EM_BB': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),
            # 'h_num_EE_BB': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),
            # 'h_den_MM_BB': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),
            # 'h_den_ME_BB': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),
            # 'h_den_EM_BB': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),
            # 'h_den_EE_BB': hist.Hist(dataset_axis, lead_axis, trail_axis, storage=hist.storage.Weight()),


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
        #output = self.accumulator.identity()
        output = self.accumulator.copy()
        dataset = events.metadata["dataset"]
        # HLT
        hlt_avail = events.HLT.layout.keys()
        met_cut = events.PFMET.pt>200 
        events = events[met_cut]
        events_MET = self.HLT_MET(events, hlt_avail)
        events_LEP = self.HLT_LEP(events_MET, hlt_avail)
        dic_pt_MET = self.get_pTs_from_events(events_MET)
        dic_pt_LEP = self.get_pTs_from_events(events_LEP)
        dic_pt_data = self.get_pTs_from_events(events)
        dic_deltaR_MET = self.get_pTs_from_events(events_MET)
        dic_deltaR_LEP = self.get_pTs_from_events(events_LEP)

        # print("lead_MM_EE:", dic_pt_LEP['pt_lead_MM_EE'])
        # print("trail_MM_EE:", dic_pt_LEP['pt_trail_MM_EE'])
        # print("Num events:", len(dic_pt_LEP['pt_lead_MM_EE']))
        # print(dic_pt_LEP)
        # print(dic_pt_MET)

        # output['h_MM_EE_no_trig'].fill(dataset=dataset, lead=dic_pt_data['pt_lead_MM_EE'], trail=dic_pt_data['pt_trail_MM_EE'])
        # output['h_MM_BB_no_trig'].fill(dataset=dataset, lead=dic_pt_data['pt_lead_MM_BB'], trail=dic_pt_data['pt_trail_MM_BB'])
        # output['h_MM_BE_no_trig'].fill(dataset=dataset, lead=dic_pt_data['pt_lead_MM_BE'], trail=dic_pt_data['pt_trail_MM_BE'])
        # output['h_MM_EB_no_trig'].fill(dataset=dataset, lead=dic_pt_data['pt_lead_MM_EB'], trail=dic_pt_data['pt_trail_MM_EB'])
        # output['h_EE_EE_no_trig'].fill(dataset=dataset, lead=dic_pt_data['pt_lead_EE_EE'], trail=dic_pt_data['pt_trail_EE_EE'])
        # output['h_EE_BB_no_trig'].fill(dataset=dataset, lead=dic_pt_data['pt_lead_EE_BB'], trail=dic_pt_data['pt_trail_EE_BB'])

        output['h_num_MM'].fill(dataset=dataset, lead=dic_pt_LEP['pt_lead_MM'], trail=dic_pt_LEP['pt_trail_MM'])
        output['h_num_ME'].fill(dataset=dataset, lead=dic_pt_LEP['pt_lead_ME'], trail=dic_pt_LEP['pt_trail_ME'])
        output['h_num_EM'].fill(dataset=dataset, lead=dic_pt_LEP['pt_lead_EM'], trail=dic_pt_LEP['pt_trail_EM'])
        output['h_num_EE'].fill(dataset=dataset, lead=dic_pt_LEP['pt_lead_EE'], trail=dic_pt_LEP['pt_trail_EE'])
        output['h_den_MM'].fill(dataset=dataset, lead=dic_pt_MET['pt_lead_MM'], trail=dic_pt_MET['pt_trail_MM'])
        output['h_den_ME'].fill(dataset=dataset, lead=dic_pt_MET['pt_lead_ME'], trail=dic_pt_MET['pt_trail_ME'])
        output['h_den_EM'].fill(dataset=dataset, lead=dic_pt_MET['pt_lead_EM'], trail=dic_pt_MET['pt_trail_EM'])
        output['h_den_EE'].fill(dataset=dataset, lead=dic_pt_MET['pt_lead_EE'], trail=dic_pt_MET['pt_trail_EE'])


        # output['h_num_MM_deltaR'].fill(dataset=dataset, lead=dic_pt_LEP['pt_lead_MM'], trail=dic_pt_LEP['pt_trail_MM'], dR=dic_deltaR_LEP['deltaR_MM'])
        # output['h_num_ME_deltaR'].fill(dataset=dataset, lead=dic_pt_LEP['pt_lead_ME'], trail=dic_pt_LEP['pt_trail_ME'], dR=dic_deltaR_LEP['deltaR_ME'])
        # output['h_num_EM_deltaR'].fill(dataset=dataset, lead=dic_pt_LEP['pt_lead_EM'], trail=dic_pt_LEP['pt_trail_EM'], dR=dic_deltaR_LEP['deltaR_EM'])
        # output['h_num_EE_deltaR'].fill(dataset=dataset, lead=dic_pt_LEP['pt_lead_EE'], trail=dic_pt_LEP['pt_trail_EE'], dR=dic_deltaR_LEP['deltaR_EE'])
        # output['h_den_MM_deltaR'].fill(dataset=dataset, lead=dic_pt_MET['pt_lead_MM'], trail=dic_pt_MET['pt_trail_MM'], dR=dic_deltaR_MET['deltaR_MM'])
        # output['h_den_ME_deltaR'].fill(dataset=dataset, lead=dic_pt_MET['pt_lead_ME'], trail=dic_pt_MET['pt_trail_ME'], dR=dic_deltaR_MET['deltaR_ME'])
        # output['h_den_EM_deltaR'].fill(dataset=dataset, lead=dic_pt_MET['pt_lead_EM'], trail=dic_pt_MET['pt_trail_EM'], dR=dic_deltaR_MET['deltaR_EM'])
        # output['h_den_EE_deltaR'].fill(dataset=dataset, lead=dic_pt_MET['pt_lead_EE'], trail=dic_pt_MET['pt_trail_EE'], dR=dic_deltaR_MET['deltaR_EE'])

        # output['h_num_MM_EE'].fill(dataset=dataset, lead=dic_pt_LEP['pt_lead_MM_EE'], trail=dic_pt_LEP['pt_trail_MM_EE'])
        # output['h_num_ME_EE'].fill(dataset=dataset, lead=dic_pt_LEP['pt_lead_ME_EE'], trail=dic_pt_LEP['pt_trail_ME_EE'])
        # output['h_num_EM_EE'].fill(dataset=dataset, lead=dic_pt_LEP['pt_lead_EM_EE'], trail=dic_pt_LEP['pt_trail_EM_EE'])
        # output['h_num_EE_EE'].fill(dataset=dataset, lead=dic_pt_LEP['pt_lead_EE_EE'], trail=dic_pt_LEP['pt_trail_EE_EE'])

        # output['h_den_MM_EE'].fill(dataset=dataset, lead=dic_pt_MET['pt_lead_MM_EE'], trail=dic_pt_MET['pt_trail_MM_EE'])
        # output['h_den_ME_EE'].fill(dataset=dataset, lead=dic_pt_MET['pt_lead_ME_EE'], trail=dic_pt_MET['pt_trail_ME_EE'])
        # output['h_den_EM_EE'].fill(dataset=dataset, lead=dic_pt_MET['pt_lead_EM_EE'], trail=dic_pt_MET['pt_trail_EM_EE'])
        # output['h_den_EE_EE'].fill(dataset=dataset, lead=dic_pt_MET['pt_lead_EE_EE'], trail=dic_pt_MET['pt_trail_EE_EE'])

        # output['h_num_MM_EB'].fill(dataset=dataset, lead=dic_pt_LEP['pt_lead_MM_EB'], trail=dic_pt_LEP['pt_trail_MM_EB'])
        # output['h_num_ME_EB'].fill(dataset=dataset, lead=dic_pt_LEP['pt_lead_ME_EB'], trail=dic_pt_LEP['pt_trail_ME_EB'])
        # output['h_num_EM_EB'].fill(dataset=dataset, lead=dic_pt_LEP['pt_lead_EM_EB'], trail=dic_pt_LEP['pt_trail_EM_EB'])
        # output['h_num_EE_EB'].fill(dataset=dataset, lead=dic_pt_LEP['pt_lead_EE_EB'], trail=dic_pt_LEP['pt_trail_EE_EB'])

        # output['h_den_MM_EB'].fill(dataset=dataset, lead=dic_pt_MET['pt_lead_MM_EB'], trail=dic_pt_MET['pt_trail_MM_EB'])
        # output['h_den_ME_EB'].fill(dataset=dataset, lead=dic_pt_MET['pt_lead_ME_EB'], trail=dic_pt_MET['pt_trail_ME_EB'])
        # output['h_den_EM_EB'].fill(dataset=dataset, lead=dic_pt_MET['pt_lead_EM_EB'], trail=dic_pt_MET['pt_trail_EM_EB'])
        # output['h_den_EE_EB'].fill(dataset=dataset, lead=dic_pt_MET['pt_lead_EE_EB'], trail=dic_pt_MET['pt_trail_EE_EB'])

        # output['h_num_MM_BE'].fill(dataset=dataset, lead=dic_pt_LEP['pt_lead_MM_BE'], trail=dic_pt_LEP['pt_trail_MM_BE'])
        # output['h_num_ME_BE'].fill(dataset=dataset, lead=dic_pt_LEP['pt_lead_ME_BE'], trail=dic_pt_LEP['pt_trail_ME_BE'])
        # output['h_num_EM_BE'].fill(dataset=dataset, lead=dic_pt_LEP['pt_lead_EM_BE'], trail=dic_pt_LEP['pt_trail_EM_BE'])
        # output['h_num_EE_BE'].fill(dataset=dataset, lead=dic_pt_LEP['pt_lead_EE_BE'], trail=dic_pt_LEP['pt_trail_EE_BE'])

        # output['h_den_MM_BE'].fill(dataset=dataset, lead=dic_pt_MET['pt_lead_MM_BE'], trail=dic_pt_MET['pt_trail_MM_BE'])
        # output['h_den_ME_BE'].fill(dataset=dataset, lead=dic_pt_MET['pt_lead_ME_BE'], trail=dic_pt_MET['pt_trail_ME_BE'])
        # output['h_den_EM_BE'].fill(dataset=dataset, lead=dic_pt_MET['pt_lead_EM_BE'], trail=dic_pt_MET['pt_trail_EM_BE'])
        # output['h_den_EE_BE'].fill(dataset=dataset, lead=dic_pt_MET['pt_lead_EE_BE'], trail=dic_pt_MET['pt_trail_EE_BE'])

        # output['h_num_MM_BB'].fill(dataset=dataset, lead=dic_pt_LEP['pt_lead_MM_BB'], trail=dic_pt_LEP['pt_trail_MM_BB'])
        # output['h_num_ME_BB'].fill(dataset=dataset, lead=dic_pt_LEP['pt_lead_ME_BB'], trail=dic_pt_LEP['pt_trail_ME_BB'])
        # output['h_num_EM_BB'].fill(dataset=dataset, lead=dic_pt_LEP['pt_lead_EM_BB'], trail=dic_pt_LEP['pt_trail_EM_BB'])
        # output['h_num_EE_BB'].fill(dataset=dataset, lead=dic_pt_LEP['pt_lead_EE_BB'], trail=dic_pt_LEP['pt_trail_EE_BB'])

        # output['h_den_MM_BB'].fill(dataset=dataset, lead=dic_pt_MET['pt_lead_MM_BB'], trail=dic_pt_MET['pt_trail_MM_BB'])
        # output['h_den_ME_BB'].fill(dataset=dataset, lead=dic_pt_MET['pt_lead_ME_BB'], trail=dic_pt_MET['pt_trail_ME_BB'])
        # output['h_den_EM_BB'].fill(dataset=dataset, lead=dic_pt_MET['pt_lead_EM_BB'], trail=dic_pt_MET['pt_trail_EM_BB'])
        # output['h_den_EE_BB'].fill(dataset=dataset, lead=dic_pt_MET['pt_lead_EE_BB'], trail=dic_pt_MET['pt_trail_EE_BB'])


        # h = output['h_num_MM_EE']
        # print("Lead bin edges:", h.axes['lead'].edges)
        # print("Trail bin edges:", h.axes['trail'].edges)
        # print("Sum of weights:", h.sum(flow=True).value)

        
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
        muons = muons[muons.pt > 20.0]
        muons = muons[muons.tightId] 
        muons = muons[muons.pfRelIso04_all <= 0.15]
        muons = muons[abs(muons.dxy) < 0.045]  
        muons = muons[abs(muons.dz) < 0.2]
        #print("muon")
        return muons

    def get_good_electrons(self, electrons):
        electron_superclusterEta = electrons.eta + electrons.deltaEtaSC
        electrons = electrons[(np.abs(electron_superclusterEta) < 1.4442) | ((np.abs(electron_superclusterEta) > 1.566) & (np.abs(electron_superclusterEta)  < 2.5))]
        # skip 25 GeV
        electrons = electrons[electrons.pt > 20.0]
        electrons = electrons[electrons.mvaIso_WP90] 
        #print("electron")
        return electrons

    def get_pTs_from_events(self, events):
        # dictionary to be returned
        dic_pt_dR = {}
        

        # get leptons
        good_Ms = self.get_good_muons(events.Muon)
        good_Es = self.get_good_electrons(events.Electron)

        # at least 2 leptons
        mask_ll = ak.num(good_Ms) + ak.num(good_Es) >= 2
        good_Ls = ak.concatenate([good_Ms[mask_ll], good_Es[mask_ll]], axis=1)
        good_Ls = ak.with_name(good_Ls, "PtEtaPhiMLorentzVector")
        id_sort = ak.argsort(good_Ls.pt, ascending=False)
        good_Ls = good_Ls[id_sort]
        good_Ls = good_Ls[:,:2]
        #print("inside get pt")

        # print("Good muons:", ak.num(good_Ms))
        # print("Good electrons:", ak.num(good_Es))
        # print("After ≥2 leptons:", ak.sum(mask_ll))
        # print("Good_Ls length:", ak.num(good_Ls))


        # use the -ve sign to get osof


        # pdgID
        id_prod = (good_Ls.pdgId[:,0] * good_Ls.pdgId[:,1])

        # ElEl
        mask_ElEl = id_prod==-121  #OSOF
        good_ElEl = good_Ls[mask_ElEl]

        mask_0Br = good_ElEl.eta[:,0] < 1.5
        mask_1Br = good_ElEl.eta[:,1] < 1.5
        mask_0Ec = good_ElEl.eta[:,0] >= 1.5
        mask_1Ec = good_ElEl.eta[:,1] >= 1.5

        mask_BrBr = mask_0Br & mask_1Br
        mask_BrEc = mask_0Br & mask_1Ec
        mask_EcBr = mask_0Ec & mask_1Br
        mask_EcEc = mask_0Ec & mask_1Ec

        dic_pt_dR['pt_lead_EE'] = good_ElEl.pt[:,0]
        dic_pt_dR['pt_trail_EE'] = good_ElEl.pt[:,1]
        dic_pt_dR['deltaR_EE'] = good_ElEl[:,0].delta_r(good_ElEl[:,1])
        # dic_pt['pt_lead_EE_BB'] = good_ElEl[mask_BrBr].pt[:,0]
        # dic_pt['pt_trail_EE_BB'] = good_ElEl[mask_BrBr].pt[:,1]
        # dic_pt['pt_lead_EE_BE'] = good_ElEl[mask_BrEc].pt[:,0]
        # dic_pt['pt_trail_EE_BE'] = good_ElEl[mask_BrEc].pt[:,1]
        # dic_pt['pt_lead_EE_EB'] = good_ElEl[mask_EcBr].pt[:,0]
        # dic_pt['pt_trail_EE_EB'] = good_ElEl[mask_EcBr].pt[:,1]
        # dic_pt['pt_lead_EE_EE'] = good_ElEl[mask_EcEc].pt[:,0]
        # dic_pt['pt_trail_EE_EE'] = good_ElEl[mask_EcEc].pt[:,1]

        # MuMu
        mask_MuMu = id_prod == -169
        good_MuMu_1 = good_Ls[mask_MuMu]

        # # if no pairs, create empty arrays gracefully
        # if len(good_MuMu_1) == 0:
        #     good_MuMu = good_MuMu_1  # remains empty
        # else:
        #     # compute deltaEta and deltaPhi robustly (wrap dphi to [-pi,pi])
        #     deta = good_MuMu_1[:,0].eta - good_MuMu_1[:,1].eta
        #     dphi = (good_MuMu_1[:,0].phi - good_MuMu_1[:,1].phi + np.pi) % (2*np.pi) - np.pi

        #     deltaR = np.sqrt(deta**2 + dphi**2)
        #     good_MuMu = good_MuMu_1[deltaR >= 0.4]

        good_MuMu = good_MuMu_1



        mask_0Br = good_MuMu.eta[:,0] < 1.5
        mask_1Br = good_MuMu.eta[:,1] < 1.5
        mask_0Ec = good_MuMu.eta[:,0] >= 1.5
        mask_1Ec = good_MuMu.eta[:,1] >= 1.5

        mask_BrBr = mask_0Br & mask_1Br
        mask_BrEc = mask_0Br & mask_1Ec
        mask_EcBr = mask_0Ec & mask_1Br
        mask_EcEc = mask_0Ec & mask_1Ec

        dic_pt_dR['pt_lead_MM'] = good_MuMu.pt[:,0]
        dic_pt_dR['pt_trail_MM'] = good_MuMu.pt[:,1]
        dic_pt_dR['deltaR_MM'] = good_MuMu[:,0].delta_r(good_MuMu[:,1])
        # dic_pt['pt_lead_MM_BB'] = good_MuMu[mask_BrBr].pt[:,0]
        # dic_pt['pt_trail_MM_BB'] = good_MuMu[mask_BrBr].pt[:,1]
        # dic_pt['pt_lead_MM_BE'] = good_MuMu[mask_BrEc].pt[:,0]
        # dic_pt['pt_trail_MM_BE'] = good_MuMu[mask_BrEc].pt[:,1]
        # dic_pt['pt_lead_MM_EB'] = good_MuMu[mask_EcBr].pt[:,0]
        # dic_pt['pt_trail_MM_EB'] = good_MuMu[mask_EcBr].pt[:,1]
        # dic_pt['pt_lead_MM_EE'] = good_MuMu[mask_EcEc].pt[:,0]
        # dic_pt['pt_trail_MM_EE'] = good_MuMu[mask_EcEc].pt[:,1]



        # Mix
        mask_mix = id_prod==-143 #OSOF
        good_mix = good_Ls[mask_mix]

        # ElMu
        mask_ElMu = abs(good_mix.pdgId[:,0])==11
        good_ElMu = good_mix[mask_ElMu]

        mask_0Br = good_ElMu.eta[:,0] < 1.5
        mask_1Br = good_ElMu.eta[:,1] < 1.5
        mask_0Ec = good_ElMu.eta[:,0] >= 1.5
        mask_1Ec = good_ElMu.eta[:,1] >= 1.5

        mask_BrBr = mask_0Br & mask_1Br
        mask_BrEc = mask_0Br & mask_1Ec
        mask_EcBr = mask_0Ec & mask_1Br
        mask_EcEc = mask_0Ec & mask_1Ec

        dic_pt_dR['pt_lead_EM'] = good_ElMu.pt[:,0]
        dic_pt_dR['pt_trail_EM'] = good_ElMu.pt[:,1]
        dic_pt_dR['deltaR_EM'] = good_ElMu[:,0].delta_r(good_ElMu[:,1])
        # dic_pt['pt_lead_EM_BB'] = good_ElMu[mask_BrBr].pt[:,0]
        # dic_pt['pt_trail_EM_BB'] = good_ElMu[mask_BrBr].pt[:,1]
        # dic_pt['pt_lead_EM_BE'] = good_ElMu[mask_BrEc].pt[:,0]
        # dic_pt['pt_trail_EM_BE'] = good_ElMu[mask_BrEc].pt[:,1]
        # dic_pt['pt_lead_EM_EB'] = good_ElMu[mask_EcBr].pt[:,0]
        # dic_pt['pt_trail_EM_EB'] = good_ElMu[mask_EcBr].pt[:,1]
        # dic_pt['pt_lead_EM_EE'] = good_ElMu[mask_EcEc].pt[:,0]
        # dic_pt['pt_trail_EM_EE'] = good_ElMu[mask_EcEc].pt[:,1]

        # MuEl
        mask_MuEl = abs(good_mix.pdgId[:,0])==13
        good_MuEl = good_mix[mask_MuEl]

        mask_0Br = good_MuEl.eta[:,0] < 1.5
        mask_1Br = good_MuEl.eta[:,1] < 1.5
        mask_0Ec = good_MuEl.eta[:,0] >= 1.5
        mask_1Ec = good_MuEl.eta[:,1] >= 1.5

        mask_BrBr = mask_0Br & mask_1Br
        mask_BrEc = mask_0Br & mask_1Ec
        mask_EcBr = mask_0Ec & mask_1Br
        mask_EcEc = mask_0Ec & mask_1Ec

        dic_pt_dR['pt_lead_ME'] = good_MuEl.pt[:,0]
        dic_pt_dR['pt_trail_ME'] = good_MuEl.pt[:,1]
        dic_pt_dR['deltaR_ME'] = good_MuEl[:,0].delta_r(good_MuEl[:,1])
        # dic_pt['pt_lead_ME_BB'] = good_MuEl[mask_BrBr].pt[:,0]
        # dic_pt['pt_trail_ME_BB'] = good_MuEl[mask_BrBr].pt[:,1]
        # dic_pt['pt_lead_ME_BE'] = good_MuEl[mask_BrEc].pt[:,0]
        # dic_pt['pt_trail_ME_BE'] = good_MuEl[mask_BrEc].pt[:,1]
        # dic_pt['pt_lead_ME_EB'] = good_MuEl[mask_EcBr].pt[:,0]
        # dic_pt['pt_trail_ME_EB'] = good_MuEl[mask_EcBr].pt[:,1]
        # dic_pt['pt_lead_ME_EE'] = good_MuEl[mask_EcEc].pt[:,0]
        # dic_pt['pt_trail_ME_EE'] = good_MuEl[mask_EcEc].pt[:,1]

        return dic_pt_dR
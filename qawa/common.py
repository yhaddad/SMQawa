from coffea.lookup_tools import dense_lookup
from coffea.analysis_tools import Weights
from coffea import processor
from scipy import interpolate

import awkward as ak
import numpy as np
import uproot
import os


def met_phi_xy_correction(met, run, npv, is_mc:bool=False, era:str='2016'):
    xcor = ak.ones_like(run)
    ycor = ak.ones_like(run)
    
    # UL2016
    if is_mc:
        if '2016' in era:
            if('APV' in era):  
                xcor = -(-0.188743*npv +0.136539)
                ycor = -(0.0127927*npv +0.117747);
            else:
                xcor = -(-0.15349700*npv -0.231751)
                ycor = -( 0.00731978*npv +0.243323)
        if '2018' in era:
            xcor = -(0.183518*npv +0.546754)
            ycor = -(0.192263*npv -0.421210)
        if '2017'in era:
            xcor = -(-0.300155*npv +1.90608)
            ycor = -( 0.300213*npv -2.02232)
            
    else:
        # UL2016
        xcor = ak.where((run >= 272007) & (run <= 275376), -(-0.0214894*npv -0.188255), xcor) # UL2016B
        xcor = ak.where((run >= 272007) & (run <= 275376), -( 0.0876624*npv +0.812885), ycor) # UL2016B
        xcor = ak.where((run >= 275657) & (run <= 276283), -(-0.0322090*npv +0.067288), xcor) # UL2016C
        xcor = ak.where((run >= 275657) & (run <= 276283), -( 0.1139170*npv +0.743906), ycor) # UL2016C
        xcor = ak.where((run >= 276315) & (run <= 276811), -(-0.0293663*npv +0.211060), xcor) # UL2016D
        xcor = ak.where((run >= 276315) & (run <= 276811), -( 0.1133100*npv +0.815787), ycor) # UL2016D
        xcor = ak.where((run >= 276831) & (run <= 277420), -(-0.0132046*npv +0.200730), xcor) # UL2016E
        xcor = ak.where((run >= 276831) & (run <= 277420), -( 0.1348090*npv +0.679068), ycor) # UL2016E
        xcor = ak.where(
            (run >= 277772) & (run <= 278768) & (run==278770), 
            -(-0.0543566*npv +0.816597), xcor) # UL2016F
        xcor = ak.where(
            (run >= 277772) & (run <= 278768) & (run==278770),
            -( 0.1142250*npv +1.172660), ycor) # UL2016F
        xcor = ak.where(
            (run >= 278801) & (run <= 278808) & (run==278769),
            -( 0.1346160*npv -0.899650), xcor) # UL2016Flate
        xcor = ak.where(
            (run >= 278801) & (run <= 278808) & (run==278769),
            -( 0.0397736*npv +1.038500), ycor) # UL2016Flate
        xcor = ak.where((run >= 278820) & (run <= 280385), -( 0.1218090*npv -0.584893), xcor) #UL2016G
        xcor = ak.where((run >= 278820) & (run <= 280385), -( 0.0558974*npv +0.891234), ycor) #UL2016G
        xcor = ak.where((run >= 280919) & (run <= 284044), -( 0.0868828*npv -0.703489), xcor) #UL2016H
        xcor = ak.where((run >= 280919) & (run <= 284044), -( 0.0888774*npv +0.902632), ycor) #UL2016H

        # UL2017
        xcor = ak.where((run >= 297020) & (run <= 299329), -(-0.211161*npv +0.419333), xcor)
        xcor = ak.where((run >= 297020) & (run <= 299329), -( 0.251789*npv -1.280890), ycor)
        xcor = ak.where((run >= 299337) & (run <= 302029), -(-0.185184*npv -0.164009), xcor)
        xcor = ak.where((run >= 299337) & (run <= 302029), -( 0.200941*npv -0.568530), ycor)
        xcor = ak.where((run >= 302030) & (run <= 303434), -(-0.201606*npv +0.426502), xcor)
        xcor = ak.where((run >= 302030) & (run <= 303434), -( 0.188208*npv -0.583130), ycor)
        xcor = ak.where((run >= 303435) & (run <= 316995), -(-0.162472*npv +0.176329), xcor)
        xcor = ak.where((run >= 303435) & (run <= 316995), -( 0.138076*npv -0.250239), ycor)
        xcor = ak.where((run >= 304911) & (run <= 316995), -(-0.210639*npv +0.729340), xcor)
        xcor = ak.where((run >= 304911) & (run <= 316995), -( 0.198626*npv +1.028000), ycor)

        # UL2018
        xcor = ak.where((run >= 315252) & (run <= 316995), -(0.3628650*npv -1.945050), xcor)
        ycor = ak.where((run >= 315252) & (run <= 316995), -(0.0709085*npv -0.307365), ycor)
        xcor = ak.where((run >= 316998) & (run <= 319312), -(0.4920830*npv -2.935520), xcor)
        ycor = ak.where((run >= 316998) & (run <= 319312), -(0.1787400*npv -0.786844), ycor)
        xcor = ak.where((run >= 319313) & (run <= 320393), -(0.5213490*npv -1.445440), xcor)
        ycor = ak.where((run >= 319313) & (run <= 320393), -(0.1189560*npv -1.964340), ycor)
        xcor = ak.where((run >= 320394) & (run <= 325273), -(0.5311510*npv -1.375680), xcor)
        ycor = ak.where((run >= 320394) & (run <= 325273), -(0.0884639*npv -1.570890), ycor)
        
    metx_ = met.pt * np.cos(met.phi)+xcor
    mety_ = met.pt * np.sin(met.phi)+xcor

    pt_ = np.sqrt((metx_**2)+(mety_**2))
    phi_ = np.arctan2(mety_,metx_)
    
    met['pt'] = pt_
    met['phi'] = phi_
    return met

def trigger_rules(event, rules:dict, era:str='2018'):
    ds_name = {
        '2016' : ['DoubleMuon', 'SingleMuon', 'EGamma', 'SingleElectron', 'MuonEG', 'Photon'],
        '2017' : ['DoubleMuon', 'SingleMuon', 'EGamma', 'SingleElectron', 'MuonEG','Photon'],
        '2018' : ['DoubleMuon', 'SingleMuon', 'EGamma', 'MuonEG','Photon']
    }
    
    _pass = np.zeros(len(event), dtype='bool')
    _veto = np.zeros(len(event), dtype='bool')
    
    _ds = event.metadata['dataset'].split('_')[0]
    
    #just a hack for using photon trigger, need to make it work properly
    _ds="Photon"

    for t in rules[f'Run{era}.{_ds}']['pass']:
        print("pass",t)
        if t in event.HLT.fields:
            _pass = _pass | event.HLT[t]
        else:
            print(f'{t} not present in the data')
            
    for t in rules[f'Run{era}.{_ds}']['veto']:
        print("_veto",t)
        if t in event.HLT.fields:
            _veto =  _veto | event.HLT[t]
            
    # passing triggers and vetoing triggers from other datasets
    print(_pass)
    return _pass & ~_veto


def theory_pdf_weight(weights, pdf_weight):
    _nm = np.ones(len(weights.weight()))
    _up = np.ones(len(weights.weight()))
    _dw = np.ones(len(weights.weight()))

    if pdf_weight is not None and "306000 - 306102" in pdf_weight.__doc__:
        arg = pdf_weight[:, 1:-2] - np.ones((len(weights.weight()), 100))
        summed  = ak.sum(np.square(arg), axis=1)
        pdf_unc = np.sqrt((1. / 99.) * summed)
        weights.add('PDF_weight', _nm, pdf_unc + _nm)

        # alpha_S weights
        # Eq. 27 of same ref
        as_unc = 0.5 * (pdf_weight[:, 102] - pdf_weight[:, 101])
        weights.add('aS_weight', _nm, as_unc + _nm)

        # PDF + alpha_S weights
        # Eq. 28 of same ref
        pdfas_unc = np.sqrt(np.square(pdf_unc) + np.square(as_unc))
        weights.add('PDFaS_weight', _nm, pdfas_unc + _nm)
    else:
        weights.add('aS_weight'   , _nm, _up, _dw)
        weights.add('PDF_weight'  , _nm, _up, _dw)
        weights.add('PDFaS_weight', _nm, _up, _dw)
        

def theory_ps_weight(weights, ps_weight):
    nominal= np.ones(len(weights.weight()))
    up_isr = np.ones(len(weights.weight()))
    up_fsr = np.ones(len(weights.weight()))
    dw_isr = np.ones(len(weights.weight()))
    dw_fsr = np.ones(len(weights.weight()))

    if ps_weight is not None:
        if len(ps_weight[0]) == 4:
            up_isr   = ps_weight[:, 0]
            down_isr = ps_weight[:, 2]
            up_fsr   = ps_weight[:, 1]
            down_fsr = ps_weight[:, 3]

    weights.add('UEPS_ISR', nominal, up_isr, dw_isr)
    weights.add('UEPS_FSR', nominal, up_fsr, dw_fsr)
    


class pileup_weights:
    def __init__(self, do_syst:bool=True, era:str='2018'):
        _data_path = os.path.join(os.path.dirname(__file__), 'data/PU/')
        files_  = {
            "Nom" : f"{_data_path}/PileupHistogram-goldenJSON-13tev-{era}-69200ub-99bins.root",
            "Up"  : f"{_data_path}/PileupHistogram-goldenJSON-13tev-{era}-72400ub-99bins.root",
            "Down": f"{_data_path}/PileupHistogram-goldenJSON-13tev-{era}-66000ub-99bins.root"
        }

        hist_norm = lambda x: np.divide(x, x.sum())

        self.simu_pu = None 
        with uproot.open(f'{_data_path}/mcPileupUL{era}.root') as ifile:
            self.simu_pu = ifile['pu_mc'].values()
            self.simu_pu = np.insert(self.simu_pu,0,0.)
            self.simu_pu = self.simu_pu[:-2]

        mask = self.simu_pu > 0

        self.corrections = {}
        for var,pfile in files_.items():
            with uproot.open(pfile) as ifile:
                data_pu = hist_norm(ifile["pileup"].values())
                edges = ifile["pileup"].axis().edges()

                corr = np.divide(data_pu, self.simu_pu, where=mask)
                pileup_corr = dense_lookup.dense_lookup(corr, edges)
                self.corrections['puWeight' if 'Nom' in var else f'puWeight{var}'] = pileup_corr
    
    
    def append_pileup_weight(self, weights, pu):
        weights.add(
            'pileup_weight',
            self.corrections['puWeight'    ](pu),
            self.corrections['puWeightUp'  ](pu),
            self.corrections['puWeightDown'](pu),
        )
        return weights
    
class LinearNDInterpolatorExt(object):
    """ 
    LinearNDInterpolator with possibility to extralopate
    outside the range by using Nearest neighbour.
    """
    def __init__(self, points, values):
        self.funcinterp = interpolate.LinearNDInterpolator (points, values)
        self.funcneares = interpolate.NearestNDInterpolator(points, values)
    
    def __call__(self, *args):
        z = self.funcinterp(*args)
        h = self.funcneares(*args)
        chk = np.isnan(z)
        if chk.any():
            return np.where(chk, h, z)
        else:
            return z
    
class ewk_corrector:
    def __init__(self, process:str='ZZ', do_syst:bool=True, beam_energy:float = 6500, use_inv_masses:bool=False):
        _data_path = os.path.join(os.path.dirname(__file__), 'data/')
        self.corr_file = f"{_data_path}/ewk/data_{process}_EwkCorrections.dat"
        self.corr_data = np.loadtxt(self.corr_file)
        self.energy = beam_energy
        self.process = process
        self.use_inv_masses = use_inv_masses
        self.m_z = 91.1876
        self.m_w = 80.385 if process=='WZ' else self.m_z
        
        self.m_z2 = self.m_z**2
        self.m_w2 = self.m_w**2
        
        self.exterp = []
        for iv in range(3):
            self.exterp.append(
               LinearNDInterpolatorExt(
                   list(zip(self.corr_data[:,0], self.corr_data[:,1])), 
                   self.corr_data[:,2 + iv]
                )
            )
            
        self.corrNNLO = np.array([
            1.513834489150, #  0
            1.541738780180, #  1
            1.497829632510, #  2
            1.534956782920, #  3
            1.478217033060, #  4
            1.504330859290, #  5
            1.520626246850, #  6
            1.507013090030, #  7
            1.494243156250, #  8
            1.450536096150, #  9
            1.460812521660, # 10
            1.471603622200, # 11
            1.467700038200, # 12
            1.422408690640, # 13
            1.397184022730, # 14
            1.375593447520, # 15
            1.391901318370, # 16
            1.368564350560, # 17
            1.317884804290, # 18
            1.314019950800, # 19
            1.274641749910, # 20
            1.242346606820, # 21
            1.244727403840, # 22
            1.146259351670, # 23
            1.107804993520, # 24
            1.042053646740, # 25
            0.973608545141, # 26
            0.872169942668, # 27
            0.734505279177, # 28
            1.163152837230, # 29
            1.163152837230, # 30
            1.163152837230  # 31
        ])
        
        
    def get_weight(self, gen_coll, x1, x2, weights=None):
        id_q1 = np.abs(gen_coll.pdgId[:,0])
        id_q2 = np.abs(gen_coll.pdgId[:,1])
        
        # make sure we have quark in the initial state
        init_q_mask = (
            (((id_q1>=1) & (id_q1<=6)) | (id_q1==21)) &
            (((id_q2>=1) & (id_q2<=6)) | (id_q2==21))
        )
        # make sure we have vector bosons in the events   
        vect_v_mask = (
            ((np.abs(gen_coll.pdgId[:,2])==23) | (np.abs(gen_coll.pdgId[:,2]) == 24)) &
            ((np.abs(gen_coll.pdgId[:,3])==23) | (np.abs(gen_coll.pdgId[:,3]) == 24))
        )
        # apply corrections only on same flavour and non-gluon initiate processes        
        q1 = ak.zip({
                "x":  ak.zeros_like(x1), 
                "y":  ak.zeros_like(x1),
                "z":  x1 * self.energy,
                "t":  x1 * self.energy
            },
            with_name="Candidate",
            behavior=candidate.behavior
        )

        q2 = ak.zip(
            {
                "x":  ak.zeros_like(x2), 
                "y":  ak.zeros_like(x2),
                "z": -x2 * self.energy,
                "t":  x2 * self.energy
            },
            with_name="Candidate",
            behavior=candidate.behavior,
        )

        v1 = gen_coll[:,2]
        v2 = gen_coll[:,3]
        
        # getting the s hat
        vv = v1 + v2
        shat = vv.mass2
        
        
        # getting the t hat
        # Boost to the VV center of mass frame
        b_q1 = q1.boost( -vv.boostvec )
        b_q2 = q2.boost( -vv.boostvec )
        b_v1 = v1.boost( -vv.boostvec )

        # Unitary vectors
        uq1 = (b_q1.pvec / b_q1.pvec.p )
        uq2 = (b_q2.pvec / b_q2.pvec.p )
        uv1 = (b_v1.pvec / b_v1.pvec.p )

        # effectiuve beam axis
        diff_q = uq1 - uq2
        beam_eff_axis = diff_q/diff_q.p

        # cos theta for the effective beam axis
        costheta = beam_eff_axis.dot(uv1)
        that = 0.0 
        if self.use_inv_masses:
            b = 1./2./np.sqrt(shat) * np.sqrt(
                np.power(shat-v1.mass2-v2.mass2,2) - 4*v1.mass2*v2.mass2
            )
            a = np.sqrt(b*b + v1.mass2)
            that = v1.mass2 - np.sqrt(shat) * (a - b * costheta)
        else:
            b = 1./2./np.sqrt(shat) * np.sqrt(
                np.abs(
                    np.power(shat-self.m_z2-self.m_w2,2) - 4*self.m_w2*self.m_z2
                )
            );
            a = np.sqrt(b*b + self.m_z2);
            that = self.m_z2 - np.sqrt(shat) * (a - b * costheta)
    
    
        onshell_mask = np.sqrt(shat) < self.m_w + self.m_z
        quark_type = np.zeros_like(shat)
        
        quark_type = np.minimum(id_q1, id_q2)

        shat = ak.nan_to_num(shat, 1.0)
        that = ak.nan_to_num(that, 1.0)
        
        corr_0 = 1 + self.exterp[0](np.sqrt(shat.to_numpy()), that.to_numpy())
        corr_1 = 1 + self.exterp[0](np.sqrt(shat.to_numpy()), that.to_numpy())
        corr_2 = 1 + self.exterp[0](np.sqrt(shat.to_numpy()), that.to_numpy())
        
        # make sure the corrections are only on the on-shell
        corr_mask = (
            init_q_mask & 
            vect_v_mask &
            onshell_mask
        )
        
        corr_0 = np.where(corr_mask, np.ones_like(corr_0), corr_0)
        corr_1 = np.where(corr_mask, np.ones_like(corr_0), corr_1)
        corr_2 = np.where(corr_mask, np.ones_like(corr_0), corr_2)
    
        # correction by flavour
        weight = np.where(
            (quark_type == 1) | (quark_type == 3), 
            corr_1,
            np.where(
                (quark_type == 2) | (quark_type == 4), 
                corr_0,
                np.where(
                    (quark_type == 5), 
                    corr_2,
                    1.0
                )
            )
        )
        
        # NNLO weight
        knnlo = ak.ones_like(corr_0)
        if self.process == 'ZZ':
            delta_phi_vv = v1.delta_phi(v2)
            nnlo_ibin = np.round(delta_phi_vv.to_numpy() * 10.).astype(np.int32)
            knnlo =  np.where(
                init_q_mask & vect_v_mask, 
                1.0 + self.corrNNLO[nnlo_ibin], 
                1.0
            )
            
        # Average QCD NLO k factors from arXiv:1105.0020
        leptons = gen_coll[ 
            ((gen_coll.statusFlags & 128)!=0) & 
            (np.abs(gen_coll.pdgId) >= 11) & 
            (np.abs(gen_coll.pdgId) <= 16)
        ]
        rhovv = leptons.sum().pt/ak.sum(leptons.pt, axis=1)
        
        qcd_kfactor = ak.zeros_like(rhovv)
        if self.process == 'ZZ':
            qcd_kfactor = 15.99/ 9.89 - ak.ones_like(rhovv)# ZZ from arXiv1105.0020
        else:
            qcd_kfactor = np.where(
                (gen_coll.pdgId[:,2]*gen_coll.pdgId[:,3]) > 0, 
                28.55 / 15.51 - ak.ones_like(rhovv), # W+Z
                18.19 /  9.53 - ak.ones_like(rhovv), # W-Z
            )

        ewk_uncert = np.where(
            rhovv < 0.3, 1 + np.abs((weight-1.0)*(qcd_kfactor-1.0)), np.abs(weight)
        )
        
        # WZ: gamma-induced contribution
        if self.process=='WZ':
            gamma_ind_corr = np.where(
                ((gen_coll.pdgId[:,2]*gen_coll.pdgId[:,3]) > 0), 
                (1 + 0.00559445 - 5.17082e-6 * np.sqrt(shat) + 3.63331e-8 * shat), 
                (1 + 0.00174737 + 1.70668e-5 * np.sqrt(shat) + 2.26398e-8 * shat) 
            )
            weight = weight * np.where(corr_mask, gamma_ind_corr, 1.0)
            
        # filling the weights
        if weights is not None:
            weights.add('kEW', weight, weight*ewk_uncert, weight/ewk_uncert)
            weights.add('kNNLO', knnlo)
            




        
        
#phhoton SF
#harcoded for now
era='2018'

from coffea.lookup_tools import extractor
ext = extractor()
if '2016' in era:
    if('APV' in era):
        sff="egammaEffi_EGM2D_Pho_Tight_UL16.root"
    else :
        sff ="egammaEffi_EGM2D_Pho_Tight_UL16_postVFP.root"

if '2017' in era :
     sff ="egammaEffi_EGM2D_Pho_Tight_UL17.root"
     
if '2018' in era :
    sff ="egammaEffi_EGM2D_Pho_Tight_UL18.root"



ext.add_weight_sets(["EGamma_SF2D_T EGamma_SF2D qawa/data/Photon/"+sff,
                     "EGamma_SF2D_T_err EGamma_SF2D_error qawa/data/Photon/"+sff])

ext.finalize()
evaluator = ext.make_evaluator()
for key in evaluator.keys():
    print("\t", key)
print("testSF2d:", evaluator['EGamma_SF2D_T'])

from correctionlib import _core
def getPhotonTrigPrescale(run, lb, photrigdict, photonpt, era:str='2016'):
    photrignames = photrigdict['photon_triggers']['triggers']
    prescales=photrigdict['photon_triggers']['prescale']

    trigname="NULL"
    selectedtrig=-1
    for i in range(len(photrignames)):  
        if photonpt >= photrignames[i]["threshold"] :selectedtrig=i 
        if selectedtrig ==-1: trigname = "NULL"
        if selectedtrig >=0: trigname=photrignames[selectedtrig]["name"]


    evaluator = _core.CorrectionSet.from_file('data/Photon/'+prescales)     
    k=evaluator["HLT_prescale"].evaluate(era,trigname,run,float(lb))                                                                                                     
    return k



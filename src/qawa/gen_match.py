import os
import math
import awkward
import numpy as np
def delta_r2(p4_1, p4_2):
    """Calculate the squared delta R between two four-momentum vectors."""
    dEta = p4_1.eta - p4_2.eta
    dPhi = p4_1.phi - p4_2.phi
    if dPhi > math.pi :
        dPhi = 2*math.pi - dPhi
    return dEta ** 2 + dPhi ** 2

# Main function
def find_best_match(jets, gen_jets):
    gen = []
    for i, jet in enumerate(jets):
        gen_sub = []
        for j, Jet in enumerate(jet):
            match = None
            cur_min_dr2 = 0.04  # Use half of jet radius
            for GenJet in gen_jets[i]:
                dR2 = delta_r2(Jet, GenJet)
                if dR2 < cur_min_dr2:
                    match = GenJet
                    cur_min_dr2 = dR2
            try:
                gen_sub.append(match.pt)
            except:
                gen_sub.append(None)
        gen.append(gen_sub)
    gen = awkward.Array(gen)
    return gen
        
                    
    





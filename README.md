Fix the /coffea/nanoevents/methods/nanoaod.py, add these code for CorrT1METJet and GenJet:
@awkward.mixin_class(behavior)
class CorrT1METJet(vector.PtEtaPhiMLorentzVector, base.NanoCollection, base.Systematic):
    """NanoAOD narrow radius jet object"""
    @property
    def pt(self):
        return self["rawPt"]
    
    @property
    def rawFactor(self):
        self["rawFactor"] = awkward.zeros_like(self["rawPt"])
        return self["rawFactor"]
    
    @property
    def mass(self):
        self["mass"] = awkward.zeros_like(self["rawPt"])
        return self["mass"]
    
    @property
    def chEmEF(self):
        self["chEmEF"] = awkward.zeros_like(self["rawPt"])
        return self["chEmEF"]
    
    @property
    def neEmEF(self):
        self["neEmEF"] = awkward.zeros_like(self["rawPt"])
        return self["neEmEF"]
    

_set_repr_name("CorrT1METJet")




Fix the /coffea/nanoevents/schemas/nanoaod.py, add this key in 'mixins' dictionary:

"CorrT1METJet":"CorrT1METJet",

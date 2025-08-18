import correctionlib
import os
import awkward as ak
from pathlib import Path

class DataDrivenEventReweight:
    def __init__ (
        self,
        era: str = "2018",
    ):
        _data_path = Path(os.path.dirname(__file__)) / f"data/dd/WZ_inclusive_data_driven_{era}.json"
        assert _data_path.exists(), f"DataDrivenEventReweight could not find the expected json file: {str(_data_path)}"
        self.dd_estimator = correctionlib.CorrectionSet.from_file(str(_data_path)).compound["LNTTau_TTau_DD_Estimate"]
        
    def estimate_dd_DY(self, jet_multiplicity, tau_pt, systematic=None):
        if systematic is not None:
            raise NotImplementedError("DD DY Estimate doesn't yet include systematic variations")
        return self.dd_estimator.evaluate(ak.fill_none(jet_multiplicity, 0), ak.fill_none(tau_pt, 0.0))

if __name__ == "__main__":
    from correctionlib import schemav2 as cs
    # ../tau_pt_weight_2018.json src/qawa/data/dd
    new_cset_corrections = [
        cs.Correction(
            name="LNTTau_to_TTau_TransferFactor",
            version=1,
            inputs=[
                cs.Variable(name="jet_multiplicity", type="real", description="Number of jets (cross-cleaned against leptons, ID'd, pT > 25 GeV, |eta| < 2.5)"),
                cs.Variable(name="tau_pt", type="real", description="Reconstructed tau pT [GeV]"),
            ],
            output=cs.Variable(name="weight", type="real", description="Multiplicative weight for transfer factor estimating fake taus from Tight Tau to Loose Not Tight Tau rate"),
            data=cs.MultiBinning(
                nodetype="multibinning",
                inputs=["jet_multiplicity", "tau_pt"],
                edges=[[0, 1, 2],
                       [0.0, 20.0, 25.0, 30.0, 35.0, 40.0, 60.0, 80.0, 100.0, 110.0]
                       ],
                content=[0.0, 0.02539781, 0.04709555, 0.055916, 0.06757279, 0.08814366, 0.09833787, 0.09628843, 0.094179,
                         0.0, 0.00540946, 0.0045218, 0.00465039, 0.00396347, 0.00504108, 0.00541827,0.00556582, 0.01325192],
                flow="clamp",
            ),
        ),
        cs.Correction(
            name="LNTTau_HighMET_DY_to_Data_estimate",
            version=1,
            inputs=[
                cs.Variable(name="jet_multiplicity", type="real", description="Number of jets (cross-cleaned against leptons, ID'd, pT > 25 GeV, |eta| < 2.5)"),
                cs.Variable(name="tau_pt", type="real", description="Reconstructed tau pT [GeV]"),
            ],
            output=cs.Variable(name="weight", type="real", description="Multiplicative event weight for probability event would originate from DrellYan"),
            data=cs.MultiBinning(
                nodetype="multibinning",
                inputs=["jet_multiplicity", "tau_pt"],
                edges=[[0, 1, 2],
                       [0, 20, 25, 30, 35, 40, 60, 80, 100, 110]
                       ],
                content=[0.0, 0.97530646, 0.96670784, 0.95848627, 0.94992374, 0.92440374, 0.88844628, 0.8765197, 0.88508274,
                         0.0, 0.97916164, 0.98144271, 0.98085132, 0.98033892, 0.97478202, 0.9655555, 0.95486492, 0.93448176],
                flow="clamp",
            ),
        )
    ]
    new_cset_compound_corrections = [
        cs.CompoundCorrection(
            name="LNTTau_TTau_DD_Estimate",
            description="For generating data driven estimate of DrellYan background in a tight tau, high-MET region, weighting data from a loost-not-tight tau, high-MET region",
            inputs=[
                cs.Variable(name="jet_multiplicity", type="real", description="Number of jets (cross-cleaned against leptons, ID'd, pT > 25 GeV, |eta| < 2.5)"),
                cs.Variable(name="tau_pt", type="real", description="Reconstructed tau pT [GeV]"),
            ],
            output=cs.Variable(name="weight", type="real", description="Estimated number of events coming from tau fakes of a DrellYan background"),
            inputs_update=[],
            input_op="*",
            output_op="*",
            stack=["LNTTau_to_TTau_TransferFactor", "LNTTau_HighMET_DY_to_Data_estimate"],
        )
    ]
    new_cset = correctionlib.schemav2.CorrectionSet(
        schema_version=2,
        corrections=new_cset_corrections,
        compound_corrections=new_cset_compound_corrections,
    )
    with open(f"WZ_inclusive_data_driven_2018.json", "w") as fout:
        fout.write(new_cset.model_dump_json(exclude_unset=True))

    test =  DataDrivenEventReweight()
    import numpy as np
    test_njets = np.array([0, 0, 0, 1, 1])
    test_tau_pt = np.array([ 25, 80, 55, 33, 110])
    test_weights = test.estimate_dd_DY(test_njets, test_tau_pt, systematic=None)
    print("test_njets:", test_njets)
    print("test_tau_pt:", test_tau_pt)
    print("weights:", test_weights)

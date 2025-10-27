import correctionlib
import os
import awkward as ak
from pathlib import Path


class DataDrivenEventReweight:
    def __init__ (
        self,
        era: str = "2018",
    ):
        _data_path = Path(os.path.dirname(__file__)) / f"data/dd/{era}/WZ_inclusive_data_driven_{era}.json"
        assert _data_path.exists(), f"DataDrivenEventReweight could not find the expected json file: {str(_data_path)}"
        self.dd_estimator = correctionlib.CorrectionSet.from_file(str(_data_path)).compound["LNTTau_TTau_DD_Estimate"]
        
    def estimate_dd_DY(self, jet_multiplicity, tau_pt, systematic: str = None):
        if systematic is None:
            return self.dd_estimator.evaluate(ak.fill_none(jet_multiplicity, 0.0), ak.fill_none(tau_pt, 0.0), "nominal")
        else:
            return self.dd_estimator.evaluate(ak.fill_none(jet_multiplicity, 0.0), ak.fill_none(tau_pt, 0.0), systematic)
        # return self.dd_estimator.evaluate(ak.fill_none(jet_multiplicity, 0), ak.fill_none(tau_pt, 0.0), systematic)

if __name__ == "__main__":
    from correctionlib import schemav2 as cs

    new_cset_corrections = [
        # --- First Correction ---
        cs.Correction(
            name="LNTTau_to_TTau_TransferFactor",
            version=1,
            inputs=[
                cs.Variable(name="jet_multiplicity", type="real", description="Number of jets (cross-cleaned against leptons, ID'd, pT > 25 GeV, |eta| < 2.5)"),
                cs.Variable(name="tau_pt", type="real", description="Reconstructed tau pT [GeV]"),
                cs.Variable(name="systematic", type="string"),
            ],
            output=cs.Variable(name="weight", type="real", description="Multiplicative weight for transfer factor estimating fake taus from Tight Tau to Loose Not Tight Tau rate"),
            data=cs.Category(
                nodetype="category",
                input="systematic",
                content=[
                    {
                        "key": "nominal",
                        "value": cs.MultiBinning(
                            nodetype="multibinning",
                            inputs=["jet_multiplicity", "tau_pt"],
                            edges=[
                                [0, 1, 2],
                                [0.0, 20.0, 25.0, 30.0, 35.0, 40.0, 60.0, 80.0, 100.0, 110.0],
                            ],
                            content=[0.0, 0.05739619, 0.09598882, 0.11957105, 0.13560466, 0.17973915, 0.12666115, 0.19709274, 0.09991033,
                                     0.0, 0.01161966, 0.00915954, 0.00654436, 0.00940174, 0.00749029, 0.01999532, 0.02841252, 0.01023632],
                            flow="clamp",
                        ),
                    },
                    {
                        "key": "DDDYUp",
                        "value": cs.MultiBinning(
                            nodetype="multibinning",
                            inputs=["jet_multiplicity", "tau_pt"],
                            edges=[
                                [0, 1, 2],
                                [0.0, 20.0, 25.0, 30.0, 35.0, 40.0, 60.0, 80.0, 100.0, 110.0],
                            ],
                            content=[
                                0.0, 0.06051263, 0.10218493, 0.12878476, 0.14927168, 0.194802789, 0.15171089, 0.25596129, 0.14724972,
                                0.0, 0.01269104, 0.01023451, 0.00766978, 0.01108423, 0.00869699, 0.0242524, 0.037435960, 0.01691232,
                            ],
                            flow="clamp",
                        ),
                    },
                    {
                        "key": "DDDYDown",
                        "value": cs.MultiBinning(
                            nodetype="multibinning",
                            inputs=["jet_multiplicity", "tau_pt"],
                            edges=[
                                [0, 1, 2],
                                [0.0, 20.0, 25.0, 30.0, 35.0, 40.0, 60.0, 80.0, 100.0, 110.0],
                            ],
                            content=[
                                0.0, 0.05427975, 0.08979271, 0.11035734, 0.12193763, 0.16467551, 0.10161141, 0.13822419, 0.05257094,
                                0.0, 0.01054828, 0.00808457, 0.00541894, 0.00771925, 0.00628359, 0.01573824, 0.01938908, 0.00356032,
                            ],
                            flow="clamp",
                        ),
                    },
                ],
            ),
        ),

        # --- Second Correction ---
        cs.Correction(
            name="LNTTau_HighMET_DY_to_Data_estimate",
            version=1,
            inputs=[
                cs.Variable(name="jet_multiplicity", type="real"),
                cs.Variable(name="tau_pt", type="real"),
                cs.Variable(name="systematic", type="string"),
            ],
            output=cs.Variable(name="weight", type="real"),
            data=cs.Category(
                nodetype="category",
                input="systematic",
                content=[
                    {
                        "key": "nominal",
                        "value": cs.MultiBinning(
                            nodetype="multibinning",
                            inputs=["jet_multiplicity", "tau_pt"],
                            edges=[
                                [0, 1, 2],
                                [0.0, 20.0, 25.0, 30.0, 35.0, 40.0, 60.0, 80.0, 100.0, 110.0],
                            ],
                            content=[
                                0.0, 0.9626827, 0.95008219, 0.94075345, 0.93467044, 0.87811961, 0.87346451, 0.85218653, 0.88418479,
                                0.0, 0.97408868, 0.97599327, 0.97532076, 0.97283051, 0.9668564, 0.95365095, 0.92714796, 0.91683462,
                            ],
                            flow="clamp",
                        ),
                    },
                    {
                        "key": "DDDYUp",
                        "value": cs.MultiBinning(
                            nodetype="multibinning",
                            inputs=["jet_multiplicity", "tau_pt"],
                            edges=[
                                [0, 1, 2],
                                [0.0, 20.0, 25.0, 30.0, 35.0, 40.0, 60.0, 80.0, 100.0, 110.0],
                            ],
                            content=[
                                0.0, 0.96521145, 0.95437797, 0.94694696, 0.94306746, 0.88852756, 0.89533824, 0.890458, 0.9171906,
                                0.0, 0.97574809, 0.97785088, 0.97763328, 0.97592343, 0.96953928, 0.95995846, 0.94107971, 0.93342579,
                            ],
                            flow="clamp",
                        ),
                    },
                    {
                        "key": "DDDYDown",
                        "value": cs.MultiBinning(
                            nodetype="multibinning",
                            inputs=["jet_multiplicity", "tau_pt"],
                            edges=[
                                [0, 1, 2],
                                [0.0, 20.0, 25.0, 30.0, 35.0, 40.0, 60.0, 80.0, 100.0, 110.0],
                            ],
                            content=[
                                0.0, 0.96015395, 0.94578641, 0.93455994, 0.92627342, 0.86771166, 0.85159078, 0.81391506, 0.85117898,
                                0.0, 0.97242927, 0.97413566, 0.97300824, 0.96973758, 0.96417351, 0.94734344, 0.91321621, 0.90024345,
                            ],
                            flow="clamp",
                        ),
                    },
                ],
            ),
        ),
    ]

    new_cset_compound_corrections = [
        cs.CompoundCorrection(
            name="LNTTau_TTau_DD_Estimate",
            description="For generating data driven estimate of DrellYan background in a tight tau, high-MET region, weighting data from a loost-not-tight tau, high-MET region",
            inputs=[
                cs.Variable(name="jet_multiplicity", type="real", description="Number of jets (cross-cleaned against leptons, ID'd, pT > 25 GeV, |eta| < 2.5)"),
                cs.Variable(name="tau_pt", type="real", description="Reconstructed tau pT [GeV]"),
                cs.Variable(name="systematic", type="string", description="Systematic variation for the datadriven DY background estimation"),
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
    with open(f"WZ_inclusive_data_driven_2016APV_new.json", "w") as fout:
        fout.write(new_cset.model_dump_json(exclude_unset=True))

    test =  DataDrivenEventReweight()
    import numpy as np
    test_njets = np.array([0, 1, 0, 1, 0, 1])
    test_tau_pt = np.array([ 25, 25, 80, 80, 110, 110])
    for syst in ["nominal", "DDDYUp", "DDDYDown", None] :
        try:
            test_weights = test.estimate_dd_DY(test_njets, test_tau_pt, syst)
            print("test_njets:", test_njets)
            print("test_tau_pt:", test_tau_pt)
            print("weights:", test_weights)
        except Exception as e:
            print(syst, e)

    # # Wrap them into a CorrectionSet
    # cset = cs.CorrectionSet(schema_version=2, corrections=new_cset_corrections)

    # # # Example evaluations
    # print(new_cset["LNTTau_TTau_DD_Estimate"].evaluate(1, 35.0, "nominal"))
    # # print(cset["LNTTau_to_TTau_TransferFactor"].evaluate(1, 35.0, "up"))
    # # print(cset["LNTTau_to_TTau_TransferFactor"].evaluate(1, 35.0, "down"))

    # 
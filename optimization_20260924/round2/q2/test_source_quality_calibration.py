"""Safety and reproducibility checks for the optional P2 B8 adapter."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import joblib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "problem2_v2"))
from source_quality_calibration import fit_b8_calibrator, groups, read_ids, run, select_disjoint, sha256  # noqa: E402

B8_FILE = ROOT / "real_attachments/B_scaling_laws/supplementary_NQ_experiment_large.csv"
BASE_MODEL = ROOT / "problem2_v2/outputs/models/recommended_model.joblib"
CAL_IDS = HERE / "b8_calibration_ids.csv"
EVAL_IDS = HERE / "b8_evaluation_ids.csv"
Q_ONLY_CAL_IDS = HERE / "b8_one_ND_cell_per_N_calibration_ids.csv"
Q_ONLY_EVAL_IDS = HERE / "b8_one_ND_cell_per_N_evaluation_ids.csv"
Q_ONLY_DISJOINT_CAL_IDS = HERE / "b8_one_ND_cell_per_N_Q_disjoint_calibration_ids.csv"
Q_ONLY_DISJOINT_EVAL_IDS = HERE / "b8_one_ND_cell_per_N_Q_disjoint_evaluation_ids.csv"


class SourceCalibrationSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.b8 = pd.read_csv(B8_FILE)
        cls.cal_ids = read_ids(CAL_IDS)
        cls.eval_ids = read_ids(EVAL_IDS)

    def test_saved_partition_has_disjoint_rows_and_cells(self):
        cal, target = select_disjoint(self.b8, self.cal_ids, self.eval_ids)
        self.assertEqual((len(cal), len(target)), (200, 1280))
        self.assertFalse(set(cal.experiment_id) & set(target.experiment_id))
        self.assertFalse(set(groups(cal)) & set(groups(target)))

    def test_overlapping_row_and_cell_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "rows overlap"):
            select_disjoint(self.b8, [self.cal_ids[0]], [self.cal_ids[0]])
        first = self.b8.set_index("experiment_id").loc[self.cal_ids[0]]
        other = self.b8[(self.b8.N_params_B == first.N_params_B) &
                        (self.b8.D_tokens_B == first.D_tokens_B) &
                        (self.b8.experiment_id != self.cal_ids[0])].experiment_id.iloc[0]
        with self.assertRaisesRegex(ValueError, "cells overlap"):
            select_disjoint(self.b8, [self.cal_ids[0]], [other])

    def test_optional_run_preserves_canonical_model(self):
        before = sha256(BASE_MODEL)
        with tempfile.TemporaryDirectory() as name:
            result = run(BASE_MODEL, B8_FILE, CAL_IDS, EVAL_IDS, Path(name))
        self.assertEqual(before, sha256(BASE_MODEL))
        self.assertTrue(result["canonical_model_unchanged"])
        self.assertEqual(result["calibration_rows"], 200)
        self.assertEqual(result["evaluation_rows"], 1280)

    def test_full_interaction_rejects_one_cell_per_N(self):
        calibration, _ = select_disjoint(self.b8, self.cal_ids, self.eval_ids)
        one = calibration[calibration.D_tokens_B == calibration.groupby("N_params_B").D_tokens_B.transform("min")]
        with self.assertRaisesRegex(ValueError, "at least two D cells"):
            fit_b8_calibrator(one, joblib.load(BASE_MODEL), sha256(BASE_MODEL))

    def test_explicit_q_only_mode_runs_on_one_cell_per_N(self):
        before = sha256(BASE_MODEL)
        with tempfile.TemporaryDirectory() as name:
            result = run(BASE_MODEL, B8_FILE, Q_ONLY_CAL_IDS, Q_ONLY_EVAL_IDS,
                         Path(name), mode="q_only")
            calibrator = joblib.load(Path(name) / "b8_labeled_source_calibrator.joblib")
        self.assertEqual(result["mode"], "q_only")
        self.assertEqual(calibrator["mode"], "q_only")
        self.assertEqual((result["calibration_rows"], result["evaluation_rows"]), (100, 1380))
        self.assertAlmostEqual(result["metrics"]["source_q_only"]["rmse"], .2386990646, places=6)
        self.assertEqual(before, sha256(BASE_MODEL))

    def test_q_only_unseen_Q_levels_and_cells(self):
        cal, target = select_disjoint(self.b8, read_ids(Q_ONLY_DISJOINT_CAL_IDS),
                                      read_ids(Q_ONLY_DISJOINT_EVAL_IDS))
        self.assertFalse(set(cal.Q_score) & set(target.Q_score))
        self.assertFalse(set(groups(cal)) & set(groups(target)))
        before = sha256(BASE_MODEL)
        with tempfile.TemporaryDirectory() as name:
            result = run(BASE_MODEL, B8_FILE, Q_ONLY_DISJOINT_CAL_IDS,
                         Q_ONLY_DISJOINT_EVAL_IDS, Path(name), mode="q_only")
        self.assertEqual((result["calibration_rows"], result["evaluation_rows"]), (51, 777))
        self.assertAlmostEqual(result["metrics"]["source_q_only"]["rmse"], .2207246901, places=6)
        self.assertEqual(before, sha256(BASE_MODEL))


if __name__ == "__main__":
    unittest.main()

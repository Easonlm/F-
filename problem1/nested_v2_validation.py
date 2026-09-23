"""Nested train-only CV tests the whole v2 selection procedure without A6-A15."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from mixture_pipeline import SEED, TABLES, alr, pair
from problem1_v2 import candidates, estimator, transform


def run():
    _, p, y, _, _ = pair("train", "1m")
    outer = KFold(5, shuffle=True, random_state=SEED + 1)
    oof_v1 = np.full_like(y, np.nan)
    oof_v2 = np.full_like(y, np.nan)
    rows = []
    for outer_fold, (outer_train, outer_test) in enumerate(outer.split(p), 1):
        eps_v1 = float(p[outer_train][p[outer_train] > 0].min() / 2)
        inner = list(KFold(3, shuffle=True, random_state=SEED + outer_fold).split(outer_train))
        best = (np.inf, None)
        for spec in candidates(p[outer_train]):
            family, eps, ref, alpha, gamma = spec
            x = transform(p[outer_train], family, eps, ref)
            mse = []
            for tr, val in inner:
                m = estimator(family, alpha, gamma)
                m.fit(x[tr], y[outer_train][tr])
                mse.append(mean_squared_error(y[outer_train][val], m.predict(x[val])))
            score = float(np.mean(mse))
            if score < best[0]:
                best = (score, spec)
        family, eps, ref, alpha, gamma = best[1]
        model_v2 = estimator(family, alpha, gamma)
        model_v2.fit(transform(p[outer_train], family, eps, ref), y[outer_train])
        oof_v2[outer_test] = model_v2.predict(transform(p[outer_test], family, eps, ref))
        model_v1 = estimator("alr", 1.0)
        model_v1.fit(alr(p[outer_train], eps_v1), y[outer_train])
        oof_v1[outer_test] = model_v1.predict(alr(p[outer_test], eps_v1))
        rows.append({"outer_fold": outer_fold, "selected_family": family, "selected_eps": eps,
                     "selected_ref": ref, "selected_alpha": alpha, "selected_gamma": gamma,
                     "inner_rmse": float(np.sqrt(best[0])),
                     "v1_outer_rmse": float(np.sqrt(mean_squared_error(y[outer_test], oof_v1[outer_test]))),
                     "v2_outer_rmse": float(np.sqrt(mean_squared_error(y[outer_test], oof_v2[outer_test])))})
    assert np.all(np.isfinite(oof_v1)) and np.all(np.isfinite(oof_v2))
    result = pd.DataFrame(rows)
    result.to_csv(TABLES / "v2_nested_cv_folds.csv", index=False)
    comparison = pd.DataFrame([{"version": "v1", "nested_oof_rmse": np.sqrt(mean_squared_error(y, oof_v1))},
                               {"version": "v2", "nested_oof_rmse": np.sqrt(mean_squared_error(y, oof_v2))}])
    comparison.to_csv(TABLES / "v2_nested_cv_summary.csv", index=False)
    print(result.to_string(index=False))
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    run()

"""Reproducible, blinded A1 text sample for two independent human raters."""
from __future__ import annotations

import json
import lzma
from pathlib import Path

import numpy as np
import pandas as pd

from quality_pipeline import input_files


HERE = Path(__file__).resolve().parent
TABLES = HERE / "outputs" / "tables"
SEED = 20260923


def run():
    score = pd.read_parquet(TABLES / "sample_quality_scores.parquet")
    a1 = score.loc[score.source.eq("A1")].reset_index(drop=True).copy()
    rng = np.random.default_rng(SEED)
    chosen = []
    for domain, part in a1.groupby("domain"):
        # Four score quartiles per domain plus an extra conflict stratum.
        quartile = pd.qcut(part.Q_baseline.rank(method="first"), 4, labels=False)
        for q in range(4):
            eligible = part.index[quartile.eq(q)].to_numpy()
            chosen.extend(rng.choice(eligible, size=min(4, len(eligible)), replace=False).tolist())
        conflict = part.index[part.conflict_any & ~part.index.isin(chosen)].to_numpy()
        chosen.extend(rng.choice(conflict, size=min(4, len(conflict)), replace=False).tolist())
    chosen = sorted(set(chosen))
    content = {}
    source_path = input_files()[0][1]
    with lzma.open(source_path, "rt", encoding="utf-8") as fh:
        wanted = set(chosen)
        for pos, line in enumerate(fh):
            if pos in wanted:
                obj = json.loads(line)
                content[pos] = str(obj.get("content", ""))[:4000]
    if len(content) != len(chosen):
        raise ValueError("Could not recover every selected A1 text")
    rng.shuffle(chosen)
    review, key = [], []
    for i, pos in enumerate(chosen, 1):
        rid = f"V2-{i:03d}"
        text = content[pos]
        review.append({"review_id": rid, "text": text})
        row = a1.iloc[pos]
        key.append({"review_id": rid, "A1_row_1based": pos + 1, "id": str(row.id),
                    "domain": row.domain, "Q_baseline": float(row.Q_baseline),
                    "conflict_any": bool(row.conflict_any), "content_chars_shown": len(text)})
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "v2_blind_review_items.json").write_text(json.dumps(review, ensure_ascii=False), encoding="utf-8")
    (TABLES / "v2_blind_review_key.json").write_text(json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"sample_size": len(review), "domains": len(a1.domain.unique()),
                      "conflict_count": sum(x["conflict_any"] for x in key)}, ensure_ascii=False))


if __name__ == "__main__":
    run()

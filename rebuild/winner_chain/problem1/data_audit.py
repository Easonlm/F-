"""Read-only audit for every A1-A15 source file; no raw data is modified."""
from pathlib import Path
import csv
import json
import lzma
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"real_attachments"/"A_data_value"
TABLE=Path(__file__).resolve().parent/"outputs"/"tables"


def run():
    TABLE.mkdir(parents=True,exist_ok=True)
    rows=[]
    quality=[("A1",BASE/"slimpajama_quality_signal_sample.jsonl.xz"),
             ("A2",next((BASE/"slimpajama_quality_extended").glob("arxiv_*.xz"))),
             ("A3",next((BASE/"slimpajama_quality_extended").glob("github_*.xz")))]
    with open(TABLE/"quality_audit.json",encoding="utf-8") as fh:q=json.load(fh)
    for code,path in quality:
        with lzma.open(path,"rt",encoding="utf-8") as fh:fields=list(json.loads(fh.readline()))
        rows.append({"code":code,"file":str(path.relative_to(ROOT)),"rows":q["source_counts"][code],"columns":len(fields),
                     "bytes":path.stat().st_size,"kind":"observed sample" if code=="A1" else "observed extended",
                     "missing_cells":pd.NA,"duplicate_index":pd.NA,"simplex_max_deviation":pd.NA,"fields":"|".join(fields)})
    paths=[("A4","train_mixture_1m.csv","observed mixture"),("A5","train_pile_loss_1m.csv","observed loss"),
           ("A6","test_mixture_1m.csv","observed mixture"),("A7","test_pile_loss_1m.csv","observed loss"),
           ("A8","test_mixture_60m.csv","observed mixture"),("A9","test_pile_loss_60m.csv","observed loss"),
           ("A10","test_mixture_1B.csv","observed mixture"),("A11","test_pile_loss_1B.csv","observed loss"),
           ("A12","est_mixture_10b.csv","training-mixture subset"),("A13","est_pile_loss_10b.csv","model-estimated loss"),
           ("A14","est_mixture_70b.csv","training-mixture subset"),("A15","est_pile_loss_70b.csv","model-estimated loss")]
    for code,name,kind in paths:
        path=BASE/"regmix_tables"/name;d=pd.read_csv(path)
        vals=d.iloc[:,1:].apply(pd.to_numeric,errors="coerce")
        simple=abs(vals.sum(axis=1)-1).max() if "mixture" in name else pd.NA
        rows.append({"code":code,"file":str(path.relative_to(ROOT)),"rows":len(d),"columns":len(d.columns),
                     "bytes":path.stat().st_size,"kind":kind,"missing_cells":int(vals.isna().sum().sum()),
                     "duplicate_index":int(d["index"].duplicated().sum()),"simplex_max_deviation":simple,"fields":"|".join(d.columns)})
    pd.DataFrame(rows).to_csv(TABLE/"data_audit.csv",index=False,encoding="utf-8-sig")
    print(f"Audited {len(rows)} files")

if __name__=="__main__":run()

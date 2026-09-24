"""One-command reproduction: python problem4/run_problem4.py."""
from __future__ import annotations

import json

from data_pipeline import OUT, audit, prepare, parse_c8, attach_c8
from bridge_model import run_bridge
from history_forecast import frontier, historical, compute_growth, contributions, backtest, forecast
from mechanism import run_mechanism
from build_report import build
from verify_problem4 import verify


def main():
    inputs=audit()
    a=prepare(inputs)
    agg,raw,c8summary=parse_c8()
    c8matched,_=attach_c8(a,agg,raw)
    bridge,_,cv,_=run_bridge(inputs["c5"],inputs["c6"])
    q=frontier(a)
    d,b,resid,boots=historical(a)
    origin=a.loc[a.open_w1,"submission_date"].max()
    compute,growth=compute_growth(inputs["c4"],origin)
    contributions(q,b,boots)
    backtest(q)
    f=forecast(q,b,boots,growth,resid,origin,compute)
    run_mechanism(f,bridge)
    result=build(a,inputs,c8summary,c8matched,d,b,cv,origin)
    checks=verify()
    print(json.dumps({"origin":str(origin.date()),"open_models":result["problem4"]["open_models"],"c8_parsed":c8summary["parsed_models"],
                      "verification_passed":sum(v["passed"] for v in checks.values()),"verification_total":len(checks)},ensure_ascii=False))


if __name__=="__main__":main()

#!/usr/bin/env python3
"""Offline, complete policy research run; never creates a PDF or slides."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'models'))
from common import DataStore,write_json
from policy_calibration import calibrate
from policy_decision import run_decisions
from policy_history import run_historical_validation
from policy_nowcast import run_nowcasts
from policy_qpm import PolicyQPM,anticipated_path


def historical_factory(settings=None):
    # The 2026 teaching judgment cannot appear in a 2022 information set.
    return PolicyQPM(settings=settings,gap_judgments={})


def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    subprocess.run([sys.executable,str(ROOT/'tools/build_policy_dataset.py')],cwd=ROOT,check=True)
    store=DataStore()
    selected,calibration=calibrate(store,output/'calibration')
    run_nowcasts(output/'nowcast',ROOT/'data/policy/2026-10-22/monthly.csv')
    run_decisions(output,selected)
    history=[]
    for name,factory,solver in [('legacy',None,None),
            ('corrected_no_retrospective_judgment',historical_factory,anticipated_path)]:
        kwargs={} if factory is None else dict(model_factory=factory,path_solver=solver)
        result=run_historical_validation(output/'history'/name,store=store,**kwargs)
        history.append(result['metrics'].assign(variant=name))
    import pandas as pd
    pd.concat(history,ignore_index=True).to_csv(output/'history/historical_variant_comparison.csv',index=False)
    try:
        from summarize_policy_run import summarize
        summarize(output)
    except ImportError:
        # No research result is silently replaced by a narrative-only output.
        raise RuntimeError('Missing tools/summarize_policy_run.py')
    tracked=[]
    for folder in [ROOT/'models',ROOT/'tools',ROOT/'tests',ROOT/'data/inputs',ROOT/'data/model',
                   ROOT/'data/policy/2026-10-22',ROOT/'nbu_challenge_model_2026/model']:
        tracked += [p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    inputs={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(tracked)}
    outputs={str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(output.rglob('*')) if p.is_file() and p.name not in ['run_manifest.json','verification.json']}
    import numpy,pandas,scipy,statsmodels
    write_json(output/'run_manifest.json',dict(as_of='2026-10-22',network='offline saved-source replay',
        python=platform.python_version(),numpy=numpy.__version__,pandas=pandas.__version__,
        scipy=scipy.__version__,statsmodels=statsmodels.__version__,
        selected_settings=selected,inputs=inputs,outputs=outputs,
        status='completed; verification is a separate command',
        historical_context='35 available exchanges in the model chat plus original and environment chats recovered; sources remain read-only',
        distinction='headline/statistical forecasts versus conditional structural decision experiments; no approved policy recommendation'))
    print('Policy research completed:',output,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'results/2026-10-22-policy-decision')
    run(p.parse_args().output)

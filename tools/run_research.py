"""One offline reproduction of evidence, monthly audit and QPM policy research."""
from pathlib import Path
import os,subprocess,sys,hashlib,json,platform

ROOT=Path(__file__).resolve().parents[1]

def main():
    for script in ['tools/build_research_evidence.py','models/research_monthly.py',
                   'models/research_qpm.py','models/make_research_report.py']:
        print('\nRunning',script,flush=True)
        subprocess.run([sys.executable,str(ROOT/script)],cwd=ROOT,check=True,env=os.environ.copy())
    import numpy,pandas,scipy,statsmodels,matplotlib
    inputs=list((ROOT/'models').glob('*.py'))+[Path(__file__),ROOT/'tools/build_research_evidence.py']
    inputs+=list((ROOT/'data/inputs').glob('*.csv'))+list((ROOT/'data/model').glob('*.csv'))
    inputs+=list((ROOT/'nbu_challenge_model_2026/model').glob('*'))
    inputs+=list((ROOT/'data/research/2026-10-22/snapshots').glob('*'))
    inputs+=[ROOT/'data/snapshots/2026-10-22/policy_archive.json']
    inputs+=[ROOT/'results/2026-10-22-monthly'/name for name in ['backtest_predictions.csv','next_month_forecast.csv']]
    outputs=list((ROOT/'results/2026-10-22-research').glob('*'))+list((ROOT/'reports/2026-10-22-research').glob('*'))
    outputs+=list((ROOT/'data/research/2026-10-22').glob('*.csv'))
    outputs+=list((ROOT/'data/research/2026-10-22/excel_exports').glob('*.csv'))
    digest=lambda paths:{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths)) if p.is_file() and p.name!='research_run_manifest.json'}
    manifest=dict(as_of='2026-10-22',network_required=False,
                  versions=dict(python=platform.python_version(),numpy=numpy.__version__,pandas=pandas.__version__,
                                scipy=scipy.__version__,statsmodels=statsmodels.__version__,matplotlib=matplotlib.__version__),
                  input_sha256=digest(inputs),output_sha256=digest(outputs))
    (ROOT/'results/2026-10-22-research/research_run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('\nResearch reproduced. Read reports/2026-10-22-research/RESEARCH_UA.md')

if __name__=='__main__':main()

"""Default: three-month research only. Medium-term archive is opt-in."""
from pathlib import Path
import os,subprocess,sys,hashlib,json,platform,argparse

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--refresh',action='store_true',help='Refit historical base models instead of verified cache')
    parser.add_argument('--include-medium-term',action='store_true',help='Also reproduce archived extended research')
    args=parser.parse_args()
    steps=[['tools/build_research_evidence.py'],['models/research_short_term.py']+(['--refresh'] if args.refresh else []),
           ['models/make_short_term_report.py']]
    if args.include_medium_term:
        steps += [[s] for s in ['models/research_monthly.py','models/research_qpm.py','models/make_medium_term_report.py']]
    for step in steps:
        print('\nRunning',step[0],flush=True)
        subprocess.run([sys.executable,str(ROOT/step[0]),*step[1:]],cwd=ROOT,check=True,env=os.environ.copy())
    import numpy,pandas,scipy,statsmodels,matplotlib
    inputs=[Path(__file__),ROOT/'models/make_short_term_report.py',ROOT/'tools/build_research_evidence.py']
    cache=json.loads((ROOT/'results/2026-10-22-short-term/base_cache_manifest.json').read_text())
    inputs += [ROOT/p for p in cache['input_sha256']]
    outputs=list((ROOT/'results/2026-10-22-short-term').glob('*'))+list((ROOT/'reports/2026-10-22-research').glob('*'))
    digest=lambda paths:{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths)) if p.is_file() and p.name!='run_manifest.json'}
    manifest=dict(as_of='2026-10-22',target_months=['2026-11','2026-12','2027-01'],network_required=False,
                  versions=dict(python=platform.python_version(),numpy=numpy.__version__,pandas=pandas.__version__,
                                scipy=scipy.__version__,statsmodels=statsmodels.__version__,matplotlib=matplotlib.__version__),
                  input_sha256=digest(inputs),output_sha256=digest(outputs))
    (ROOT/'results/2026-10-22-short-term/run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('\nThree-month research reproduced: reports/2026-10-22-research/RESEARCH_UA.md')

if __name__=='__main__':main()

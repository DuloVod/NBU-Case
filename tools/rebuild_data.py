"""Rebuild source data in a separate directory; compare without overwriting."""
from pathlib import Path
import argparse,json,shutil,subprocess,sys,tempfile
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def rebuild(destination):
    shutil.copytree(ROOT/'data',destination,ignore=shutil.ignore_patterns('__pycache__','extracted','.DS_Store'))
    for script in ['audit_inputs.py','prepare_dataset.py']:
        subprocess.run([sys.executable,str(destination/'reproduce'/script)],check=True,cwd=destination)
    for path in (destination/'data').glob('*.csv'):
        shutil.copy2(path,destination/'inputs'/path.name)
    subprocess.run([sys.executable,str(destination/'reproduce/calibrate.py'),'--root',str(destination)],check=True,cwd=destination)
    return destination


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check',action='store_true')
    group.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.check:
        with tempfile.TemporaryDirectory(prefix='nbu-data-rebuild-') as tmp:
            generated=rebuild(Path(tmp)/'dataset')
            compared=[]
            for original in sorted((ROOT/'data/model').glob('*.csv')):
                actual=pd.read_csv(generated/'model'/original.name)
                expected=pd.read_csv(original)
                pd.testing.assert_frame_equal(actual,expected,check_dtype=False,check_exact=False,rtol=1e-10,atol=1e-9)
                compared.append(original.name)
            print(json.dumps({'status':'PASS','model_csv_files':len(compared),'files':compared},ensure_ascii=False,indent=2))
    else:
        target=args.output.resolve()
        if target.exists():parser.error('Output must be a new directory; refusing to replace existing files')
        rebuild(target)
        print(f'Prepared independent dataset: {target}')

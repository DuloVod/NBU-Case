"""Verify tracked raw-source hashes and the seven-observable core."""
from pathlib import Path
import csv,hashlib,unittest
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


class DataIntegrityTests(unittest.TestCase):
    def test_original_source_hashes(self):
        with (ROOT/'data/documentation/raw_manifest.csv').open() as stream:
            for item in csv.DictReader(stream):
                with self.subTest(file=item['file']):
                    file=ROOT/'data/raw'/item['file']
                    self.assertTrue(file.is_file())
                    self.assertEqual(hashlib.sha256(file.read_bytes()).hexdigest(),item['sha256'])

    def test_core_shape_types_and_missing_latest_gdp(self):
        core=pd.read_csv(ROOT/'data/model/quarterly_training.csv',index_col=0)
        self.assertEqual(core.shape,(66,7))
        self.assertTrue(np.isfinite(core.to_numpy(float)).all())
        self.assertEqual(core.index[0],'2010Q1');self.assertEqual(core.index[-1],'2026Q2')
        levels=pd.read_csv(ROOT/'data/model/quarterly_levels.csv',index_col=0)
        self.assertTrue(levels.loc['2026Q3',['gdp','gdp_ea']].isna().all())


if __name__=='__main__':unittest.main()

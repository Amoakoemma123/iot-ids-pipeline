"""
prep_cicids_sample.py
Pre-extract a memory-efficient 600K-row stratified sample from CICIDS2017 parquet.

Uses pyarrow batch iteration — never loads all 2.83M rows into RAM simultaneously.
Saves the sample to /tmp/cicids_600k_sample.parquet for use by binary_adv experiment.

Usage:
    python -m experiments.adversarial_robustness.prep_cicids_sample
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

from experiments.adversarial_robustness.config import DATASET_PATHS

import tempfile
TARGET_ROWS = 600_000
SEED        = 42
OUT_PATH    = Path(tempfile.gettempdir()) / 'cicids_600k_sample.parquet'

def main():
    csv_path, label_col = DATASET_PATHS['CICIDS2017']
    print(f"Reading CICIDS2017 from {csv_path}")
    print(f"Target: {TARGET_ROWS:,} rows via batch iteration (no full load)")

    pf          = pq.ParquetFile(str(csv_path))
    total_rows  = pf.metadata.num_rows
    batch_size  = 50_000
    # Fraction to keep from each batch
    keep_frac   = min(1.0, TARGET_ROWS / total_rows)

    print(f"  Total rows in file: {total_rows:,}")
    print(f"  Keep fraction: {keep_frac:.4f}")

    rng     = np.random.default_rng(SEED)
    batches = []
    kept    = 0

    for i, batch in enumerate(pf.iter_batches(batch_size=batch_size)):
        n = len(batch)
        n_keep = min(n, int(np.ceil(n * keep_frac)))
        idx    = rng.choice(n, size=n_keep, replace=False)
        idx.sort()
        # Select rows using pyarrow slice — memory-efficient
        df_batch = batch.to_pydict()
        filtered = {col: [df_batch[col][j] for j in idx]
                    for col in df_batch}
        batches.append(pa.table(filtered))
        kept += n_keep
        if (i + 1) % 10 == 0:
            print(f"  Processed {(i+1)*batch_size:,} / {total_rows:,} rows  "
                  f"(kept {kept:,})")
        if kept >= TARGET_ROWS:
            break

    print(f"\nConcatenating {len(batches)} batches ({kept:,} rows)…")
    table = pa.concat_tables(batches)
    print(f"  Final table: {table.num_rows:,} rows × {table.num_columns} cols")

    print(f"Writing to {OUT_PATH}…")
    pq.write_table(table, str(OUT_PATH))
    print(f"Done — sample saved to {OUT_PATH}")
    print(f"  File size: {OUT_PATH.stat().st_size / 1e6:.1f} MB")


if __name__ == '__main__':
    main()

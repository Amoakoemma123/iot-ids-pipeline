"""
src/data/loader.py
Loads and cleans the TON_IoT network traffic dataset.
Matches the notebook cleaning pipeline exactly (IDS_IoT_Training.ipynb, Cell 6).
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# High-cardinality identifier columns — dropped entirely (not encodable)
_DROP_COLS = [
    'src_ip', 'dst_ip',
    'dns_query',
    'ssl_subject', 'ssl_issuer', 'ssl_cipher', 'ssl_version',
    'http_uri', 'http_user_agent', 'http_orig_mime_types', 'http_resp_mime_types',
    'weird_name', 'weird_addl', 'weird_notice',
    # UNSW-NB15 sequential row index: carries no traffic information but is
    # correlated with the label through the file's ordering (v1 leakage; the
    # v1 RFECV kept it and SHAP ranked it first). Dropped from v2 onwards.
    'id',
]

# Port columns, removed only when drop_ports=True (shortcut-learning control:
# testbed services sit on fixed ports, so ports can proxy the attack label).
_PORT_COLS = ['src_port', 'dst_port', 'Source Port', 'Destination Port',
              'sport', 'dsport']

# Threshold for one-hot encoding categorical columns (≤ this many unique values)
_OHE_CARDINALITY = 30


def load_dataset(csv_path: str | Path,
                 label_col: str = 'type',
                 test_size: float = 0.15,
                 val_size: float = 0.15,
                 seed: int = 42,
                 drop_ports: bool = False,
                 drop_id: bool = True,
                 deduplicate: bool = True,
                 return_meta: bool = False) -> dict:
    """
    Load, clean and split the dataset.

    Cleaning steps (mirrors notebook Cell 6):
      1. Drop identifier / high-cardinality columns
      2. Drop binary 'label' column when multi-class 'type' is used
      3. Normalise strings (strip + lowercase)
      4. Convert numeric-looking object columns; drop high-cardinality and
         one-hot encode low-cardinality categoricals (≤ 30 unique values);
         optionally drop port columns
      5. Remove exact duplicate rows (after every column drop, v2.1)
      -- stratified 70/15/15 split --
      6. Drop numeric columns with > 40% missing values  (train rows only)
      7. Median imputation for numeric NaNs               (train medians)
      8. Replace ±inf with column 99th percentile         (train p99)
      9. Drop zero-variance features                      (train std)
      10. Cross-split guard: drop val/test rows identical (features + label)
          to a training row

    Steps 6–9 are statistics fitted on the training partition only.

    Returns
    -------
    dict with keys:
        X_train, X_val, X_test  : float32 arrays
        y_train, y_val, y_test  : integer arrays
        feature_names            : list of feature column names (after OHE)
        class_names              : list of class name strings
        label_encoder            : fitted LabelEncoder
    """
    csv_path = Path(csv_path)
    if csv_path.suffix == '.parquet':
        df = pd.read_parquet(csv_path)
    else:
        df = pd.read_csv(csv_path, low_memory=False)
    df.columns = df.columns.str.strip()   # normalise column name whitespace (e.g. CICIDS ' Label')
    print(f"Loaded {len(df):,} rows × {df.shape[1]} cols from {Path(csv_path).name}")

    # ── 1. Drop identifier columns ────────────────────────────────────────────
    # (return_meta: keep the IP addresses aside, aligned by row index, e.g. for a
    #  natural device-level federated partition; they never become features)
    meta = df[[c for c in ('src_ip', 'dst_ip') if c in df.columns]].copy() if return_meta else None
    drop = [c for c in _DROP_COLS if c in df.columns and (drop_id or c != 'id')]
    df.drop(columns=drop, inplace=True)

    # ── 2. Drop binary label when multi-class label exists ────────────────────
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found. "
                         f"Available: {df.columns.tolist()}")
    if 'label' in df.columns and label_col != 'label':
        df.drop(columns=['label'], inplace=True)
    df.dropna(subset=[label_col], inplace=True)

    # ── 3. Normalise strings ──────────────────────────────────────────────────
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.strip().str.lower()

    # ── 4. Encode categoricals ───────────────────────────────────────────────
    # Fill NaN in object columns with 'unknown' so sparse categoricals (e.g.
    # ssl_resumed, http_method in TON_IoT) are not discarded before encoding.
    # This matches the paper's pipeline and recovers ~14 extra features on TON_IoT.
    cat_cols = [c for c in df.select_dtypes(include='object').columns
                if c != label_col]
    # Try to coerce numeric-looking strings first
    for col in cat_cols:
        try:
            df[col] = pd.to_numeric(df[col], errors='raise')
        except (ValueError, TypeError):
            pass   # genuine categorical — keep as object

    remaining_cat = [c for c in df.select_dtypes(include='object').columns
                     if c != label_col]
    # Fill NaN with 'unknown' so high-missing categoricals survive as OHE features
    if remaining_cat:
        df[remaining_cat] = df[remaining_cat].fillna('unknown')
    high_card = [c for c in remaining_cat if df[c].nunique() > _OHE_CARDINALITY]
    low_card  = [c for c in remaining_cat if df[c].nunique() <= _OHE_CARDINALITY]
    if high_card:
        df.drop(columns=high_card, inplace=True)
    if low_card:
        df = pd.get_dummies(df, columns=low_card, drop_first=True)
        print(f"One-hot encoded {len(low_card)} columns: {low_card}")

    if drop_ports:
        ports = [c for c in _PORT_COLS if c in df.columns]
        df.drop(columns=ports, inplace=True)
        print(f"drop_ports=True: removed {ports}")

    # ── 5. Remove duplicates (after every column drop) ────────────────────────
    # v2.1: deduplicate only once all identifier / high-cardinality columns are
    # gone; deduplicating earlier (v1, v2.0) left records that differ only in a
    # dropped column (e.g. UNSW `proto`) as exact duplicates of each other.
    # (drop_id / deduplicate switches exist only for the Paper 5 leakage decomposition;
    #  duplicates are always judged on all columns except `id`.)
    n_before = len(df)
    if deduplicate:
        df.drop_duplicates(subset=[c for c in df.columns if c != 'id'], inplace=True)
    print(f"Removed {n_before - len(df):,} duplicate rows.")

    # ── Encode labels ─────────────────────────────────────────────────────────
    le = LabelEncoder()
    y = le.fit_transform(df[label_col])
    df.drop(columns=[label_col], inplace=True)
    if meta is not None:
        meta = meta.loc[df.index].reset_index(drop=True)
    df.reset_index(drop=True, inplace=True)

    # ── Stratified split: train / val / test (70 / 15 / 15) ──────────────────
    # Split row indices first so that every statistic below is train-only.
    # Fall back to random split if any class is too rare for stratification
    # (e.g. CICIDS2017 Heartbleed=11 → only ~1 sample lands in val partition).
    idx = np.arange(len(df))
    val_frac_of_trainval = val_size / (1.0 - test_size)
    try:
        idx_tv, idx_test, y_tv, y_test = train_test_split(
            idx, y, test_size=test_size, stratify=y, random_state=seed)
    except ValueError:
        print("  Warning: classes too rare for stratified test split — using random split")
        idx_tv, idx_test, y_tv, y_test = train_test_split(
            idx, y, test_size=test_size, stratify=None, random_state=seed)
    try:
        idx_train, idx_val, y_train, y_val = train_test_split(
            idx_tv, y_tv, test_size=val_frac_of_trainval, stratify=y_tv, random_state=seed)
    except ValueError:
        print("  Warning: classes too rare for stratified val split — using random split")
        idx_train, idx_val, y_train, y_val = train_test_split(
            idx_tv, y_tv, test_size=val_frac_of_trainval, stratify=None, random_state=seed)

    train_df = df.iloc[idx_train]

    # ── 6. Drop NUMERIC columns with > 40% missing (train rows) ──────────────
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    missing_pct = train_df[num_cols].isnull().mean()
    high_missing = missing_pct[missing_pct > 0.40].index.tolist()
    if high_missing:
        df.drop(columns=high_missing, inplace=True)

    # ── 7. Median imputation (train medians) ──────────────────────────────────
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    train_df = df.iloc[idx_train]
    df[num_cols] = df[num_cols].fillna(train_df[num_cols].median())

    # ── 8. Replace ±inf with train 99th-percentile ────────────────────────────
    train_df = df.iloc[idx_train]
    for col in num_cols:
        p99 = train_df[col].replace([np.inf, -np.inf], np.nan).quantile(0.99)
        df[col] = df[col].replace([np.inf, -np.inf], p99)

    # ── 9. Drop zero-variance features (train std) ────────────────────────────
    train_df = df.iloc[idx_train]
    zero_var = [c for c in num_cols if train_df[c].std() == 0]
    if zero_var:
        df.drop(columns=zero_var, inplace=True)

    print(f"Shape after cleaning: {df.shape}")

    feature_names = df.columns.tolist()
    X = df.values.astype(np.float32)

    # ── 10. Cross-split guard ─────────────────────────────────────────────────
    # Train-fitted column drops (steps 6 and 9) can make a val/test record an
    # exact copy of a training record again. Remove such records (same features
    # and label) from val and test, so that no test record was seen in training.
    def _keys(A, yy):
        return pd.util.hash_pandas_object(
            pd.DataFrame(np.column_stack([A, yy])), index=False).values
    train_keys = set(_keys(X[idx_train], y_train))
    dedup_info = {}
    for name in (('val', 'test') if deduplicate else ()):
        ix = idx_val if name == 'val' else idx_test
        yy = y_val if name == 'val' else y_test
        k = _keys(X[ix], yy)
        keep = np.array([kk not in train_keys for kk in k])
        dedup_info[f'{name}_removed_cross_split'] = int((~keep).sum())
        if name == 'val':
            idx_val, y_val = idx_val[keep], y_val[keep]
        else:
            idx_test, y_test = idx_test[keep], y_test[keep]
    if deduplicate:
        print(f"Cross-split guard removed {dedup_info['val_removed_cross_split']:,} val and "
              f"{dedup_info['test_removed_cross_split']:,} test rows that duplicate a training row.")

    X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]
    print(f"Classes ({len(le.classes_)}): {list(le.classes_)}")
    print(f"X shape: {X.shape}")
    print(f"Split → train: {len(X_train):,}  val: {len(X_val):,}  test: {len(X_test):,}")

    extra = {}
    if meta is not None:
        extra = {f'meta_{n}': meta.iloc[ix].reset_index(drop=True) for n, ix in
                 (('train', idx_train), ('val', idx_val), ('test', idx_test))}
    return dict(
        **extra,
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
        feature_names=feature_names,
        class_names=list(le.classes_),
        label_encoder=le,
        dedup_info=dedup_info,
    )

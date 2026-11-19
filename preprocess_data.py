#!/usr/bin/env python3
"""Prepare CIC-IDS2017 for binary classification.
Install: python -m pip install numpy pandas scikit-learn joblib
Run: python preprocess_data.py --data-dir data/MachineLearningCVE
Outputs raw/scaled .npy arrays, metadata CSVs, fitted scaler and JSON report.
This is a randomized within-dataset baseline, not an unseen-attack evaluation.
Loads the combined numeric dataset into memory; allow several GB of free RAM.
"""
import argparse
import csv
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def headers(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        original = next(csv.reader(f))
    used, names = set(), []
    for name in original:
        name = name.strip()
        candidate, n = name, 2
        while candidate in used:
            candidate = f'{name}__{n}'
            n += 1
        used.add(candidate)
        names.append(candidate)
    return names


def normalize_label(label):
    label = str(label).strip()
    if label.startswith('Web Attack'):
        # Keep the attack subtype, replacing broken separator characters.
        for subtype in ['Brute Force', 'XSS', 'Sql Injection']:
            if label.endswith(subtype):
                return 'Web Attack - ' + subtype
    return re.sub(r'\s+', ' ', label)


def counts(series):
    return {str(k): int(v) for k, v in series.value_counts().items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=Path('data'))
    p.add_argument('--output-dir', type=Path, default=Path('data/processed'))
    p.add_argument('--seed', type=int, default=195)
    p.add_argument('--train-ratio', type=float, default=.6)
    p.add_argument('--val-ratio', type=float, default=.2)
    p.add_argument('--test-ratio', type=float, default=.2)
    args = p.parse_args()
    ratios = np.array([args.train_ratio, args.val_ratio, args.test_ratio])
    if not np.all(np.isfinite(ratios)) or np.any(ratios <= 0) or not np.isclose(ratios.sum(), 1):
        p.error('Split ratios must be positive and sum to 1.')
    if not args.data_dir.is_dir():
        p.error(f'Data directory not found: {args.data_dir}')
    files = sorted(f for f in args.data_dir.rglob('*') if f.is_file() and f.suffix.lower() == '.csv')
    if not files:
        p.error('No CSV files found.')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        p.error('Output directory is not empty. Choose a new --output-dir to preserve previous results.')
    frames, metadata, schema, file_report = [], [], None, []
    for path in files:
        print(f'Loading {path.name} ...', flush=True)
        names = headers(path)
        if 'Label' not in names:
            raise ValueError(f'Label column missing: {path}')
        frame = pd.read_csv(path, encoding='utf-8-sig', low_memory=False)
        frame.columns = names
        labels = frame.pop('Label').map(normalize_label)
        if labels.str.lower().isin(['', 'nan', 'none', 'null', '<na>']).any():
            raise ValueError(f'Missing labels in {path}; resolve before splitting.')
        numeric = frame.apply(pd.to_numeric, errors='coerce').astype('float64')
        if schema is None:
            schema = list(numeric.columns)
        if set(numeric.columns) != set(schema):
            raise ValueError(f'Feature schema differs in {path}')
        numeric = numeric[schema]
        valid = np.isfinite(numeric.to_numpy()).all(axis=1)
        file_report.append({'file': str(path), 'input_rows': len(frame),
                            'dropped_nonfinite_by_label': counts(labels[~valid])})
        frames.append(numeric.loc[valid].reset_index(drop=True))
        metadata.append(pd.DataFrame({
            'attack_label': labels[valid].to_numpy(),
            'source_file': str(path.relative_to(args.data_dir)),
            'source_row': np.flatnonzero(valid) + 2,
        }))
    X = pd.concat(frames, ignore_index=True)
    meta = pd.concat(metadata, ignore_index=True)
    del frames, metadata, frame, numeric
    if X.empty:
        raise ValueError('No finite rows remain.')
    report = {'seed': args.seed, 'ratios': ratios.tolist(), 'files': file_report,
              'protocol': 'Random split stratified by attack category after cleaning and deduplication. '
                          'Does not establish temporal or unseen-attack generalization.',
              'invalid_value_policy': 'Drop rows containing missing, nonnumeric, or infinite features.',
              'label_counts_after_nonfinite_removal': counts(meta.attack_label)}
    # Remove the repeated header column only after checking its actual values.
    duplicate_header = 'Fwd Header Length__2'
    report['removed_identical_header_column'] = None
    if duplicate_header in X and 'Fwd Header Length' in X:
        if X[duplicate_header].equals(X['Fwd Header Length']):
            X = X.drop(columns=duplicate_header)
            report['removed_identical_header_column'] = duplicate_header
    print('Checking conflicting labels and repeated inputs ...', flush=True)
    # Hashes only identify candidate groups. Exact pandas row comparisons below
    # resolve conflicts and duplicates, so hash collisions cannot merge records.
    hashes = pd.util.hash_pandas_object(X, index=False)
    candidate = hashes.duplicated(keep=False)
    conflict = np.zeros(len(X), dtype=bool)
    candidate_indices = np.flatnonzero(candidate.to_numpy())
    if len(candidate_indices):
        subset = X.iloc[candidate_indices].copy()
        subset['_attack_label'] = meta.attack_label.iloc[candidate_indices].to_numpy()
        # Assign exact groups to candidate feature rows, with identical floats.
        group = subset.groupby(list(X.columns), sort=False, dropna=False).ngroup()
        label_counts = subset['_attack_label'].groupby(group).transform('nunique')
        conflict[candidate_indices] = label_counts.to_numpy() > 1
    report['quarantined_conflicting_by_label'] = counts(meta.attack_label[conflict])
    quarantine = meta.loc[conflict].copy()
    X = X.loc[~conflict].reset_index(drop=True)
    meta = meta.loc[~conflict].reset_index(drop=True)
    duplicate = X.duplicated(keep='first')
    report['removed_repeated_inputs_by_label'] = counts(meta.attack_label[duplicate])
    X = X.loc[~duplicate].reset_index(drop=True)
    meta = meta.loc[~duplicate].reset_index(drop=True)
    meta['y'] = (meta.attack_label != 'BENIGN').astype('int8')
    if meta.y.nunique() != 2:
        raise ValueError('Both BENIGN and attack examples must remain.')
    # Per-category allocation preserves rare attack types where at least 3 remain.
    rng = np.random.default_rng(args.seed)
    splits = [[], [], []]
    rare = []
    for label, group in meta.groupby('attack_label', sort=True):
        indices = rng.permutation(group.index.to_numpy())
        n = len(indices)
        if n < 3:
            allocation = np.array([n, 0, 0])
            rare.append(str(label))
        else:
            allocation = np.ones(3, dtype=int)
            ideal = (n - 3) * ratios
            allocation += np.floor(ideal).astype(int)
            remainder = n - allocation.sum()
            order = np.argsort(-(ideal - np.floor(ideal)), kind='stable')
            allocation[order[:remainder]] += 1
        cuts = np.cumsum(allocation)
        for destination, part in zip(splits, np.split(indices, cuts[:-1])):
            destination.extend(part.tolist())
    train, val, test = [rng.permutation(np.array(s, dtype=int)) for s in splits]
    if not len(val) or not len(test):
        raise ValueError('Not enough examples to create nonempty validation/test sets.')
    report['categories_with_fewer_than_3_examples_train_only'] = rare
    # Determine constant columns using training rows only.
    varying = X.iloc[train].nunique(dropna=False) > 1
    report['removed_training_constant_columns'] = list(X.columns[~varying])
    X = X.loc[:, varying]
    if not len(X.columns):
        raise ValueError('No varying features remain.')
    # Removing training-constant features can collapse previously distinct inputs.
    # Refuse output if this would place matching final inputs in different splits.
    final_hashes = pd.util.hash_pandas_object(X, index=False).to_numpy()
    hash_sets = [set(final_hashes[idx]) for idx in [train, val, test]]
    if any(hash_sets[a] & hash_sets[b] for a, b in [(0,1), (0,2), (1,2)]):
        raise ValueError('Final feature hashes overlap across splits after constant removal. '
                         'No split files written; review feature grouping before training.')
    print('Fitting scaler on training data only ...', flush=True)
    scaler = StandardScaler().fit(X.iloc[train].to_numpy())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not quarantine.empty:
        quarantine.to_csv(args.output_dir / 'quarantined_conflict_metadata.csv', index=False)
    report['feature_names'] = list(X.columns)
    report['splits'] = {}
    for name, idx in [('train', train), ('val', val), ('test', test)]:
        raw = X.iloc[idx].to_numpy()
        scaled = scaler.transform(raw)
        if not np.isfinite(scaled).all():
            raise ValueError('Scaling produced nonfinite values; inspect numeric ranges.')
        # Float64 preserves the numeric precision used for duplicate checks.
        np.save(args.output_dir / f'X_{name}.npy', raw)
        np.save(args.output_dir / f'X_{name}_scaled.npy', scaled)
        np.save(args.output_dir / f'y_{name}.npy', meta.y.iloc[idx].to_numpy())
        meta.iloc[idx].to_csv(args.output_dir / f'{name}_metadata.csv', index=False)
        report['splits'][name] = {'rows': len(idx), 'attack_labels': counts(meta.attack_label.iloc[idx]),
                                'binary_labels': counts(meta.y.iloc[idx])}
        print(f'{name}: {len(idx):,} rows, {len(X.columns)} features', flush=True)
    joblib.dump(scaler, args.output_dir / 'scaler.joblib')
    (args.output_dir / 'preprocessing_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Done. Outputs: {args.output_dir.resolve()}')
    print('Trees: X_train.npy; MLP/CNN: X_train_scaled.npy. Same rows and labels.')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Train a scikit-learn Random Forest on preprocessed CIC-IDS2017 arrays.

Install: python -m pip install numpy pandas scikit-learn joblib
Run from project root: python train_random_forest.py
Uses raw training/validation features. Never loads the test set.
Binary labels: 0 = BENIGN, 1 = attack. Runs on CPU.
"""
import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)


def load_split(directory, name):

    X = np.load(directory / f'X_{name}.npy', mmap_mode='r', allow_pickle=False)
    y = np.load(directory / f'y_{name}.npy', allow_pickle=False)

    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or len(y) == 0:
        raise ValueError(f'{name}: expected nonempty aligned 2D features and 1D labels.')
    
    if not np.array_equal(np.unique(y), [0, 1]):
        raise ValueError(f'{name}: both binary labels 0 and 1 must be present.')
    
    # sklearn trees use float32 internally; check that conversion is finite.
    for start in range(0, len(X), 100_000):
        with np.errstate(over='ignore', invalid='ignore'):
            batch = np.asarray(X[start:start + 100_000], dtype=np.float32)
        if not np.isfinite(batch).all():
            raise ValueError(f'{name}: features are nonfinite or exceed float32 range.')
    return X, y


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=Path('data/processed'))
    p.add_argument('--output-dir', type=Path, default=None)
    p.add_argument('--n-estimators', type=int, default=100)
    p.add_argument('--max-depth', type=int, default=None)
    p.add_argument('--min-samples-leaf', type=int, default=1)
    p.add_argument('--class-weight', choices=['none', 'balanced', 'balanced_subsample'], default='none')
    p.add_argument('--n-jobs', type=int, default=4, help='Parallel CPU workers; -1 uses all cores.')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    output = args.output_dir or Path('results') / ('random_forest_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))

    print('Loading training and validation arrays ...', flush=True)
    X_train, y_train = load_split(args.data_dir, 'train')
    X_val, y_val = load_split(args.data_dir, 'val')

    report_path = args.data_dir / 'preprocessing_report.json'
    preprocessing = json.loads(report_path.read_text(encoding='utf-8'))
    feature_names = preprocessing['feature_names']
    metadata = pd.read_csv(args.data_dir / 'val_metadata.csv')

    if args.n_estimators < 1 or args.min_samples_leaf < 1 or args.n_jobs == 0:
        p.error('Tree count and min-samples-leaf must be positive; n-jobs cannot be zero.')

    if args.max_depth is not None and args.max_depth < 1:
        p.error('max-depth must be positive.')
    
    if output.exists() and any(output.iterdir()):
        p.error('Output directory is not empty. Choose another --output-dir.')

    if X_train.shape[1] != X_val.shape[1]:
        raise ValueError('Training and validation feature counts differ.')
    
    if len(feature_names) != X_train.shape[1] or len(set(feature_names)) != len(feature_names):
        raise ValueError('Feature names do not match arrays.')
   
    if len(metadata) != len(y_val) or not np.array_equal(metadata['y'].to_numpy(), y_val):
        raise ValueError('Validation metadata does not align with validation labels.')
    
    if not np.array_equal((metadata.attack_label != 'BENIGN').astype(int).to_numpy(), y_val):
        raise ValueError('Attack names do not agree with binary labels.')


    
    print(f'Building Model...', flush=True)
    model = RandomForestClassifier(
        n_estimators=args.n_estimators, max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf, max_features='sqrt',
        class_weight=None if args.class_weight == 'none' else args.class_weight,
        random_state=args.seed, n_jobs=args.n_jobs, verbose=1)
    print(f'Training {args.n_estimators} trees on CPU ...', flush=True)
    start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - start

    print('Evaluating validation set ...', flush=True)
    start = time.perf_counter()
    positive_column = int(np.flatnonzero(model.classes_ == 1)[0])
    probability = model.predict_proba(X_val)[:, positive_column]
    
    
    # Matches sklearn's binary argmax tie handling: a probability of .5 is benign
    #I used Ai to generate some of these lines because Im lazy and didnt want to write them all out and it doesnt effect the model
    prediction = (probability > .5).astype(np.int8)
    predict_seconds = time.perf_counter() - start
    tn, fp, fn, tp = map(int, confusion_matrix(y_val, prediction, labels=[0, 1]).ravel())
    metrics = {
        'accuracy': float(accuracy_score(y_val, prediction)),
        'balanced_accuracy': float(balanced_accuracy_score(y_val, prediction)),
        'attack_precision': float(precision_score(y_val, prediction, zero_division=0)),
        'attack_recall': float(recall_score(y_val, prediction, zero_division=0)),
        'attack_f1': float(f1_score(y_val, prediction, zero_division=0)),
        'false_positive_rate': fp / (fp + tn),
        'roc_auc': float(roc_auc_score(y_val, probability)),
        'average_precision': float(average_precision_score(y_val, probability)),
        'always_benign_accuracy': float(np.mean(y_val == 0)),
        'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp},
        'classification_report': classification_report(y_val, prediction, labels=[0, 1],
            target_names=['BENIGN', 'ATTACK'], output_dict=True, zero_division=0),
    }
    
    
    category_rows = []
    for label, rows in metadata.groupby('attack_label', sort=True):
        idx = rows.index.to_numpy()
        detected = int(np.sum(prediction[idx] == 1))
        is_benign = label == 'BENIGN'
        category_rows.append({'attack_label': label, 'support': len(idx),
            'predicted_attack': detected,
            'attack_detection_recall': None if is_benign else detected / len(idx),
            'false_positive_rate': detected / len(idx) if is_benign else None})
    ### The code below was AI generated. It is for saving the model and evaluation metrics
    ### The model is already trained and evaluated
    ### It does not affect the training or evaluation of the model
    
    summary = {
        'model': 'RandomForestClassifier', 'evaluation_split': 'validation',
        'task': 'binary intrusion detection', 'label_mapping': {'0': 'BENIGN', '1': 'ATTACK'},
        'decision_rule': 'attack probability > 0.5; ties are benign',
        'data_dir': str(args.data_dir.resolve()),
        'train_rows': len(y_train), 'validation_rows': len(y_val),
        'feature_names': feature_names, 'parameters': model.get_params(),
        'fit_seconds': fit_seconds, 'validation_predict_seconds': predict_seconds,
        'versions': {'python': platform.python_version(), 'sklearn': sklearn.__version__,
                     'numpy': np.__version__, 'pandas': pd.__version__},
        'metrics': metrics,
        'note': 'Random within-dataset known-attack evaluation. Rare-category recall has low support.',
    }
    
    
    
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump({'model': model, 'feature_names': feature_names, 'label_mapping': summary['label_mapping'],
                 'sklearn_version': sklearn.__version__}, output / 'random_forest.joblib', compress=3)
    (output / 'validation_metrics.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    pd.DataFrame(category_rows).to_csv(output / 'validation_by_attack.csv', index=False)
    pd.DataFrame({'feature': feature_names, 'importance': model.feature_importances_}).sort_values(
        'importance', ascending=False).to_csv(output / 'feature_importance.csv', index=False)
    np.save(output / 'validation_attack_probability.npy', probability)
    np.save(output / 'validation_predictions.npy', prediction)
    for name in ['accuracy', 'attack_precision', 'attack_recall', 'attack_f1', 'false_positive_rate', 'average_precision']:
        print(f'{name}: {metrics[name]:.6f}')
    print(f'Confusion matrix: TN={tn:,}, FP={fp:,}, FN={fn:,}, TP={tp:,}')
    print(f'Training time: {fit_seconds:.1f} seconds')
    print(f'Results saved to {output.resolve()}', flush=True)


if __name__ == '__main__':
    main()

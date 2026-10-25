"""
run_xai_experiments.py — Paper 5: Multi-Level XAI for IoT IDS

Experiments:
  Exp 1: SHAP global + local + feature interactions (RF & CNN, TON-IoT + UNSW)
  Exp 2: LIME local explanations + stability (RF & CNN, TON-IoT + UNSW)
  Exp 3: Counterfactual explanations via DiCE (RF, TON-IoT)
  Exp 4: Consistency evaluation (SHAP vs LIME feature agreement, cross-model)

Run all:     python experiments/paper5_xai/run_xai_experiments.py
Run one:     python experiments/paper5_xai/run_xai_experiments.py --exp shap
             --exp shap | lime | counterfactual | consistency | all
"""

import sys, os, json, time, warnings, argparse
import numpy as np
import joblib
from pathlib import Path

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / 'experiments' / 'paper5_xai' / 'results'
FIGURES_DIR = ROOT / 'experiments' / 'paper5_xai' / 'figures'
MODELS_DIR  = ROOT / 'experiments' / 'baseline' / 'results' / 'models'
SEED = 42
np.random.seed(SEED)

DATASETS = {
    'TON_IoT':  {'n_feat': 25, 'n_cls': 10,
                 'classes': ['backdoor','ddos','dos','injection','mitm',
                             'normal','password','ransomware','scanning','xss']},
    'UNSW_NB15':{'n_feat': 16, 'n_cls': 10,
                 'classes': ['analysis','backdoor','dos','exploits','fuzzers',
                             'generic','normal','reconnaissance','shellcode','worms']},
}

CASE_STUDY_CLASSES = {
    'TON_IoT': ['dos', 'ddos', 'ransomware', 'injection'],
}


# ── Data loading ──────────────────────────────────────────────────────────────
def load_data(dataset_name):
    """Load preprocessed test data via the saved pipeline + baseline config.
    Uses the same seed/split as Paper 1 to ensure identical test sets."""
    from experiments.baseline.config import DATASETS
    from src.data.loader import load_dataset

    pipe_path = ROOT / 'experiments' / 'baseline' / 'results' / dataset_name / 'pipeline.pkl'
    pipe = joblib.load(pipe_path)

    csv_path, label_col = DATASETS[dataset_name]
    # load_dataset does its own split with seed=42 matching Paper 1
    raw = load_dataset(csv_path, label_col=label_col, seed=SEED)
    X_train_raw = raw['X_train']
    X_test_raw  = raw['X_test']
    y_train     = raw['y_train']
    y_test      = raw['y_test']
    label_encoder = raw['label_encoder']

    # Apply saved preprocessing pipeline (MI filter + RFECV + scaler)
    # transform() returns (X_out, y_out) tuple
    X_train_proc, _ = pipe.transform(X_train_raw)
    X_test_proc,  _ = pipe.transform(X_test_raw)
    feature_names = pipe.selected_features_

    return {
        'X_train': X_train_proc, 'y_train': y_train,
        'X_test':  X_test_proc,  'y_test':  y_test,
        'feature_names': feature_names,
        'label_encoder': label_encoder,
        'classes': label_encoder.classes_.tolist(),
    }


def load_model(dataset_name, model_type):
    """Load a saved model."""
    mdir = MODELS_DIR / dataset_name
    if model_type == 'rf':
        return joblib.load(mdir / 'model_rf.pkl')
    elif model_type == 'cnn':
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
        return tf.keras.models.load_model(mdir / 'model_cnn.h5')
    raise ValueError(f'Unknown model type: {model_type}')


def cnn_predict_proba(model, X):
    X_exp = np.expand_dims(X.astype(np.float32), -1)
    return model.predict(X_exp, verbose=0, batch_size=2048)


# ── Exp 1: SHAP ──────────────────────────────────────────────────────────────
def run_shap(dataset_name, data, args_exp):
    import shap
    print(f'\n[SHAP] {dataset_name}')
    out = RESULTS_DIR / dataset_name
    out.mkdir(parents=True, exist_ok=True)
    cache = out / 'shap_results.json'
    cache_rf = out / 'shap_rf_partial.json'  # intermediate cache after RF section
    if cache.exists():
        print(f'  Loading cached SHAP results...')
        return json.loads(cache.read_text())

    X_train = data['X_train']
    X_test  = data['X_test']
    y_test  = data['y_test']
    feat    = data['feature_names']
    classes = data['classes']
    results = {}

    # Load RF partial cache if available (avoids re-running 11-min TreeSHAP)
    if cache_rf.exists():
        partial = json.loads(cache_rf.read_text())
        if 'rf' in partial:
            print('  Loading cached RF SHAP partial result...')
            results['rf'] = partial['rf']

    # ── RF: TreeSHAP (exact) ────────────────────────────────────────────────
    if 'rf' not in results:
        print('  [RF] TreeSHAP global...')
        rf = load_model(dataset_name, 'rf')
        t0 = time.time()
        bg = shap.sample(X_train, 500, random_state=SEED)
        explainer_rf = shap.TreeExplainer(rf, bg, model_output='probability')

        # Global: mean |shap| over 1000 test samples
        X_eval = X_test[:1000]
        shap_vals_rf = explainer_rf.shap_values(X_eval)
        shap_time_rf = time.time() - t0

        # SHAP 0.46+ may return ndarray (n_samples, n_feat, n_cls) instead of list
        if isinstance(shap_vals_rf, np.ndarray) and shap_vals_rf.ndim == 3:
            # (n_samples, n_feat, n_cls) → list[n_cls] of (n_samples, n_feat)
            shap_vals_rf = [shap_vals_rf[:, :, i] for i in range(shap_vals_rf.shape[2])]
        elif isinstance(shap_vals_rf, np.ndarray) and shap_vals_rf.ndim == 2:
            shap_vals_rf = [shap_vals_rf]

        # Mean |shap| per feature (averaged across classes)
        mean_abs_rf = np.mean([np.abs(sv).mean(axis=0) for sv in shap_vals_rf], axis=0)
        global_rf = dict(zip(feat, mean_abs_rf.tolist()))

        # Per-class global importance
        per_class_rf = {}
        for ci, cls in enumerate(classes):
            mc = np.abs(shap_vals_rf[ci]).mean(axis=0)
            per_class_rf[cls] = dict(zip(feat, mc.tolist()))

        # Local SHAP: skipped — paper generator uses LIME local_explanations, not SHAP local
        local_rf = {}

        # Save partial cache NOW (before slow interactions) so CNN failure doesn't require re-running TreeSHAP
        results['rf'] = {
            'global_importance': global_rf,
            'per_class_importance': per_class_rf,
            'local_explanations': local_rf,
            'interaction_matrix': [],  # placeholder until interactions done
            'shap_time_1000': round(shap_time_rf, 2),
        }
        cache_rf.write_text(json.dumps(results, indent=2))
        print('  [RF] Global SHAP partial result saved.')

        # Feature interactions (RF, first 50 samples — reduced for speed)
        # tree_path_dependent avoids needing background data and supports interactions
        print('  [RF] TreeSHAP interactions (50 samples)...')
        t_int = time.time()
        explainer_rf_raw = shap.TreeExplainer(rf, feature_perturbation='tree_path_dependent')
        interact_vals = explainer_rf_raw.shap_interaction_values(X_eval[:50])
        # interact_vals shape: (n_samples, n_feat, n_feat, n_cls) in SHAP 0.46+
        #                   or (n_cls, n_samples, n_feat, n_feat) in older SHAP
        iv_arr = np.array(interact_vals) if isinstance(interact_vals, list) else interact_vals
        if iv_arr.ndim == 4:
            if iv_arr.shape[0] == len(classes):  # (n_cls, n_samples, n_feat, n_feat)
                mean_interact = np.abs(iv_arr).mean(axis=(0, 1))
            else:  # (n_samples, n_feat, n_feat, n_cls)
                mean_interact = np.abs(iv_arr).mean(axis=(0, 3))
        elif iv_arr.ndim == 3:  # (n_samples, n_feat, n_feat)
            mean_interact = np.abs(iv_arr).mean(axis=0)
        else:
            mean_interact = np.abs(iv_arr)
        np.fill_diagonal(mean_interact, 0)  # zero out self-interactions
        interact_time = time.time() - t_int

        results['rf']['interaction_matrix'] = mean_interact.tolist()
        results['rf']['interact_time_50'] = round(interact_time, 2)
        # Update partial cache with interactions included
        cache_rf.write_text(json.dumps(results, indent=2))
        print('  [RF] Interactions saved to partial cache.')
    else:
        print('  [RF] Loaded from partial cache — skipping TreeSHAP.')
        # If partial cache has empty interaction_matrix (interrupted), recompute interactions
        if not results['rf'].get('interaction_matrix'):
            print('  [RF] Recomputing interactions (partial cache had empty matrix)...')
            rf = load_model(dataset_name, 'rf')
            X_eval = X_test[:1000]
            t_int = time.time()
            explainer_rf_raw = shap.TreeExplainer(rf, feature_perturbation='tree_path_dependent')
            interact_vals = explainer_rf_raw.shap_interaction_values(X_eval[:50])
            iv_arr = np.array(interact_vals) if isinstance(interact_vals, list) else interact_vals
            if iv_arr.ndim == 4:
                if iv_arr.shape[0] == len(classes):
                    mean_interact = np.abs(iv_arr).mean(axis=(0, 1))
                else:
                    mean_interact = np.abs(iv_arr).mean(axis=(0, 3))
            elif iv_arr.ndim == 3:
                mean_interact = np.abs(iv_arr).mean(axis=0)
            else:
                mean_interact = np.abs(iv_arr)
            np.fill_diagonal(mean_interact, 0)
            results['rf']['interaction_matrix'] = mean_interact.tolist()
            results['rf']['interact_time_50'] = round(time.time() - t_int, 2)
            cache_rf.write_text(json.dumps(results, indent=2))
            print('  [RF] Interactions saved.')
        # Set X_eval for CNN section (loaded from cache path skips X_eval assignment)
        X_eval = X_test[:1000]

    # ── CNN: GradientExplainer ───────────────────────────────────────────────
    print('  [CNN] GradientExplainer global...')
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    cnn = load_model(dataset_name, 'cnn')

    t0 = time.time()
    bg_cnn = np.expand_dims(
        shap.sample(X_train, 200, random_state=SEED).astype(np.float32), -1)
    explainer_cnn = shap.GradientExplainer(cnn, bg_cnn)

    X_eval_cnn = np.expand_dims(X_eval.astype(np.float32), -1)
    shap_vals_cnn = explainer_cnn.shap_values(X_eval_cnn)
    shap_time_cnn = time.time() - t0

    # SHAP 0.46+ may return ndarray (n, n_feat, n_ch, n_cls) or (n, n_feat, n_cls)
    if isinstance(shap_vals_cnn, np.ndarray):
        if shap_vals_cnn.ndim == 4:  # (n_samples, n_feat, n_channels, n_cls)
            shap_vals_cnn = shap_vals_cnn[:, :, 0, :]  # drop channel dim → (n, n_feat, n_cls)
        if shap_vals_cnn.ndim == 3:  # (n_samples, n_feat, n_cls)
            shap_vals_cnn = [shap_vals_cnn[:, :, i] for i in range(shap_vals_cnn.shape[2])]
    elif isinstance(shap_vals_cnn, list):
        # Old SHAP: list[n_cls] of (n, n_feat, 1) — remove channel dim
        shap_vals_cnn = [sv[:, :, 0] if sv.ndim == 3 else sv for sv in shap_vals_cnn]

    mean_abs_cnn = np.mean([np.abs(sv).mean(axis=0) for sv in shap_vals_cnn], axis=0)
    global_cnn = dict(zip(feat, mean_abs_cnn.tolist()))

    per_class_cnn = {}
    for ci, cls in enumerate(classes):
        mc = np.abs(shap_vals_cnn[ci]).mean(axis=0)
        per_class_cnn[cls] = dict(zip(feat, mc.tolist()))

    # Local CNN SHAP: skipped — paper generator does not use SHAP local_explanations
    local_cnn = {}

    results['cnn'] = {
        'global_importance': global_cnn,
        'per_class_importance': per_class_cnn,
        'local_explanations': local_cnn,
        'shap_time_1000': round(shap_time_cnn, 2),
    }

    cache.write_text(json.dumps(results, indent=2))
    print(f'  → Saved: {cache.name}')
    return results


# ── Exp 2: LIME ──────────────────────────────────────────────────────────────
def run_lime(dataset_name, data):
    from lime.lime_tabular import LimeTabularExplainer
    print(f'\n[LIME] {dataset_name}')
    out = RESULTS_DIR / dataset_name
    out.mkdir(parents=True, exist_ok=True)
    cache = out / 'lime_results.json'
    if cache.exists():
        print(f'  Loading cached LIME results...')
        return json.loads(cache.read_text())

    X_train = data['X_train']
    X_test  = data['X_test']
    y_test  = data['y_test']
    feat    = data['feature_names']
    classes = data['classes']
    n_cls   = len(classes)

    explainer = LimeTabularExplainer(
        X_train, feature_names=feat, class_names=classes,
        discretize_continuous=False, random_state=SEED)

    rf  = load_model(dataset_name, 'rf')
    import tensorflow as tf; tf.config.set_visible_devices([], 'GPU')
    cnn = load_model(dataset_name, 'cnn')

    def rf_predict(X):
        return rf.predict_proba(X)

    def cnn_predict(X):
        return cnn_predict_proba(cnn, X)

    results = {'rf': {}, 'cnn': {}}

    # ── Local explanations for case-study classes (TON-IoT) + 5 random ───────
    study_classes = CASE_STUDY_CLASSES.get(dataset_name, [classes[0]])
    for model_name, pred_fn in [('rf', rf_predict), ('cnn', cnn_predict)]:
        print(f'  [{model_name.upper()}] local explanations...')
        local_expls = {}
        timing = []
        for cls in study_classes:
            ci = classes.index(cls)
            idx = np.where(y_test == ci)[0][:3]
            cls_expls = []
            for i in idx:
                t0 = time.time()
                exp = explainer.explain_instance(
                    X_test[i], pred_fn, num_features=10,
                    num_samples=1000, top_labels=1)
                elapsed = time.time() - t0
                timing.append(elapsed)
                label = exp.top_labels[0]
                feat_weights = dict(exp.as_list(label=label))
                # Clean feature names (LIME appends range strings)
                clean = {}
                for k, v in feat_weights.items():
                    base = k.split(' ')[0].split('<')[0].split('>')[0].strip()
                    # Match back to original feature
                    matched = next((f for f in feat if f in k), base)
                    clean[matched] = v
                cls_expls.append({
                    'instance_idx': int(i),
                    'true_class': cls,
                    'predicted_label': int(label),
                    'feature_weights': clean,
                    'elapsed_sec': round(elapsed, 3),
                    'surrogate_r2': round(float(exp.score), 4),
                })
            local_expls[cls] = cls_expls

        # ── Fidelity: mean surrogate R² per class ─────────────────────────
        fidelity_results = {}
        all_r2 = []
        for cls, expls in local_expls.items():
            r2s = [e['surrogate_r2'] for e in expls if 'surrogate_r2' in e]
            if r2s:
                fidelity_results[cls] = {
                    'mean_r2': round(float(np.mean(r2s)), 4),
                    'std_r2':  round(float(np.std(r2s)),  4),
                }
                all_r2.extend(r2s)

        # ── Stability: 20 runs on same instance, Jaccard@5 ────────────────
        print(f'  [{model_name.upper()}] stability (20 runs)...')
        stability_results = {}
        for cls in study_classes[:2]:  # 2 classes for speed
            ci = classes.index(cls)
            idx = np.where(y_test == ci)[0]
            if len(idx) == 0:
                continue
            inst = X_test[idx[0]]
            top5_sets = []
            for _ in range(20):
                exp = explainer.explain_instance(
                    inst, pred_fn, num_features=10,
                    num_samples=1000, top_labels=1)
                label = exp.top_labels[0]
                top5 = set(k.split(' ')[0] for k, _ in exp.as_list(label=label)[:5])
                top5_sets.append(top5)
            # Pairwise Jaccard
            jaccards = []
            for a in range(len(top5_sets)):
                for b in range(a+1, len(top5_sets)):
                    s1, s2 = top5_sets[a], top5_sets[b]
                    j = len(s1 & s2) / len(s1 | s2) if s1 | s2 else 1.0
                    jaccards.append(j)
            stability_results[cls] = {
                'mean_jaccard@5': round(float(np.mean(jaccards)), 4),
                'std_jaccard@5':  round(float(np.std(jaccards)), 4),
            }

        results[model_name] = {
            'local_explanations': local_expls,
            'stability': stability_results,
            'fidelity': {
                'per_class': fidelity_results,
                'mean_r2': round(float(np.mean(all_r2)), 4) if all_r2 else None,
            },
            'mean_elapsed_sec': round(float(np.mean(timing)), 3),
        }

    cache.write_text(json.dumps(results, indent=2))
    print(f'  → Saved: {cache.name}')
    return results


# ── Exp 3: Counterfactual (DiCE) ─────────────────────────────────────────────
def run_counterfactual(dataset_name, data):
    print(f'\n[Counterfactual/DiCE] {dataset_name}')
    out = RESULTS_DIR / dataset_name
    out.mkdir(parents=True, exist_ok=True)
    cache = out / 'counterfactual_results.json'
    if cache.exists():
        print(f'  Loading cached counterfactual results...')
        return json.loads(cache.read_text())

    import dice_ml
    import pandas as pd

    X_train = data['X_train']
    y_train = data['y_train']
    X_test  = data['X_test']
    y_test  = data['y_test']
    feat    = data['feature_names']
    classes = data['classes']

    rf = load_model(dataset_name, 'rf')

    # Build DiCE data object
    df_train = pd.DataFrame(X_train, columns=feat)
    df_train['label'] = y_train
    dice_data = dice_ml.Data(
        dataframe=df_train,
        continuous_features=feat,
        outcome_name='label')

    # Model wrapper
    def rf_predict_dice(df):
        return rf.predict(df[feat].values)

    dice_model = dice_ml.Model(model=rf, backend='sklearn')
    exp_dice = dice_ml.Dice(dice_data, dice_model, method='random')

    results = {}
    study_classes = CASE_STUDY_CLASSES.get(dataset_name, [classes[0]])
    # Target class: 'normal' if present, else class 0
    target_cls = 'normal' if 'normal' in classes else classes[0]
    target_ci = classes.index(target_cls)

    for cls in study_classes:
        ci = classes.index(cls)
        idx = np.where(y_test == ci)[0][:3]
        cls_cfs = []
        for i in idx:
            try:
                t0 = time.time()
                df_inst = pd.DataFrame([X_test[i]], columns=feat)
                cf = exp_dice.generate_counterfactuals(
                    df_inst, total_CFs=3,
                    desired_class=target_ci,
                    random_seed=SEED)
                elapsed = time.time() - t0
                cf_df = cf.cf_examples_list[0].final_cfs_df
                if cf_df is None or len(cf_df) == 0:
                    continue
                # Compute changes
                changes = []
                for _, row in cf_df.iterrows():
                    diff = {}
                    for f in feat:
                        orig = float(X_test[i][feat.index(f)])
                        new  = float(row[f])
                        if abs(orig - new) > 1e-6:
                            diff[f] = {'original': round(orig, 4),
                                       'counterfactual': round(new, 4),
                                       'delta': round(new - orig, 4)}
                    changes.append(diff)
                cls_cfs.append({
                    'instance_idx': int(i),
                    'true_class': cls,
                    'counterfactuals': changes,
                    'n_features_changed': [len(c) for c in changes],
                    'elapsed_sec': round(elapsed, 3),
                })
            except Exception as e:
                cls_cfs.append({'instance_idx': int(i), 'error': str(e)})
        results[cls] = cls_cfs

    # Summary stats
    all_n_changed = []
    for cls_cfs in results.values():
        for cf in cls_cfs:
            all_n_changed.extend(cf.get('n_features_changed', []))

    summary = {
        'mean_features_changed': round(float(np.mean(all_n_changed)), 2) if all_n_changed else None,
        'std_features_changed': round(float(np.std(all_n_changed)), 2) if all_n_changed else None,
        'min_features_changed': int(np.min(all_n_changed)) if all_n_changed else None,
        'max_features_changed': int(np.max(all_n_changed)) if all_n_changed else None,
    }
    output = {'by_class': results, 'summary': summary}
    cache.write_text(json.dumps(output, indent=2))
    print(f'  → Saved: {cache.name}  (mean Δfeatures: {summary["mean_features_changed"]})')
    return output


# ── Exp 4: Consistency ────────────────────────────────────────────────────────
def run_consistency(dataset_name, shap_res, lime_res, data):
    print(f'\n[Consistency] {dataset_name}')
    out = RESULTS_DIR / dataset_name
    out.mkdir(parents=True, exist_ok=True)
    cache = out / 'consistency_results.json'
    if cache.exists():
        print(f'  Loading cached consistency results...')
        return json.loads(cache.read_text())

    feat    = data['feature_names']
    classes = data['classes']
    results = {}

    for model_name in ['rf', 'cnn']:
        shap_global = shap_res.get(model_name, {}).get('global_importance', {})
        lime_local  = lime_res.get(model_name, {}).get('local_explanations', {})

        # Top-K feature sets from global SHAP
        # Flatten any nested lists (CNN SHAP may store list values due to shape mismatch)
        shap_global = {
            k: float(np.mean(v)) if isinstance(v, list) else float(v)
            for k, v in shap_global.items()
        }
        sorted_shap = sorted(shap_global.items(), key=lambda x: -x[1])
        shap_top5  = set(f for f, _ in sorted_shap[:5])
        shap_top10 = set(f for f, _ in sorted_shap[:10])

        # Top-K from LIME (aggregate across all local instances)
        lime_feat_scores = {}
        for cls_expls in lime_local.values():
            for expl in cls_expls:
                for f, w in expl.get('feature_weights', {}).items():
                    lime_feat_scores[f] = lime_feat_scores.get(f, 0) + abs(w)
        sorted_lime = sorted(lime_feat_scores.items(), key=lambda x: -x[1])
        lime_top5  = set(f for f, _ in sorted_lime[:5])
        lime_top10 = set(f for f, _ in sorted_lime[:10])

        def jaccard(a, b):
            return len(a & b) / len(a | b) if a | b else 1.0

        results[model_name] = {
            'shap_top5':   sorted([f for f, _ in sorted_shap[:5]]),
            'shap_top10':  sorted([f for f, _ in sorted_shap[:10]]),
            'lime_top5':   sorted([f for f, _ in sorted_lime[:5]]),
            'lime_top10':  sorted([f for f, _ in sorted_lime[:10]]),
            'jaccard@5':   round(jaccard(shap_top5, lime_top5), 4),
            'jaccard@10':  round(jaccard(shap_top10, lime_top10), 4),
            'overlap@5':   sorted(shap_top5 & lime_top5),
            'overlap@10':  sorted(shap_top10 & lime_top10),
        }

    # Cross-model SHAP agreement (RF vs CNN)
    def _flat_gi(gi):
        return {k: float(np.mean(v)) if isinstance(v, list) else float(v) for k, v in gi.items()}
    rf_shap  = sorted(_flat_gi(shap_res.get('rf',  {}).get('global_importance', {})).items(), key=lambda x:-x[1])
    cnn_shap = sorted(_flat_gi(shap_res.get('cnn', {}).get('global_importance', {})).items(), key=lambda x:-x[1])
    rf_top5   = set(f for f, _ in rf_shap[:5])
    cnn_top5  = set(f for f, _ in cnn_shap[:5])
    rf_top10  = set(f for f, _ in rf_shap[:10])
    cnn_top10 = set(f for f, _ in cnn_shap[:10])

    results['cross_model_shap'] = {
        'rf_top10':   [f for f, _ in rf_shap[:10]],
        'cnn_top10':  [f for f, _ in cnn_shap[:10]],
        'jaccard@5':  round(len(rf_top5 & cnn_top5) / len(rf_top5 | cnn_top5), 4),
        'jaccard@10': round(len(rf_top10 & cnn_top10) / len(rf_top10 | cnn_top10), 4),
        'overlap@5':  sorted(rf_top5 & cnn_top5),
        'overlap@10': sorted(rf_top10 & cnn_top10),
    }

    # Fidelity: LIME surrogate accuracy on perturbed neighborhood
    # Already computed implicitly by LIME (R² of local surrogate) — approximate via top-5 overlap
    # as a proxy (not directly exposed by lime package)
    results['stability_summary'] = {
        m: lime_res.get(m, {}).get('stability', {})
        for m in ['rf', 'cnn']
    }
    results['lime_timing'] = {
        m: lime_res.get(m, {}).get('mean_elapsed_sec') for m in ['rf', 'cnn']
    }

    cache.write_text(json.dumps(results, indent=2))
    print(f'  → Saved: {cache.name}')
    return results


# ── Figure generation ─────────────────────────────────────────────────────────
def generate_figures(dataset_name, shap_res, lime_res, consistency_res, data):
    print(f'\n[Figures] {dataset_name}')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    feat    = data['feature_names']
    classes = data['classes']
    FIGS    = FIGURES_DIR / dataset_name
    FIGS.mkdir(parents=True, exist_ok=True)

    # ── Fig 1: Global SHAP — RF vs CNN side by side ──────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, model_name, color in zip(axes, ['rf', 'cnn'], ['#2196F3', '#FF5722']):
        gi_raw = shap_res.get(model_name, {}).get('global_importance', {})
        gi = {k: float(np.mean(v)) if isinstance(v, list) else float(v) for k, v in gi_raw.items()}
        sorted_feat = sorted(gi.items(), key=lambda x: -x[1])[:12]
        names = [f for f, _ in sorted_feat]
        vals  = [v for _, v in sorted_feat]
        bars = ax.barh(range(len(names)), vals, color=color, alpha=0.85, edgecolor='white')
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('Mean |SHAP value|', fontsize=10)
        ax.set_title(f'Global SHAP — {model_name.upper()}', fontsize=11, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
    plt.suptitle(f'Global Feature Importance via SHAP — {dataset_name}',
                 fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(FIGS / 'fig_shap_global_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: fig_shap_global_comparison.png')

    # ── Fig 2: Feature Interaction Heatmap (RF) ───────────────────────────────
    interact = np.array(shap_res.get('rf', {}).get('interaction_matrix', []))
    if interact.size > 0:
        n = interact.shape[0]
        fig, ax = plt.subplots(figsize=(10, 8))
        # Zero diagonal (self-interactions dominate)
        np.fill_diagonal(interact, 0)
        im = ax.imshow(interact, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(range(n)); ax.set_xticklabels(feat, rotation=90, fontsize=7)
        ax.set_yticks(range(n)); ax.set_yticklabels(feat, fontsize=7)
        plt.colorbar(im, ax=ax, label='Mean |SHAP interaction value|')
        ax.set_title(f'RF Feature Interaction Matrix — {dataset_name}',
                     fontsize=11, fontweight='bold')
        plt.tight_layout()
        plt.savefig(FIGS / 'fig_shap_interactions.png', dpi=150, bbox_inches='tight')
        plt.close()
        print('  Saved: fig_shap_interactions.png')

    # ── Fig 3: LIME local explanation (one instance per case-study class) ─────
    study_classes = CASE_STUDY_CLASSES.get(dataset_name, [classes[0]])
    lime_rf = lime_res.get('rf', {}).get('local_explanations', {})
    if lime_rf and study_classes:
        fig, axes = plt.subplots(1, min(4, len(study_classes)), figsize=(16, 5))
        if len(study_classes) == 1:
            axes = [axes]
        colors_pos = '#4CAF50'; colors_neg = '#F44336'
        for ax, cls in zip(axes, study_classes[:4]):
            expls = lime_rf.get(cls, [])
            if not expls:
                ax.set_visible(False)
                continue
            fw = expls[0].get('feature_weights', {})
            sorted_fw = sorted(fw.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
            names = [f for f, _ in sorted_fw]
            vals  = [v for _, v in sorted_fw]
            colors = [colors_pos if v > 0 else colors_neg for v in vals]
            ax.barh(range(len(names)), vals, color=colors, alpha=0.85, edgecolor='white')
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=8)
            ax.invert_yaxis()
            ax.axvline(0, color='black', linewidth=0.8)
            ax.set_title(f'{cls.upper()}', fontsize=10, fontweight='bold')
            ax.set_xlabel('LIME weight', fontsize=8)
        pos_patch = mpatches.Patch(color=colors_pos, label='Supports class')
        neg_patch = mpatches.Patch(color=colors_neg, label='Contradicts class')
        fig.legend(handles=[pos_patch, neg_patch], loc='lower center',
                   ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.05))
        plt.suptitle(f'LIME Local Explanations (RF) — {dataset_name}',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(FIGS / 'fig_lime_local.png', dpi=150, bbox_inches='tight')
        plt.close()
        print('  Saved: fig_lime_local.png')

    # ── Fig 4: Consistency — SHAP vs LIME top-10 overlap bar chart ────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    k_vals = [5, 10]
    for ax, model_name in zip(axes, ['rf', 'cnn']):
        cr = consistency_res.get(model_name, {})
        j5  = cr.get('jaccard@5', 0)
        j10 = cr.get('jaccard@10', 0)
        cm_j5  = consistency_res.get('cross_model_shap', {}).get('jaccard@5', 0)
        cm_j10 = consistency_res.get('cross_model_shap', {}).get('jaccard@10', 0)
        bars = ax.bar(['SHAP vs LIME\n@5', 'SHAP vs LIME\n@10',
                       'RF vs CNN\nSHAP@5', 'RF vs CNN\nSHAP@10'],
                      [j5, j10, cm_j5, cm_j10],
                      color=['#2196F3','#1565C0','#FF5722','#BF360C'], alpha=0.85)
        for bar, val in zip(bars, [j5, j10, cm_j5, cm_j10]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Jaccard Similarity', fontsize=10)
        ax.set_title(f'{model_name.upper()} Explanation Consistency', fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    plt.suptitle(f'Explanation Consistency — {dataset_name}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGS / 'fig_consistency.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: fig_consistency.png')

    # ── Fig 5: Stability box plot (LIME, 20 runs) ─────────────────────────────
    stab = consistency_res.get('stability_summary', {})
    stab_data = {}
    for model_name in ['rf', 'cnn']:
        for cls, v in stab.get(model_name, {}).items():
            key = f'{model_name.upper()}\n{cls}'
            stab_data[key] = v.get('mean_jaccard@5', 0)
    if stab_data:
        fig, ax = plt.subplots(figsize=(8, 4))
        keys = list(stab_data.keys())
        vals = [stab_data[k] for k in keys]
        bars = ax.bar(keys, vals, color=['#2196F3','#2196F3','#FF5722','#FF5722'], alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Mean Jaccard@5 (20 runs)', fontsize=10)
        ax.set_title(f'LIME Explanation Stability — {dataset_name}',
                     fontsize=11, fontweight='bold')
        ax.axhline(0.8, color='green', linestyle='--', alpha=0.6, label='Good stability threshold')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIGS / 'fig_lime_stability.png', dpi=150, bbox_inches='tight')
        plt.close()
        print('  Saved: fig_lime_stability.png')

    # ── Fig 6: Per-class SHAP heatmap ────────────────────────────────────────
    for model_name in ['rf', 'cnn']:
        pci = shap_res.get(model_name, {}).get('per_class_importance', {})
        if not pci:
            continue
        cls_list = list(pci.keys())
        # Top 10 globally important features
        gi_raw = shap_res[model_name]['global_importance']
        gi = {k: float(np.mean(v)) if isinstance(v, list) else float(v) for k, v in gi_raw.items()}
        top_feats = [f for f, _ in sorted(gi.items(), key=lambda x: -x[1])[:10]]
        matrix = np.array([[pci[c].get(f, 0) for f in top_feats] for c in cls_list])
        fig, ax = plt.subplots(figsize=(12, 5))
        im = ax.imshow(matrix, cmap='Blues', aspect='auto')
        ax.set_xticks(range(len(top_feats)))
        ax.set_xticklabels(top_feats, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(cls_list)))
        ax.set_yticklabels(cls_list, fontsize=9)
        plt.colorbar(im, ax=ax, label='Mean |SHAP value|')
        ax.set_title(f'Per-Class SHAP Importance ({model_name.upper()}) — {dataset_name}',
                     fontsize=11, fontweight='bold')
        plt.tight_layout()
        plt.savefig(FIGS / f'fig_shap_perclass_{model_name}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Saved: fig_shap_perclass_{model_name}.png')


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', default='all',
                        help='shap | lime | counterfactual | consistency | figures | all')
    parser.add_argument('--dataset', default='all', help='TON_IoT | UNSW_NB15 | all')
    args = parser.parse_args()

    def should_run(name):
        return args.exp in ('all', name)

    datasets_to_run = (
        ['TON_IoT', 'UNSW_NB15'] if args.dataset == 'all'
        else [args.dataset])

    all_shap = {}; all_lime = {}; all_consistency = {}

    for ds in datasets_to_run:
        print(f'\n{"="*60}\n  Dataset: {ds}\n{"="*60}')
        data = load_data(ds)
        print(f'  X_train: {data["X_train"].shape}  X_test: {data["X_test"].shape}')
        print(f'  Features: {data["feature_names"]}')
        print(f'  Classes: {data["classes"]}')

        shap_res  = run_shap(ds, data, args.exp) if should_run('shap') else \
            json.loads((RESULTS_DIR / ds / 'shap_results.json').read_text()) \
            if (RESULTS_DIR / ds / 'shap_results.json').exists() else {}

        lime_res  = run_lime(ds, data) if should_run('lime') else \
            json.loads((RESULTS_DIR / ds / 'lime_results.json').read_text()) \
            if (RESULTS_DIR / ds / 'lime_results.json').exists() else {}

        if should_run('counterfactual') and ds == 'TON_IoT':
            run_counterfactual(ds, data)

        consistency_res = run_consistency(ds, shap_res, lime_res, data) \
            if (should_run('consistency') and shap_res and lime_res) else \
            json.loads((RESULTS_DIR / ds / 'consistency_results.json').read_text()) \
            if (RESULTS_DIR / ds / 'consistency_results.json').exists() else {}

        if should_run('figures') or should_run('all'):
            if shap_res and lime_res and consistency_res:
                generate_figures(ds, shap_res, lime_res, consistency_res, data)

        all_shap[ds] = shap_res
        all_lime[ds] = lime_res
        all_consistency[ds] = consistency_res

    print('\n[DONE] All XAI experiments complete.')

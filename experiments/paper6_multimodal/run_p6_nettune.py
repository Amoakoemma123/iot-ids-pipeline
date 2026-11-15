"""Final review round 2, Paper 6, referee M3: a small symmetric hyperparameter grid for the CNN early and late fusion
models, chosen on validation macro-F1 like the trees (RF/GBDT grids in run_p6_v2.tune_trees).

Grid: learning rate {1e-3, 3e-4} x dropout {0.1, 0.3} x encoder width {1, 2} = 8 configurations (the paper's networks
are lr 1e-3, dropout 0.3, width 1). Each configuration is trained on seed 0 (semantic split, split 0) and scored on the
validation split; the best configuration per (dataset, model) is then trained on seeds 1-4. Test scores are saved
for every run but are not used for the choice.

Outputs: results_v4/<ds>/nettune/<arch>_<cfg>/seed0.json (+ _pred.npy) for the grid, results_v4/<ds>/<arch>_tuned/
seed<k>.json (+ _pred.npy) for the five seeds of the chosen configuration, results_v4/<ds>/tune_net.json (summary).
Resumable. Run:  python -m experiments.paper6_multimodal.run_p6_nettune --dataset TON_IoT [--smoke]
"""
import argparse, gc, json, shutil, time
import numpy as np
from experiments.paper6_multimodal import run_p6_v2 as R

GRID = [dict(lr=lr, dropout=do, width=w) for lr in (1e-3, 3e-4) for do in (0.1, 0.3) for w in (1.0, 2.0)]
ARCHS = ('early', 'late')


def tag(g):
    return f"lr{g['lr']:.0e}_do{g['dropout']}_w{g['width']:g}"


def fit(D, arch, s1, s2, seed, g, path):
    if path.exists():
        return json.loads(path.read_text())
    R.LR, R.DROPOUT, R.WMULT = g['lr'], g['dropout'], g['width']
    t0 = time.time()
    res, pred = R.run_cnn(D, arch, s1, s2, seed)
    res.update(cfg=dict(arch=arch, seed=seed, split=0, epochs_budget=R.EPOCHS, grid=g, exp=path.parent.name))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path.with_name(f'seed{seed}_pred.npy'), pred.astype(np.int8))
    path.write_text(json.dumps(res, indent=1))
    print(f"  {path.parent.name} seed{seed}: val macro-F1 {res['val_macro_f1']:.4f}  test {res['macro_f1']:.4f} "
          f"({time.time() - t0:.0f}s, best epoch {res['best_epoch']})", flush=True)
    gc.collect()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=('TON_IoT', 'UNSW_NB15'))
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    out = R.OUT
    grid, seeds = GRID, R.SEEDS
    if a.smoke:
        out = R.OUT.parent / 'results_v4_smoke'
        R.EPOCHS, grid, seeds = 2, GRID[:2], (0, 1)
    import tensorflow as tf
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    ds = a.dataset
    D = R.load(ds, 0)
    s1, s2 = R.semantic_split(D)
    print(f'== {ds}: network grid, {len(grid)} configurations x {ARCHS}', flush=True)
    summary = {}
    for arch in ARCHS:
        scores = []
        for g in grid:
            r = fit(D, arch, s1, s2, 0, g, out / ds / 'nettune' / f'{arch}_{tag(g)}' / 'seed0.json')
            scores.append(dict(grid=g, val_macro_f1=r['val_macro_f1'], test_macro_f1=r['macro_f1'], best_epoch=r['best_epoch']))
        best = max(scores, key=lambda z: z['val_macro_f1'])['grid']
        dst = out / ds / f'{arch}_tuned'
        dst.mkdir(parents=True, exist_ok=True)
        src = out / ds / 'nettune' / f'{arch}_{tag(best)}'
        for f in ('seed0.json', 'seed0_pred.npy'):
            if not (dst / f).exists():
                shutil.copy(src / f, dst / f)
        for seed in seeds[1:]:
            fit(D, arch, s1, s2, seed, best, dst / f'seed{seed}.json')
        summary[arch] = dict(grid=scores, best=best, default=dict(lr=1e-3, dropout=0.3, width=1.0))
        print(f'  {arch}: chose {tag(best)}', flush=True)
    (out / ds / 'tune_net.json').write_text(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()

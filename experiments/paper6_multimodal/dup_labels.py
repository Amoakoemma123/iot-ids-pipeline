"""P6 final review round 2, minor 2: label agreement of the test records that match a (pre-SMOTE) training record on
the selected features. No training. Writes results_v4/<ds>/dup_labels.json.
Run:  python -m experiments.paper6_multimodal.dup_labels"""
import collections, json
import numpy as np
from experiments.paper6_multimodal import run_p6_v2 as R

for ds in ('UNSW_NB15', 'TON_IoT'):
    D = R.load(ds, 0)
    lab = collections.defaultdict(collections.Counter)
    for r, y in zip(np.ascontiguousarray(D['Xtr']), D['ytr']):
        lab[r.tobytes()][int(y)] += 1
    dup = ~D['clean']
    any_other = all_other = maj_other = 0
    for r, y in zip(np.ascontiguousarray(D['Xte'])[dup], D['yte'][dup]):
        c, y = lab[r.tobytes()], int(y)
        any_other += any(k != y for k in c)          # at least one matching training record has another label
        all_other += y not in c                       # no matching training record has the test label
        maj_other += c.most_common(1)[0][0] != y      # the majority label of the matches differs
    out = dict(n_matched=int(dup.sum()), any_other_label=any_other, all_other_label=all_other, majority_other_label=maj_other)
    print(ds, out, flush=True)
    (R.OUT / ds / 'dup_labels.json').write_text(json.dumps(out, indent=1))

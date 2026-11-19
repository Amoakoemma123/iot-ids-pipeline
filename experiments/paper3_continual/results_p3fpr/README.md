Recall at a fixed benign false-positive rate (Paper 3, Section 5.3). Produced by run_p3_fixed_fpr.py.
fixed_fpr.json: per-stage summaries and per-run operating points. <dataset>/runs/*.json: run records (confusion
matrices, checked against the stored main runs). The per-record score files (*.npz, about 58 MB) are not included;
`--phase cnn` / `--phase rf` regenerate them, and `--phase analyze` rebuilds fixed_fpr.json.

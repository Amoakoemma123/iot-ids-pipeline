Deployment benchmark (Paper 6, Section 5.6). Produced by run_p6_deploy_bench.py: tree ensembles refitted with the
paper's settings, exported to ONNX (skl2onnx) and timed with ONNX Runtime on one thread; networks timed on one thread.
deploy_bench.json holds all measurements. The ONNX files themselves (about 590 MB) are not included; the script
regenerates them.

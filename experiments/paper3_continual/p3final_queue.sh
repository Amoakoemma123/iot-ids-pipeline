#!/bin/bash
# Paper 3 final review round (2026-11-07): corrected A-GEM, replay-weighting ablation, random forest on the replay
# memory, and the range-share diagnostic. Code: experiments/paper3_continual/run_cl_v2.py (groups agemfix, replayabl,
# rfmem, rangeshare). Results: experiments/paper3_continual/results_p3final/<dataset>/<group>/ (results_v3 untouched).
#
# Resumable: every run writes one JSON and is skipped when that file exists, so re-launching the script continues
# where it stopped. Stages run one after another, never in parallel.
# Launch only after the other long experiment has finished (it may then use all cores and the GPU):
#   nohup bash logs/p3final_queue.sh > /dev/null 2>&1 &
# End marker: "P3FINAL DONE" in logs/p3final_queue.log. Per-stage logs: logs/p3final_<group>_<dataset>.log
#
# Determinism / devices follow the runs these are paired with (results_v3 main/orders):
#   TON-IoT and UNSW-NB15 on the CPU (P3_CPU=1, as in p3v3_cpu_queue.sh), N-BaIoT on the GPU (as in p3v3_gpu_queue.sh);
#   op determinism is enabled inside run_cl_v2.py; Task 0 is reused from cache_v2/task0 (same cache keys), so every new
#   run starts from the same Task-0 model as the stored runs of that seed. The random forest runs on the CPU.
#   If N-BaIoT finds no GPU it still runs (on the CPU); the device list is printed at the top of its log; then the
#   paired comparison with results_v3 also contains a CPU/GPU numerical difference (say so in the text).
#
# Estimates (8 cores, 16 GB RAM, GTX 1650; from the wall times of the matching results_v3 runs):
#   S0 rangeshare x3 (no training, data load + transform)      ~5 min,    peak RAM ~3 GB
#   S1 agemfix TON  (5 seeds main + 4 orders, CPU)              ~15-25 min, ~3 GB
#   S2 agemfix UNSW (5 + 4, CPU)                                ~15-25 min, ~3 GB
#   S3 agemfix NB   (5 + 4, GPU)                                ~20-30 min, ~3 GB RAM + ~2 GB VRAM
#   S4 replayabl TON  (ER-2000-pool, 5 seeds, CPU)              ~15-25 min, ~3 GB
#   S5 replayabl UNSW (5 seeds, CPU)                            ~15-25 min, ~3 GB
#   S6 replayabl NB   (5 seeds, GPU)                            ~10-15 min, ~3 GB + ~2 GB VRAM
#   S7 rfmem x3 (RF on memory + RF joint, 5 seeds x 3 stages)   ~10-15 min, ~3-4 GB (200 trees, all cores)
#   Total ~2-3 h (A-GEM, the critical part, first: ~1-1.3 h).
cd "/e/Research_Works/IDS_IOT_Research/Papers pipeline" || exit 1
PY="/e/Research_Works/IDS_IOT_Research/ids_env/Scripts/python.exe"
L=logs/p3final_queue.log
export TF_CPP_MIN_LOG_LEVEL=2
echo "P3FINAL START $(date)" >> $L

run() {   # run <group> <dataset> <cpu|gpu>
  if [ "$3" = cpu ]; then export P3_CPU=1; else unset P3_CPU; fi
  $PY -m experiments.paper3_continual.run_cl_v2 --dataset $2 --group $1 >> logs/p3final_$1_$2.log 2>&1
  echo "P3FINAL $1 $2 ($3) exit $? $(date)" >> $L
}

# S0 range share (no training; CPU)
for d in TON_IoT UNSW_NB15 N_BaIoT; do run rangeshare $d cpu; done
# S1-S3 corrected A-GEM (critical: the A-GEM conclusions wait for these)
run agemfix TON_IoT cpu
run agemfix UNSW_NB15 cpu
run agemfix N_BaIoT gpu
# S4-S6 replay-weighting ablation (ER-2000-pool)
run replayabl TON_IoT cpu
run replayabl UNSW_NB15 cpu
run replayabl N_BaIoT gpu
# S7 random forest retrained on the replay memory, and the all-data forest after every task
for d in TON_IoT UNSW_NB15 N_BaIoT; do run rfmem $d cpu; done

# completeness check: 9 agem2 + 5 er2000p + 10 rf files per dataset, 3 rangeshare files
n=$(find experiments/paper3_continual/results_p3final -name '*.json' | wc -l)
echo "P3FINAL result files: $n (expected 75) $(date)" >> $L
echo "P3FINAL DONE $(date)" >> $L

# A Generic Framework for Side-Channel Attacks against LWE-based Cryptosystems

Artifacts for the IACR publication in EUROCRYPT 2025, see [https://ia.cr/2024/1211](https://ia.cr/2024/1211) for an extended version.

## Note
Note that this repository does not contain the final cleaned-up version of the solver but the version we used during the evaluation.
It still contains unfinished features and unfinished/failed attempts at improvements.
Some performance improvements relevant for sparser hints have also been implemented after the evaluations.
For the latest version, see [https://github.com/juliusjh/distribution_hints_solvers](https://github.com/juliusjh/distribution_hints_solvers).

# Reproduction Instructions

## Requirements:
- Rust
- Python >= 3.9, <=3.10
- maturin (tested with 1.5.1 and 1.8.2; some earlier versions give no warnings)
- venv
- tmux (if ``start_runs.sh`` or ``start_runs_all_threads.sh`` is being used)

## Setup environment:
- Install rust and maturin (see [https://www.rust-lang.org/tools/install](https://www.rust-lang.org/tools/install) and [https://www.maturin.rs/installation.html](https://www.maturin.rs/installation.html))
- Make sure a python version >= 3.9 is installed
- Use setup.sh to create a virtual environment (under ``.env/``), install required python packages, and compile the rust files
- Note that you need to source ``setup.sh``, i.e., ``source setup.sh``
- Note that the size of the message (SZ_MSG) needs to be adapted in the rust code, which needs to be recompiled after changing it.
    
## Calls to reproduce Figures
The following calls require tmux (see [https://github.com/tmux/tmux/wiki](https://github.com/tmux/tmux/wiki)). After calling them, use ``tmux attach`` to see the output of the calls. Results will be under ``results/``.
The plot script will generate the coordinates in a pgfplots compatible format, as well as simple coordinates based on the result json.
For general usage see below.

### Fig. 3:

#### a)

- run: ``./start_runs_all_threads.sh perfect_hint.json hints.json 20 perfect_hints``
- plot: ``python scripts/plot.py -y bikz_avg -x p -p solver nfac p --use-all-jsons --sorted results/perfect_hint_hints``

#### b)

- run: ``./start_runs_all_threads.sh perfect_hint_bino.json hints.json 20 perfect_hints_bino_gr``
- plot: ``python scripts/plot.py -y bikz_avg -x p -p solver nfac p  --use-all-jsons --sorted results/perfect_hint_bino_hints``

and 

- run: ``./start_runs_all_threads.sh perfect_hint_bino.json bino_hints.json 10 perfect_hints_bino_bp``
- plot: ``python scripts/plot.py -y bikz_avg -x p -p solver nfac p  --use-all-jsons --sorted results/perfect_hint_bino_bino_hints``

#### c)

- run: ``./start_runs_all_threads.sh approximate_hint.json approximate_hint_greedy.json 10 approximate_hint_gr``
- plot: ``python scripts/plot.py -y bikz_avg -x sigma -p solver nfac sigma  --use-all-jsons --sorted results/approximate_hint_approximate_hint_greedy``
#### d)

- run: ``./start_runs_all_threads.sh approximate_hint_bino.json approximate_hint.json 16 approximate_hint_bino``
- plot: ``python scripts/plot.py -y bikz_avg -x sigma -p solver nfac sigma  --use-all-jsons --sorted results/approximate_hint_bino_approximate_hint``
### Fig. 4:

- run: ``./start_runs_all_threads.sh approximate_hint.json approximate_hint_single_greedy.json 40 approximate_hint_single_greedy`` 
- plot: ``python scripts/plot.py -y bikz_avg bikz_min bikz_max -x nfac -p solver nfac sigma  --use-all-jsons --sorted results/approximate_hint_approximate_hint_single_greedy``

and 

- run: ``./start_runs_all_threads.sh approximate_hint_bino.json approximate_hint_single.json 40 approximate_hint_single_bino``
- plot: ``python scripts/plot.py -y bikz_avg bikz_min bikz_max -x nfac -p solver nfac sigma  --use-all-jsons --sorted results/approximate_hint_bino_approximate_hint_single``

### Fig. 6:

- run: ``./start_runs_all_threads.sh nt_leakage_value.json nt_vl_bp.json 15 nt_vl_bp``
- plot: ``python scripts/plot.py -y bikz_avg bikz_min bikz_max -x nfac -p solver nfac sigma  --use-all-jsons --sorted results/nt_leakage_value_nt_vl_bp``

and 

- run: ``./start_runs_all_threads.sh nt_leakage_value.json nt_vl_greedy.json 15 nt_vl_greedy``
- plot: ``python scripts/plot.py -y bikz_avg bikz_min bikz_max -x nfac -p solver nfac sigma  --use-all-jsons --sorted results/nt_leakage_value_nt_vl_greedy``

### Fig. 7:

- run: ``./start_runs_all_threads.sh nt_leakage_hw.json nt_hw_bp.json 18 nt_hw_bp`` 
- plot: ``python scripts/plot.py -y bikz_avg bikz_min bikz_max -x sigma -p solver sigma nfac  --use-all-jsons --sorted results/nt_leakage_hw_nt_hw_bp``

and 

- run: ``./start_runs_all_threads.sh nt_leakage_hw.json nt_hw_greedy.json 18 nt_hw_gr``
- plot: ``python scripts/plot.py -y bikz_avg bikz_min bikz_max -x sigma -p solver sigma nfac conv  --use-all-jsons --sorted results/nt_leakage_hw_nt_hw_greedy``

### Fig. 8:

- run: ``./start_runs_all_threads.sh nt_leakage_intt.json nt_hw_intt_bp.json 12 nt_hw_intt_bp`` 
- plot: ``python scripts/plot.py -y bikz_avg bikz_min bikz_max -x sigma -p solver sigma nfac  --use-all-jsons --sorted results/nt_leakage_intt_nt_hw_intt_bp``

and 

- run: ``./start_runs_all_threads.sh nt_leakage_intt.json nt_hw_intt_greedy.json 12 nt_hw_intt_gr``
- plot: ``python scripts/plot.py -y bikz_avg bikz_min bikz_max -x sigma -p solver sigma nfac nd --use-all-jsons --sorted results/nt_leakage_intt_nt_hw_intt_greedy``

### Fig. 9:

- run: ``./start_runs_all_threads.sh nt_leakage_hw.json numerical_bp.json 6 nt_hw_numerical_bp``
- plot: ``python scripts/plot.py -y bikz_avg bikz_min bikz_max -x sigma -p solver sigma nfac  --use-all-jsons --sorted results/nt_leakage_hw_numerical_bp``

and 

(same as in Fig. 7)
- run: ``./start_runs_all_threads.sh nt_leakage_hw.json nt_hw_greedy.json 18 nt_hw_numerical_gr``
- plot: ``python scripts/plot.py -y bikz_avg bikz_min bikz_max -x sigma -p solver sigma nfac conv  --use-all-jsons --sorted results/nt_leakage_hw_nt_hw_greedy``

### Fig. 10:

- run: ``./start_runs_all_threads.sh nt_leakage_hw.json conv_eval_bp.json 12 nt_hw_conv_bp`` 
- plot: ``python scripts/plot.py -y bikz_avg bikz_min bikz_max -x conv -p solver sigma nfac conv  --use-all-jsons --sorted results/nt_leakage_hw_conv_eval_bp``

and 

- run: ``./start_runs_all_threads.sh nt_leakage_hw.json conv_eval_gr.json 12 nt_hw_conv_gr`` 
- plot: ``python scripts/plot.py -y bikz_avg bikz_min bikz_max -x conv -p solver sigma nfac conv  --use-all-jsons --sorted results/nt_leakage_hw_conv_eval_gr``

### Fig. 11:

- run: ``./start_runs_all_threads.sh modular_hint.json mod_hints_uni.json 10 mod_hints_uni`` 
- plot: ``python scripts/plot.py -y bikz_avg -x nfac -p solver nfac --use-all-jsons --sorted results/modular_hint_mod_hints_uni``

and 

- run: ``./start_runs_all_threads.sh modular_hint_bino.json mod_hints_bino.json 10 mod_hints_bino``
- plot: ``python scripts/plot.py -y bikz_avg -x nfac -p solver nfac --use-all-jsons --sorted results/modular_hint_bino_mod_hints_bino``


### runtime

- run: ``./start_runs.sh configs/leakage/perfect_hint_bino.json configs/settings/runtime_perf.json 0 2 2 40 runtime_perfect_hint_40``
- plot: ``python scripts/plot.py -y seconds_avg seconds_min seconds_max -x nfac -p solver nfac  --use-all-jsons --sorted results/perfect_hint_bino_runtime_perf``

and

- run: ``./start_runs.sh configs/leakage/perfect_hint_bino.json configs/settings/runtime_perf.json 0 2 1 1 runtime_perfect_hint_1``
- plot: ``python scripts/plot.py -y seconds_avg seconds_min seconds_max -x nfac -p solver nfac  --use-all-jsons --sorted results/perfect_hint_bino_runtime_perf``

and

- run: ``./start_runs.sh configs/leakage/approximate_hint_bino.json configs/settings/runtime.json 0 2 2 40 runtime_approx_hint_40``
- plot: ``python scripts/plot.py -y seconds_avg seconds_min seconds_max -x nfac -p solver nfac sigma  --use-all-jsons --sorted results/approximate_hint_bino_runtime``

and

- run: ``./start_runs.sh configs/leakage/approximate_hint_bino.json configs/settings/runtime_greedy.json 0 2 1 1 runtime_approx_hint_1``
- plot: ``python scripts/plot.py -y seconds_avg seconds_min seconds_max -x nfac -p solver nfac sigma  --use-all-jsons --sorted results/approximate_hint_bino_runtime_greedy``

## General usage

Use ``python eval.py [leakage_file] [setting file] --threads $THREADS --parts [from which setting to which setting]``.

For plotting use ``python scripts/plot.py -y [y_axis] -x [x_axis] --parameters [list all paremerts] --use-all-jsons --sorted [path_to_json]``.

## Settings 

### Fig. 3:

a) perfect_hint.json hints.json
b) perfect_hint_bino.json hints.json and perfect_hint_bino.json bino_hints.json
c) approximate_hint.json approximate_hint_greedy.json
d) approximate_hint_bino.json approximate_hint.json

### Fig. 4:

approximate_hint.json approximate_hint_single_greedy.json and approximate_hint_bino.json approximate_hint.json

### Fig. 6:

nt_leakage_value.json nt_vl_bp.json and nt_leakage_value.json nt_vl_greedy.json

### Fig. 7:

nt_leakage_hw.json nt_hw_bp.json and nt_leakage_hw.json nt_hw_greedy.json

### Fig. 8:

nt_leakage_intt.json nt_hw_intt_bp.json and nt_leakage_intt.json nt_hw_intt_greedy.json

### Fig. 9:

nt_leakage_hw.json numerical_bp.json
and 
(same as in Fig. 7)
nt_leakage_hw.json nt_hw_greedy.json

### Fig. 10:

nt_leakage_hw.json conv_eval_bp.json
and 
nt_leakage_hw.json conv_eval_gr.json

### Fig. 11:

modular_hint.json mod_hints_uni.json and modular_hint_bino.json mod_hints_uni.json

### runtime

perfect_hint_bino.json runtime_perf.json with 40 threads and 1 thread,
approximate_hint_bino.json runtime.json with 40 threads, and
approximate_hint_bino.json runtime_greedy.json with 1 thread


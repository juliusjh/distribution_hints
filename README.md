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
    
## Calls to reproduce Figures
The following calls require tmux (see [https://github.com/tmux/tmux/wiki](https://github.com/tmux/tmux/wiki)). After calling them, use ``tmux attach`` to see the output of the calls. Results will be under ``results/``.
For general usage see below.

### Fig. 3:

#### a)

``./start_runs_all_threads.sh perfect_hint.json hints.json 20 perfect_hints``

#### b)

``./start_runs_all_threads.sh perfect_hint_bino.json hints.json 20 perfect_hints_bino_gr``

and 

``./start_runs_all_threads.sh perfect_hint_bino.json bino_hints.json 10 perfect_hints_bino_bp``

#### c)

``./start_runs_all_threads.sh approximate_hint.json approximate_hint_greedy.json 10 approximate_hint_gr``

#### d)

``./start_runs_all_threads.sh approximate_hint_bino.json approximate_hint.json 16 approximate_hint_bino``

### Fig. 4:

``./start_runs_all_threads.sh approximate_hint.json approximate_hint_single_greedy.json 40 approximate_hint_single_greedy`` 

and 

``./start_runs_all_threads.sh approximate_hint_bino.json approximate_hint_single.json 40 approximate_hint_single_bino``

### Fig. 6:

``./start_runs_all_threads.sh nt_leakage_value.json nt_vl_bp.json 15 nt_vl_bp``

and 

``./start_runs_all_threads.sh nt_leakage_value.json nt_vl_greedy.json 15 nt_vl_greedy``

### Fig. 7:

``./start_runs_all_threads.sh nt_leakage_hw.json nt_hw_bp.json 18 nt_hw_bp`` 

and 

``./start_runs_all_threads.sh nt_leakage_hw.json nt_hw_greedy.json 18 nt_hw_gr``

### Fig. 8:

``./start_runs_all_threads.sh nt_leakage_intt.json nt_hw_intt_bp.json 12 nt_hw_intt_bp`` 

and 

``./start_runs_all_threads.sh nt_leakage_intt.json nt_hw_intt_greedy.json 12 nt_hw_intt_gr``

### Fig. 9:

``./start_runs_all_threads.sh nt_leakage_hw.json numerical_bp.json 6 nt_hw_numerical_bp`` 

and 

(same as in Fig. 7)
``./start_runs_all_threads.sh nt_leakage_hw.json nt_hw_greedy.json 18 nt_hw_numerical_gr``


### Fig. 10:

``./start_runs_all_threads.sh nt_leakage_hw.json conv_eval_bp.json 12 nt_hw_conv_bp`` 

and 

``./start_runs_all_threads.sh nt_leakage_hw.json conv_eval_gr.json 12 nt_hw_conv_gr`` 

### Fig. 11:

``./start_runs_all_threads.sh modular_hint.json mod_hints_uni.json 10 mod_hints_uni`` 

and 

``./start_runs_all_threads.sh modular_hint_bino.json mod_hints_bino.json 10 mod_hints_bino``


### runtime

``./start_runs.sh configs/leakage/perfect_hint_bino.json configs/settings/runtime_perf.json 0 2 2 40 runtime_perfect_hint_40``

``./start_runs.sh configs/leakage/perfect_hint_bino.json configs/settings/runtime_perf.json 0 2 1 1 runtime_perfect_hint_1``

``./start_runs.sh configs/leakage/approximate_hint_bino.json configs/settings/runtime.json 0 2 2 40 runtime_approx_hint_40``

``./start_runs.sh configs/leakage/approximate_hint_bino.json configs/settings/runtime_greedy.json 0 2 1 1 runtime_approx_hint_1``

## General usage

Use ``python eval.py [leakage_file] [setting file] --threads $THREADS --parts [from which setting to which setting]``.

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


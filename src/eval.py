import statistics
import json
import numpy as np
import git
import socket
import hashlib
import pathlib
import itertools
import functools
from datetime import datetime
from propagate import propagate_bp, propagate_greedy
from leakage import get_nt_leakage, get_approximate_hint, get_perfect_hint, get_mod_hint


def run_config(leakage_file, settings_file, bp_file, greedy_file, part=None, threads=None):
    print(
        f"Settings: {leakage_file=}, {settings_file=}, {bp_file=}, {greedy_file=}")
    with open(leakage_file) as f:
        params_leakage = json.load(f)
    with open(settings_file) as f:
        params_settings = json.load(f)
    with open(bp_file) as f:
        params_bp = json.load(f)
    with open(greedy_file) as f:
        params_greedy = json.load(f)

    results = AutoDict()
    settings = params_settings['settings']
    settings_all = list(itertools.product(*settings.values()))
    runs = params_settings['runs']

    save_json = params_settings.get("save_json", True)

    outpath = params_settings.get('outpath')
    if outpath is None:
        filename = leakage_file.split(
            "/")[-1].split(".")[0] + "_" + settings_file.split("/")[-1].split(".")[0]
        if params_settings['is_proper_eval']:
            outpath = f"../results/{filename}"
        else:
            outpath = f"../test_output/{filename}"

    results, metadata, number_runs, outpath = init(
        outpath=outpath,
        parameters_leakage=params_leakage,
        parameters_bp=params_bp,
        parameters_greedy=params_greedy,
        runs=settings | {'run_sets': list(range(runs))},
        save_json=save_json,
        part=part
    )

    count_settings = len(settings_all)
    current_setting = 0

    split_run = part is not None and (part[0] > 0 or part[1] < count_settings)
    metadata['split_run'] = split_run
    metadata['leakage_file'] = leakage_file
    metadata['settings_file'] = settings_file
    metadata['bp_file'] = bp_file
    metadata['greedy_file'] = greedy_file
    if split_run:
        metadata['lower_split'] = part[0]
        metadata['upper_split'] = part[1]

    def add_empty_stats(res_dict, setting_dict):
        res_dict = _get_run_stats(res_dict, setting_dict)
        res_dict["recovered"] = []
        res_dict["bikz"] = []
        res_dict["bikz_last"] = []
        res_dict["seconds"] = []
        res_dict["correct"] = []
        res_dict["abort"] = []
        res_dict["steps"] = []

    def add_run_stats(res_dict, stats, setting_dict):
        for kw in stats["bp"]:
            res_bp = _get_run_stats(res_dict["bp"], setting_dict)
            res_bp[kw].append(stats["bp"][kw])
        for kw in stats["greedy"]:
            res_greedy = _get_run_stats(res_dict["greedy"], setting_dict)
            res_greedy[kw].append(stats["greedy"][kw])

    runs_counter = 0
    if not split_run:
        total_runs = runs*count_settings
    else:
        total_runs = runs*(min(part[1], count_settings) - max(part[0], 0))

    base_seed = params_leakage["seed"]
    if params_settings.get("seed") is not None:
        print("Seed overwritten by settings.")
        base_seed = params_settings["seed"]

    for setting in settings_all:
        setting_dict = dict(zip(settings.keys(), setting))
        if split_run and (current_setting < part[0] or current_setting >= part[1]):
            print(
                f"Skipping setting {current_setting}/{count_settings} with {setting_dict=}")
            current_setting += 1
            continue
        if not params_settings.get("disable_bp", False):
            add_empty_stats(results["bp"], setting_dict)
        if not params_settings.get("disable_greedy", False):
            add_empty_stats(results["greedy"], setting_dict)
        print("\n" + "="*12 +
              f"Setting {current_setting}/{count_settings} with {setting_dict=}" + "="*12 + "\n")
        for run_set in range(runs):
            print(
                "-"*5 + f"Run {run_set}/{runs} in setting {current_setting}/{count_settings} (total runs: {runs_counter}/{total_runs}) with {setting_dict=}" + "-"*5 + "\n")
            params_leakage["seed"] = base_seed + current_setting*runs + run_set
            print("Seed is " + str(params_leakage["seed"]))
            res, disable_bp, disable_greedy = run_solvers(
                outpath=params_settings.get("outpath"),
                parameters_leakage=params_leakage | setting_dict,
                parameters_bp=params_bp | {'threads': threads},
                parameters_greedy=params_greedy | {'threads': threads},
                disable_bp=params_settings.get("disable_bp", False),
                disable_greedy=params_settings.get("disable_greedy", False),
            )
            params_settings['disable_bp'] = disable_bp
            params_settings['disable_greedy'] = disable_greedy
            add_run_stats(results, res, setting_dict)
            runs_counter += 1

            if save_json:
                save_results_json(results, outpath, metadata, part=True)
            print()
        for kw in ["correct", "bikz", "seconds", "recovered", "steps"]:
            if not params_settings.get("disable_bp", False):
                compute_stats(results, kw, setting_dict, "bp")
            if not params_settings.get("disable_greedy", False):
                compute_stats(results, kw, setting_dict, "greedy")
        current_setting += 1
    print("\n" + "*"*10 + "Done with all settings." + "*"*10 + "\n")
    if params_settings['is_proper_eval']:
        write_pgf_data(results, outpath)
    if save_json:
        save_results_json(results, outpath, metadata)
        # write_table(results, outpath, settings_all)


def _get_run_stats(res_dict, setting_dict):
    if isinstance(setting_dict, dict):
        setting_val = setting_dict.values()
    else:
        setting_val = setting_dict
    result = functools.reduce(lambda d, key: d[key], setting_val, res_dict)
    return result


def compute_stats(results, category, setting_dict, tp):
    res_dict = _get_run_stats(results[tp], setting_dict)
    min_v = min(res_dict[category])
    max_v = max(res_dict[category])
    avg_v = sum(res_dict[category]) / \
        len(res_dict[category])
    median_v = statistics.median(res_dict[category])
    res_dict[category+"_min"] = min_v
    res_dict[category+"_max"] = max_v
    res_dict[category+"_avg"] = avg_v
    res_dict[category+"_med"] = median_v


def write_table(results, outpath, settings_all):
    split = outpath.split("/")
    name = split[-1]
    outpath = "/".join(split[:-1])
    settings_done = []
    for setting in settings_all:
        if len(_get_run_stats(results['bp'], setting)) != 0 or len(_get_run_stats(results['greedy'], setting)) != 0:
            settings_done.append(setting)
    settings_done = {name: settings_done}
    outfile = f"{outpath}/table.json"
    f = pathlib.Path(outfile)
    if f.is_file():
        with open(outfile, 'r') as f:
            prev_data = json.load(f)
    else:
        prev_data = {}
    prev_data |= settings_done
    with open(outfile, 'w') as f:
        json.dump(prev_data, f, indent=4)
    print(f"Added to table in {outfile}")


def write_pgf_data(results, outpath):
    pass


def init(outpath, parameters_leakage, parameters_bp, parameters_greedy, runs, save_json, part):

    metadata = get_metadata(
        parameters_leakage=parameters_leakage,
        parameters_bp=parameters_bp,
        parameters_greedy=parameters_greedy,
        runs=runs
    )

    time = datetime.now().strftime('%m-%d-%Y_%H:%M:%S')
    dirty = "_dirty" if metadata['id']['git_dirty'] else ""
    if part is None:
        out_shared = f"{outpath}/{metadata['id']['git_commit'][:10]}{dirty}/run_{time}"
    else:
        out_shared = f"{outpath}/{metadata['id']['git_commit'][:10]}{dirty}/run_{part[0]}_{part[1]}_{time}"
    outpath = out_shared
    test_filename(f"{outpath}.json")
    test_filename(f"{outpath}.part.json")
    number_runs = np.prod([len(x) for x in runs.values()])

    results = AutoDict()
    if save_json:
        print(f"Starting, will save results to {outpath}.json")
    else:
        print("Starting..")
    return results, metadata, number_runs, outpath


def save_results_json(results, outpath, metadata, part=False):
    metadata['part'] = part
    filename = f"{outpath}.json" if not part else f"{outpath}.part.json"
    results['metadata'] = metadata
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved {'partial ' if part else ''}results to {filename}")


def run_solvers(outpath,
                parameters_leakage, parameters_bp, parameters_greedy, disable_bp=False, disable_greedy=False
                ):
    metadata = get_metadata(
        parameters_leakage=parameters_leakage,
        parameters_bp=parameters_bp,
        parameters_greedy=parameters_greedy)
    print(f"Run parameters: {metadata['parameters']}\n")

    # generate graph
    print("--" + "Generating leakage" + "--")
    time_start_leakage = datetime.now()
    if parameters_leakage['type'] == 'noiseterm':
        del parameters_leakage["type"]
        if not disable_greedy:
            greedy, key, a, b, _, _, _, kyber_seed = get_nt_leakage(
                **parameters_leakage | {"is_greedy": True})
        if not disable_bp:
            graph, key, a, b, _, _, _, kyber_seed = get_nt_leakage(
                **parameters_leakage | {"is_greedy": False})
    elif parameters_leakage['type'] == 'perfecthint':
        del parameters_leakage["type"]
        greedy, graph, key, a, b, _, _, _ = get_perfect_hint(**parameters_leakage)
    elif parameters_leakage['type'] == 'modhint':
        del parameters_leakage["type"]
        greedy, graph, key, a, b = get_mod_hint(**parameters_leakage)
    elif parameters_leakage['type'] == 'approximatehint':
        del parameters_leakage["type"]
        greedy, graph, key, a, b, _, _, _ = get_approximate_hint(**parameters_leakage | {"round": True})
    else:
        raise NotImplementedError
    parameters_greedy |= {"key": key, "a": a, "b": b}
    parameters_bp |= {"key": key, "a": a, "b": b}
    time_end_leakage = datetime.now()
    time_spent_leakage = f"{(time_end_leakage-time_start_leakage).total_seconds()/3600:.4f} hours"
    print(f"Leakage time: {time_spent_leakage}")
    print()

    statss_greedy = []
    abort_greedy = None
    disable_greedy = disable_greedy or greedy is None
    if not disable_greedy:
        print("--" + "Greedy solver" + "--")
        # solve with greedy
        time_spent_greedy, abort_greedy, statss_greedy, bikz_greedy, steps_greedy = propagate_greedy(
            greedy,
            **parameters_greedy
        )
        print()

    time_spent_bp = None
    stats_bp = {}
    statss_bp = []
    abort_bp = None
    disable_bp = disable_bp or graph is None
    if not disable_bp:
        # propagate graph
        print("--" + "BP solver" + "--")
        time_spent_bp, statss_bp, dists, _, _, abort_bp, steps_bp = propagate_bp(
            graph, **parameters_bp)
        print()

    res = AutoDict()

    if not disable_bp:
        res["bp"]["abort"] = abort_bp
        res["bp"]["seconds"] = time_spent_bp
        res["bp"]["steps"] = steps_bp
        res["bp"]["correct"] = get_max_from_statss(statss_bp, "correct")
        res["bp"]["recovered"] = get_max_from_statss(statss_bp, "recovered")
        res["bp"]["bikz"] = get_min_from_statss(statss_bp, "bikz")
        res["bp"]["bikz_last"] = statss_bp[-1]["bikz"]
    if not disable_greedy:
        res["greedy"]["abort"] = abort_greedy
        res["greedy"]["seconds"] = time_spent_greedy
        res["greedy"]["steps"] = steps_greedy
        res["greedy"]["correct"] = get_max_from_statss(
            statss_greedy, "correct")
        res["greedy"]["recovered"] = get_max_from_statss(
            statss_greedy, "recovered")
        res["greedy"]["bikz"] = get_min_from_statss(statss_greedy, "bikz")
        res["greedy"]["bikz_last"] = statss_greedy[-1]["bikz"]
    return res, disable_bp, disable_greedy


def get_min_from_statss(statss, kw):
    v = None
    for stats in statss:
        if v is None or stats[kw] < v:
            v = stats[kw]
    return v


def get_max_from_statss(statss, kw):
    v = None
    for stats in statss:
        if v is None or stats[kw] > v:
            v = stats[kw]
    return v


def stats_to_str(stats, abort, time_spent, metadata):
    stats = stats[-1]
    fileid = metadata['id']['fileid']
    result = f"avg_ent:{stats['avg_ent']:.2f}/min_ent:{stats['min_ent']:.2f}/max_ent:{stats['max_ent']:.2f}/correct:{stats['correct']}/recovered:{stats['recovered']}/abort:{abort}/time:{time_spent}/id:{fileid}"
    return result


def test_filename(filename, is_dir=False):
    f = pathlib.Path(filename)
    if not is_dir:
        d = f.parent
    else:
        d = f
    d.mkdir(parents=True, exist_ok=True)
    if not is_dir:
        f.touch(exist_ok=False)
    return True


def get_metadata(**kwparams):
    metadata = {}
    parameters = {}
    for key, value in kwparams.items():
        parameters[key] = value
    metadata['parameters'] = parameters
    param_hash = get_hash(parameters)  # base hash only on parameters
    git_hash, git_dirty = get_git_info()
    fileid = get_fileid(param_hash, git_hash, git_dirty)
    metadata['id'] = {
        'param_hash': param_hash,
        'git_commit': git_hash,
        'git_dirty': git_dirty,
        'fileid': fileid
    }
    time = datetime.now().isoformat()
    hostname = socket.gethostname()
    import __main__
    mainscript = __main__.__file__
    import sys
    metadata['environment'] = {
        'time': time,
        'hostname': hostname,
        'mainscript': mainscript,
        'cmd': " ".join(sys.argv[:])
    }
    return metadata


def get_git_info():
    repo = git.Repo(search_parent_directories=True)
    git_hash = repo.head.object.hexsha
    dirty = repo.is_dirty()
    return git_hash, dirty


def get_hash(data):
    hash = hashlib.sha3_256()
    params_string = json.dumps(data).encode('utf-8')
    hash.update(params_string)
    hash_string = hash.hexdigest()
    return hash_string


def get_fileid(param_hash, git_hash, git_dirty):
    fileid = f"{param_hash[:8]}_{git_hash[:8]}"
    fileid += "_D" if git_dirty else ""
    return fileid


class AutoDict(dict):
    # dict that automatically initializes a subdict if unset key is accessed
    def __getitem__(self, key):
        if key not in self:
            self[key] = AutoDict()
        return super().__getitem__(key)

    @classmethod
    def convert_to_dict(cls, auto_dict):
        if isinstance(auto_dict, dict):
            return {key: cls.convert_to_dict(value) for key, value in auto_dict.items()}
        else:
            return auto_dict


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int)
    parser.add_argument("leakage", type=str)
    parser.add_argument("settings", type=str)
    parser.add_argument("--bp", type=str, default="../configs/solvers/bp.json")
    parser.add_argument("--greedy", type=str,
                        default="../configs/solvers/greedy.json")
    parser.add_argument("--parts", type=int, nargs=2)
    args = parser.parse_args()
    run_config(leakage_file=args.leakage, part=args.parts, threads=args.threads,
               settings_file=args.settings, bp_file=args.bp, greedy_file=args.greedy)

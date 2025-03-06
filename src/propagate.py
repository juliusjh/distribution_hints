import math
import logging
import traceback
from datetime import datetime
from tqdm import tqdm
from util import get_stats
from solve import solve_from_greedy


def propagate_bp(graph, key, steps, bikz_abort, no_change_bikz_abort, a=None, b=None, batch=None, damping=None, worst_vars=None, threads=None):
    """
    steps: Number of steps to run, might abort early on success
    batch: batch size for faster updating (unfinished feature)
    damping: daming factor (unfinished feature)
    worst_vars: only update worst n var_nodes -> might converge faster (unfinished feature)
    """
    if threads is not None:
        print(f"Setting threads to {threads}.")
        graph.set_nthreads(threads)
    print(f"Propagating with {graph.get_nthreads()} threads.")
    statss = []
    distss = []
    facss = []
    varss = []
    start = datetime.now()

    print(f"Started: {start.strftime('%d.%m.%Y, %H:%M:%S')}")
    print("Descriptions: avg. entropy/min. entropy/max. entropy/correct/recovered/bikz")

    if isinstance(batch, tuple) or isinstance(batch, list):
        pass  # a tuple/list of batches will be run after each other, each a full number of steps
    else:
        batch = (batch, )
    tq = tqdm(range(steps*len(batch)), position=0, leave=True)

    def update(first=False):
        dists = graph.get_results()
        stats = get_stats(dists, key, a, b)
        statss.append(stats)
        distss.append(dists)
        if not first:
            print()
        tq.set_description(get_desc_str(stats))
    update(first=True)

    num_abort = len(key)//2
    abort = False
    last_bikz = []
    for i in tq:
        batch_i = batch[i//steps]
        try:
            graph.propagate(batch_i, damping, worst_vars=worst_vars)
        except:
            logging.error(traceback.format_exc())
            abort = "Error"
            break
        update()
        last_bikz.append(statss[-1]['bikz'])
        if statss[-1]['bikz'] is not None and statss[-1]['bikz'] <= bikz_abort:
            print(
                f"\nBIKZ sufficiently low ({statss[-1]['bikz']}<={bikz_abort}). stats={get_desc_str(statss[-1])}, step:{i}")
            abort = "Success (bikz low)"
            break
        if statss[-1]['recovered'] >= num_abort:
            print(
                f"\nRecovered sufficiently many ({statss[-1]['recovered']}) coefficients. stats={get_desc_str(statss[-1])}, step:{i}")
            abort = "Success (number recovered)"
            break
        if no_change_bikz_abort is not None and len(last_bikz) >= 2*no_change_bikz_abort:
            idx = len(last_bikz) - 2*no_change_bikz_abort
            idx2 = len(last_bikz) - no_change_bikz_abort
            min_bikz_last = min(last_bikz[idx:idx2])
            min_bikz = min(last_bikz[idx:])
            if min_bikz >= min_bikz_last:
                abort = "Fail (bikz diff)"
                print(
                    f"Finished without success (bikz diff). stats={get_desc_str(statss[-1])}, step:{i}")
                break
    else:
        print(
            f"Finished without success (max steps). stats={get_desc_str(statss[-1])}, step:{i}")
        abort = "Fail (max steps)"

    end = datetime.now()
    dists = graph.get_results()
    stats = get_stats(dists, key, a, b)
    time = (end-start).total_seconds()
    print(get_desc_str(stats))
    print(
        f"Done at {end.strftime('%m/%d/%Y, %H:%M:%S')} after {time/3600:.4f} hours.")
    return time, statss, distss, facss, varss, abort, i


def get_desc_str(stats):
    desc_string = f"{stats['avg_ent']:.2f}/{stats['min_ent']:.2f}/{stats['max_ent']:.2f}/{stats['correct']}/{stats['recovered']}/{stats['bikz']}"
    return desc_string


def propagate_greedy(greedy, key, steps, bikz_abort, no_change_bikz_abort, a=None, b=None, f_k=None, threads=None):
    nvar = len(key)
    a = a
    b = b

    round_len = int(math.log2(2*768))

    def default_fk(iter):
        div = pow(2, iter % round_len)
        k = int(nvar//div)
        return k
    if f_k is None:
        f_k = default_fk
    if steps is None:
        steps = 5000

    if threads is not None:
        print(f"Setting threads to {threads}.")
        greedy.set_nthreads(threads)
    print(f"Propagating with {greedy.get_nthreads()} threads.")

    start = datetime.now()
    print(f"Started: {start.strftime('%d.%m.%Y, %H:%M:%S')}")
    print("Descriptions: k/distance/correct/recovered/bikz")
    tq = tqdm(range(steps), position=0, leave=True)
    statss = []

    def update():
        k = default_fk(i)
        actions = greedy.get_actions()
        s2 = greedy.get_guess()
        recovered, rec = get_recovered_greedy(actions, key, s2)
        _, dif, cor = dist_keys(key, s2)
        bikz, _, _ = solve_from_greedy(a, b, s2, key, actions, rec)
        stats = f"{k:04d}/{dif:04d}/{cor}/{recovered}/{bikz}"
        tq.set_description(stats)
        return k, dif, bikz, recovered, {"recovered": recovered, "bikz": bikz, "correct": cor, "distance": dif}, cor

    last_bikz = []
    for i in tq:
        k, d, bikz, recovered, stats, cor = update()
        last_bikz.append(bikz)
        statss.append(stats)
        if bikz <= bikz_abort:
            print(f"BIKZ <= {bikz_abort}. {stats=}, step:{i}")
            abort = "Success (bikz low)"
            break
        if recovered > nvar//2:
            print(
                f"Found sufficiently many coefficients ({recovered=}). {stats=}, step:{i}")
            abort = "Success (number recovered)"
            break
        if no_change_bikz_abort is not None and len(last_bikz) >= 2*round_len*no_change_bikz_abort:
            idx = len(last_bikz) - 2*round_len*no_change_bikz_abort
            idx2 = len(last_bikz) - round_len*no_change_bikz_abort
            min_bikz_last = min(last_bikz[idx:idx2])
            min_bikz = min(last_bikz[idx:])
            if min_bikz >= min_bikz_last:
                print(
                    f"Finished without success (bikz abort). {stats=}, step:{i}")
                abort = "Fail (bikz diff)"
                break
        greedy.solve(k)
    else:
        print(f"Finished without success (max steps). {stats=}, step:{i}")
        abort = "Fail (max steps)"
    _, _, final_bikz, _, _, _ = update()
    end = datetime.now()
    print(
        f"Done at {end.strftime('%m/%d/%Y, %H:%M:%S')} after {(end-start).total_seconds()/3600:.4f} hours.")
    time_spent = (end-start).total_seconds()
    return time_spent, abort, statss, final_bikz, i


def get_recovered_greedy(actions, s, s2):
    recovered = []
    for i in range(len(s)):
        if actions[i][0] != 0:
            continue
        j = actions[i][2]
        if s[j] != s2[j]:
            break
        recovered.append((j, s[j]))
    return len(recovered), recovered


def dist_keys(s, s2):
    dist = [si-s2i for si, s2i in zip(s, s2)]
    cor = sum(1 if dist_i == 0 else 0 for dist_i in dist)
    return dist, sum(map(lambda x: pow(x, 2), dist)), cor

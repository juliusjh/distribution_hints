import math
import json
import random
import scipy
import numpy as np
from error_term_dict import get_vrange, get_vrange_limits
from solve import solve_from_bp

KYBER_Q = 3329


def check_nan(dist):
    for p in dist:
        if math.isnan(p) or p is None:
            return False
    return True


def get_stats(dists, key, a, b):
    entropies = []
    guess = []
    max_ent = -1
    min_ent = 1 << 16
    avg_ent = 0
    for dist in dists:
        entropy = scipy.stats.entropy(dist)
        if entropy < 0:
            print(dist)
        likliest = to_likeliest(dist)
        entropies.append(entropy)
        guess.append(likliest)
        max_ent = max(max_ent, entropy)
        min_ent = min(min_ent, entropy)
        avg_ent += entropy
    avg_ent /= len(dists)
    recovered = 0
    broken_chain = False
    correct = 0
    sorted_ent = sorted(enumerate(entropies), key=lambda x: x[1])
    for i, _ in sorted_ent:
        if guess[i] == key[i]:
            correct += 1
            if not broken_chain:
                recovered += 1
        else:
            broken_chain = True

    if a is not None and b is not None:
        bikz, _, _ = solve_from_bp(a, b, (dists, sorted_ent, recovered), key)
    else:
        bikz = None
    dist = sum([pow(si-s2i, 2) for si, s2i in zip(guess, key)])
    return {'avg_ent': avg_ent, 'max_ent': max_ent, 'min_ent': min_ent, 'guess': guess, 'entropies': entropies, 'correct': correct, 'recovered': recovered, 'bikz': bikz, 'dist': dist}


def hw(v, int_size):
    if v >= 0:
        return int.bit_count(v)
    return int_size - int.bit_count(-v - 1)


def hw_leakage_to_value(hw_dist, cut_off, int_size=16):
    # Factors in noise term probabilities
    v_dist = []
    error_term_vrange = get_vrange(cut_off)
    for v, p in error_term_vrange:  # get_vrange contains noise term probs
        v_cong = get_all_congruent_q(v)
        p_hw = 0.0
        for v_i in v_cong:
            p_hw += hw_dist.get(hw(v_i, int_size), 0.0)
        if p_hw >= cut_off:
            v_dist.append((v, p*p_hw))
    # normalize_tuples(v_dist)
    return v_dist


def get_all_congruent_q(value, limits=(int(-0.51*KYBER_Q), int(1.5*KYBER_Q))):
    # limits determined from simulation
    result = []
    x = value
    while x >= limits[0]:
        result.append(x)
        x -= KYBER_Q
    x = value + KYBER_Q
    while x <= limits[1]:
        result.append(x)
        x += KYBER_Q
    return result


def hw_leakage_to_value_no_fac(hw_dist, cut_off, int_size=16):
    # Does not factor in noise term value probabilities
    # most likely used in masked version (see hw_leakage_to_value_masked)
    v_dist = []
    for v in range(-(KYBER_Q//2), (KYBER_Q//2)+1):
        p_hw = hw_dist.get(hw(v, int_size), 0.0)
        if p_hw >= cut_off:
            v_dist.append((v, p_hw))
    # normalize_tuples(v_dist)
    return v_dist


def hw_leakage_to_value_no_fac_reduce(hw_dist, cut_off, int_size=16):
    # Does not factor in noise term value probabilities
    # most likely used in masked version (see hw_leakage_to_value_masked)
    v_dist = []
    for v in range(-(KYBER_Q//2), (KYBER_Q//2)+1):
        v_cong = get_all_congruent_q(v)
        p_hw = 0.0
        for v_i in v_cong:
            p_hw += hw_dist.get(hw(v_i, int_size), 0.0)
        if p_hw >= cut_off:
            v_dist.append((v, p_hw))
    # normalize_tuples(v_dist)
    return v_dist


def unmask_val_dists(dist0, dist1, cut_off):
    # combine values of two masked distributions to get original value
    # only consider likely original values (p>=cut_off)
    # Does not factor in noise term value probabilities
    vrange_min, vrange_max = get_vrange_limits(
        cut_off)  # gets range of likely values
    # # Original version, replaced by optimized numpy version
    # dist_res = {x: 0.0 for x in range(vrange_min, vrange_max+1)}
    # dist1_dict = dict(dist1)
    # for v0, p0 in dist0:
    #     for v in range(vrange_min, vrange_max+1):
    #         v1 = v - v0
    #         p1 = dist1_dict.get(v1, 0.0)
    #         p = p0*p1
    #         dist_res[v] += p
    # # Optimized numpy version
    v0_vals, p0_vals = zip(*dict(dist0).items())
    v1_vals, p1_vals = zip(*dict(dist1).items())

    v0_arr = np.array(v0_vals)
    p0_arr = np.array(p0_vals)
    v1_arr = np.array(v1_vals)
    p1_arr = np.array(p1_vals)

    v_arr = reduce_sym(v0_arr[:, np.newaxis] + v1_arr)
    p_arr = p0_arr[:, np.newaxis] * p1_arr

    valid_indices = np.where((v_arr >= vrange_min) & (v_arr <= vrange_max))
    valid_v = v_arr[valid_indices]
    valid_p = p_arr[valid_indices]
    dist_res_np = np.zeros(vrange_max - vrange_min + 1)
    np.add.at(dist_res_np, valid_v, valid_p)
    dist_res_dict = {i: dist_res_np[i]
                     for i in range(vrange_min, vrange_max + 1)}
    # Test to verify numpy version
    # for k,v in dist_res.items():
    #     assert dist_res_dict[k] == v
    return dist_res_dict


def apply_noise_term_prob(dist, cut_off, params_nd, factor_in):
    if not isinstance(dist, dict):
        dist = dict(dist)
    # Factors in noise term probabilities
    dist_res = {}
    error_term_vrange = get_vrange(cut_off, params_nd)
    for v, p in error_term_vrange:
        p_val = dist.get(v, 0.0)
        if p_val >= cut_off:
            if factor_in:
                dist_res[v] = p*p_val
            else:
                dist_res[v] = p_val
    return dist_res


def flatten(ls):
    return (li for l in ls for li in l)


def reduce_sym(a, q=KYBER_Q):
    return ((a + (q//2)) % q) - (q//2)


def reduce_sym_list(a, q=3329):
    return list(map(lambda x: reduce_sym(x, q), a))


def normalize_tuples(dist):
    s = sum(d[1] for d in dist)
    for i in range(len(dist)):
        dist[i] = (dist[i][0], dist[i][1]/s)


def to_likeliest(res):
    return from_compl(np.argmax(res), len(res))


def from_compl(v, sz_msg):
    if v > sz_msg/2:
        return v - sz_msg
    else:
        return v


def to_compl(v, sz_msg):
    if v < 0:
        return v + sz_msg
    else:
        return v


def dict_to_compl(dist, sz_msg):
    compl = [0.0 for _ in range(sz_msg)]
    for v in dist:
        compl[to_compl(v, sz_msg)] = dist[v]
    return compl


def bino(eta):
    def binp(x): return scipy.special.binom(eta, x) / 2**eta
    return {
        i: sum([binp(x) * binp(x + i) for x in range(-eta, eta + 1)])
        for i in range(-eta, eta + 1)
    }


def sample_bino(eta):
    return sum([random.randint(0, 1) for _ in range(eta)]) - sum([random.randint(0, 1) for _ in range(eta)])


def to_probabilities(dist):
    if isinstance(dist, dict):
        s = sum(d for d in dist.values())
        return {x: dist[x]/s for x in dist}
    else:
        s = sum(d[1] for d in dist)
        return [(x[0], x[1]/s) for x in dist]


def subtract_noise_const(dist, noise_const):
    if isinstance(dist, dict):
        dist_new = {x-noise_const: y for x, y in dist.items()}
    else:
        dist_new = [(x[0]-noise_const, x[1]) for x in dist]
    return dist_new


def subtract_noise_const_reduced(dist, noise_const):
    if isinstance(dist, dict):
        dist_new = {reduce_sym(x-noise_const): y for x, y in dist.items()}
    else:
        dist_new = [(reduce_sym(x[0]-noise_const), x[1]) for x in dist]
    return dist_new


def load_key(file):
    with open(file) as f:
        ls = json.load(f)
        return ls['a'], ls['b'], ls['sk']


def entropy(dist):
    ent = 0
    if isinstance(dist, dict):
        for p in dist.values():
            ent -= p*math.log2(p)
    else:
        for _, p in dist:
            ent -= p*math.log2(p)
    return ent


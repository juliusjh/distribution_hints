import random
import scipy
import numpy as np
import python_kyber768 as python_kyber
from scipy.stats import norm
from hint_solver import PyBP as Bp
from hint_solver import PyProbBP as ProbBp
from hint_solver import PyGreedy as Greedy
from util import dict_to_compl, bino, sample_bino, hw_leakage_to_value, hw, to_probabilities, apply_noise_term_prob, subtract_noise_const, hw_leakage_to_value_no_fac, reduce_sym_list, reduce_sym, flatten, subtract_noise_const_reduced, entropy
from error_term_dict import get_vrange
from error_term import extract_error_term_coefficients

NVAR = 2*768
cut_off = 1e-10
KYBER_Q = 3329
KYBER_N = 256
RANGE_MASK = KYBER_Q//2


def create_graph(eqs, rhs, prior=None, eta=2, p=None):
    if prior is None:
        prior = dict_to_compl(bino(eta), 2*eta+1)
    else:
        assert len(prior) == 2*eta+1
    if p is None:
        graph = Bp(prior, eqs, rhs)
    else:
        graph = ProbBp(prior, eqs, rhs, p)
    chk_sz = graph.get_chk_sz()
    print(f"Size of sum distributions: {min(chk_sz)=}, {max(chk_sz)=}")
    return graph


def test_bp(nfac, eq_range=(-10, 10), eta=2, hw_leakage=False):
    s = [sample_bino(eta) for _ in range(NVAR)]
    eqs = [[random.randint(eq_range[0], eq_range[1])
            for _ in range(NVAR)] for _ in range(nfac)]
    rhs = []
    for eq in eqs:
        rh = sum([eqi*si for eqi, si in zip(eq, s)])
        if hw_leakage:
            hw_leak = hw_leakage_to_value(dict([(hw(rh, 16), 1.0)]), cut_off)
            rhs.append(hw_leak)
        else:
            rhs.append([(rh, 1.0)])
    graph = create_graph(
        eqs, rhs, eta=eta)
    return graph, s, eqs, rhs


def flatten_key(sample):
    s = [si.to_list()
         for si in sample.sk.sk.intt().montgomery_reduce().to_list()]
    sflat = [sik for si in s for sik in si]
    e = [ei.to_list() for ei in sample.e.to_list()]
    eflat = [eik for ei in e for eik in ei]
    key = eflat + sflat
    key = reduce_sym_list(key)
    return key


def get_nt_leakage(nfac, sigma, hw_leakage, int_size, seed, max_noise_const, params_noise_dist, target, is_greedy, v_filter=None):
    """"
    sigma: Noise level (None should be used for 0)
    nfac: Number of Factor Nodes (up to 256 factor node per trace)
    """
    np.random.seed(seed)
    range_hw = None
    if hw_leakage:
        range_hw = range(0, int_size + 1)
    kyber_instance_seed = [np.random.randint(0, 1 << 32) for _ in range(32)]
    print(f"Seed for Kyber instance: {kyber_instance_seed}")
    python_kyber.set_seed(kyber_instance_seed)
    sample = python_kyber.KyberSample.generate(False)
    key = flatten_key(sample)

    n = nfac
    rhs = []
    eqs = []
    mess = []
    ents = []
    count = 0
    num_cts = 0
    while count < n:
        print(f"{count}/{n}\t\t\t", end='\r')
        sample = python_kyber.KyberSample.generate_with_key(
            False, sample.pk, sample.sk, sample.e
        )
        error_term = reduce_sym_list(calc_error_term(sample)[0].to_list())
        msg_bits = []
        nu = sample.nu
        used_ct = False
        for k in range(32):
            for bi in range(8):
                msg_bits.append((nu[k] >> bi) & 1)
        # print(msg_bits)
        v_list = sample.ct.v.to_list()
        for i in range(256):
            if msg_bits[i] == 1:
                continue
            noise_coeffs, noise_const = extract_error_term_coefficients(
                sample, max_delta_v=max_noise_const, coeff_index=i)
            v_ct = v_list[i]
            if not noise_coeffs or (v_filter and v_ct >= v_filter):
                continue
            used_ct = True
            # noisy_message2 = sum(ci*ki for ci, ki in zip(noise_coeffs, key)) + noise_const
            noisy_message = error_term[i]
            # assert noisy_message == noisy_message2, f"{noisy_message=}, {noisy_message2=}"
            leak_dist, mes = get_leakage_dist(sigma=sigma, v_ct=v_ct, noise_const=noise_const, noisy_message=noisy_message, int_size=int_size,
                                              range_hw=range_hw, hw_leakage=hw_leakage, params_noise_dist=params_noise_dist, target=target, reduced=True, is_greedy=is_greedy)

            ents.append(entropy(leak_dist))

            rhs.append(leak_dist)
            eqs.append(noise_coeffs)
            mess.append(mes)
            count += 1
            print(f"{count}/{n}\t\t\t", end='\r')
            if count == n:
                break
        if used_ct:
            num_cts += 1
        if count == n:
            break

    print(f"Required {num_cts} ciphertexts for {count} hints.")
    print(f"Loaded {count}/{n}. Creating graph&greedy..")
    print(f"Entropy: max={max(ents)}, min={min(ents)}, avg={np.mean(ents)}, med={np.median(ents)}")
    if is_greedy:
        solver = Greedy(eqs, rhs)
    else:
        solver = create_graph(eqs, rhs, eta=2)
    print("Done.")
    a, b = lwe_from_mlwe(sample)
    return solver, key, a, b, eqs, rhs, mess, kyber_instance_seed


def calc_error_term(sample):
    msg = sample.get_msg()
    su = python_kyber.Polyvec.scalar(sample.sk.sk, sample.ct.b.ntt()).intt()
    rec = (sample.ct.v - su).reduce()
    assert sample.nu == rec.to_msg()
    r = rec - msg
    return r, rec


def get_hw_leak_dist(actual_hw, sigma, range_hw):
    if sigma is not None:
        hw_leak = np.random.normal(actual_hw, sigma)
        hw_dist_ar = norm.pdf(hw_leak, range_hw, sigma)
        hw_dist = dict(list(zip(range_hw, hw_dist_ar)))
    else:
        hw_dist = dict([(actual_hw, 1.0)])
    return hw_dist


def get_vl_leak_dist(actual_val, sigma, range_val):
    if sigma is not None:
        val_leak = np.random.normal(actual_val, sigma)
        val_dist_ar = norm.pdf(val_leak, range_val, sigma)
        val_dist = dict(list(zip(range_val, val_dist_ar)))
    else:
        val_leak = actual_val
        val_dist = dict([(actual_val, 1.0)])
    return val_dist, actual_val


def gen_mask(range_mask=RANGE_MASK):
    # Proper range of mask
    # According to mkm4
    # Source: https://github.com/masked-kyber-m4/mkm4/blob/f2b62907a95b8b035282cdd1cd650b66ad34a616/crypto_kem/kyber768/m4/masked.c#L71-L73
    # Mask generation:
    # random int16, cut to 12 bits, and shifted symmetric around 0
    # discarded if outside of >q/2 or -q/2
    # applied as x_0=m, x_1=-m+s
    # Range of mask: -q/2 to q/2
    mask = random.randint(-range_mask, range_mask)
    return mask


def get_from_invntt_ll_to_ravi(val, v):
    f = 512  # mont/128
    # fqmul (includes 1/mont) with f = 1441 #mont^2/128
    val_out = (val * f) % KYBER_Q
    val_diff = v - val_out
    return val_diff


def get_leakage_dist_masked_hw():
    raise NotImplementedError


def get_leakage_dist_masked_value(sigma, noise_const, noisy_message, noisy_message_masked, reduced):
    raise NotImplementedError


def get_leakage_dist_value(sigma, noise_const, noisy_message, reduced, params_noise_dist, is_greedy):
    if reduced:
        range_val = range(-(KYBER_Q//2), (KYBER_Q//2)+1)
    else:
        range_val = range(int(-0.51*KYBER_Q), int(1.5*KYBER_Q))
    val_dist, val_leak = get_vl_leak_dist(noisy_message, sigma, range_val)
    val_dist = subtract_noise_const(val_dist, noise_const)
    val_dist = apply_noise_term_prob(
        val_dist, cut_off, params_nd=params_noise_dist, factor_in=is_greedy)
    val_dist = to_probabilities(val_dist)
    return list(val_dist.items()), val_leak


def get_leakage_dist_hw_intt(sigma, ct_v, noise_const, noisy_message, int_size, range_hw, params_noise_dist, is_greedy):
    us = reduce_sym(ct_v-noisy_message)
    actual_hw = hw(us, int_size)
    hw_dist = get_hw_leak_dist(actual_hw, sigma, range_hw)
    val_dist = hw_leakage_to_value_no_fac(hw_dist, cut_off)
    val_dist = to_probabilities(val_dist)
    val_dist = negate_dist(val_dist)
    val_dist = subtract_noise_const_reduced(val_dist, (noise_const-ct_v))
    val_dist = apply_noise_term_prob(
        val_dist, cut_off, params_nd=params_noise_dist, factor_in=is_greedy)
    val_dist = to_probabilities(val_dist)
    return list(val_dist.items()), actual_hw


def negate_dist(val_dist):
    new_d = []
    for v in val_dist:
        new_d.append((-v[0], v[1]))
    return new_d


def get_leakage_dist_hw(sigma, noise_const, noisy_message, int_size, range_hw, params_noise_dist, is_greedy):
    actual_hw = hw(noisy_message, int_size)
    hw_dist = get_hw_leak_dist(actual_hw, sigma, range_hw)
    val_dist = hw_leakage_to_value_no_fac(hw_dist, cut_off)
    val_dist = to_probabilities(val_dist)
    val_dist = subtract_noise_const(val_dist, noise_const)
    val_dist = apply_noise_term_prob(
        val_dist, cut_off, params_nd=params_noise_dist, factor_in=is_greedy)
    val_dist = to_probabilities(val_dist)
    return list(val_dist.items()), actual_hw


def get_leakage_dist(sigma, v_ct, noise_const, noisy_message, int_size, range_hw, hw_leakage, target, reduced, is_greedy, params_noise_dist=None):
    if target == "subtraction":
        if hw_leakage:
            return get_leakage_dist_hw(sigma, noise_const, noisy_message, int_size, range_hw, params_noise_dist, is_greedy=is_greedy)
        else:
            return get_leakage_dist_value(sigma, noise_const, noisy_message, reduced, params_noise_dist, is_greedy=is_greedy)
    elif target == "intt":
        if hw_leakage:
            return get_leakage_dist_hw_intt(sigma, v_ct, noise_const, noisy_message, int_size, range_hw, params_noise_dist, is_greedy=is_greedy)
        else:
            raise NotImplementedError
        pass
    else:
        raise ValueError
        # return get_leakage_dist_masked_value()


def test_simulated_intt_leakage(ntraces, nshares, sigma, v, us):
    pass


def get_approximate_hint(nfac, sigma, seed, eq_range=[-3329//2, 3329//2], bino=False, eta=10, round=True):
    NVAR = 1536
    np.random.seed(seed)
    kyber_instance_seed = [np.random.randint(0, 1 << 32) for _ in range(32)]
    print(f"Seed for Kyber instance: {kyber_instance_seed}")
    python_kyber.set_seed(kyber_instance_seed)
    sample = python_kyber.KyberSample.generate(False)
    s = flatten_key(sample)
    a, b = lwe_from_mlwe(sample)
    if bino:
        eqs = [[sample_bino(eta)
                for _ in range(NVAR)] for _ in range(nfac)]
    else:
        eqs = [[random.randint(eq_range[0], eq_range[1])
                for _ in range(NVAR)] for _ in range(nfac)]
    rhs = []
    mes = []
    for eq in eqs:
        n = np.random.normal(0, sigma)
        if round:
            rh = int(np.round(sum([eqi*si for eqi, si in zip(eq, s)]) + n))
        else:
            rh = sum([eqi*si for eqi, si in zip(eq, s)]) + n
        mes.append(rh)
        rh_i = []
        for i in range(100):
            rh_i.append((rh+i, scipy.stats.norm.pdf(i, 0, sigma)))
        rhs.append(rh_i)

    graph = None
    greedy = None
    if round:
        if bino:
            graph = create_graph(eqs, rhs, eta=2)
        greedy = Greedy(eqs, rhs)
    # time_spent_greedy, abort_greedy, statss_greedy, bikz_greedy, steps_greedy = propagate_greedy(greedy, steps=5000, s=s, a=a, b=b)
    return greedy, graph, s, a, b, eqs, rhs, mes


def get_perfect_hint(nfac, seed, eq_range=[-3329//2, 3329//2], bino=False, eta=10, p=None):
    NVAR = 1536
    np.random.seed(seed)
    kyber_instance_seed = [np.random.randint(0, 1 << 32) for _ in range(32)]
    print(f"Seed for Kyber instance: {kyber_instance_seed}")
    python_kyber.set_seed(kyber_instance_seed)
    sample = python_kyber.KyberSample.generate(False)
    s = flatten_key(sample)
    a, b = lwe_from_mlwe(sample)
    if bino:
        eqs = [[sample_bino(eta)
                for _ in range(NVAR)] for _ in range(nfac)]
    else:
        eqs = [[random.randint(eq_range[0], eq_range[1])
                for _ in range(NVAR)] for _ in range(nfac)]
    rhs = []
    mes = []
    n = len(eqs)
    j = 0
    incor = 0
    for eq in eqs:
        rh = sum([eqi*si for eqi, si in zip(eq, s)])
        rh_i = []
        if p is not None and p < 1.0:
            max_range = 0
            for c in eq:
                max_range += 2*abs(c)  # Assumes Kyber768 or 1024!
            if random.random() > p:
                incor += 1
                rh = random.randint(-max_range, max_range)
        mes.append(rh)
        rh_i = [(rh, 1.0)]
        rhs.append(rh_i)
        print(f"{j}/{n}", end='\r')
        j += 1
    print(f"Sampled {n} perfect hints ({incor} incorrects).")
    graph = None
    if bino:
        if p is None or p >= 1.0:
            graph = create_graph(eqs, rhs, eta=2)
        else:
            graph = create_graph(eqs, rhs, eta=2, p=[p]*len(rhs))
    greedy = Greedy(eqs, rhs)
    # time_spent_greedy, abort_greedy, statss_greedy, bikz_greedy, steps_greedy = propagate_greedy(greedy, steps=5000, s=s, a=a, b=b)
    return greedy, graph, s, a, b, eqs, rhs, mes


def get_mod_hint(nfac, seed, eq_range, bino, eta):
    NVAR = 1536
    q = 3329
    np.random.seed(seed)
    kyber_instance_seed = [np.random.randint(0, 1 << 32) for _ in range(32)]
    print(f"Seed for Kyber instance: {kyber_instance_seed}")
    python_kyber.set_seed(kyber_instance_seed)
    sample = python_kyber.KyberSample.generate(False)
    s = flatten_key(sample)
    a, b = lwe_from_mlwe(sample)
    if bino:
        eqs = [[sample_bino(eta)
            for _ in range(NVAR)] for _ in range(nfac)]
    else:
        eqs = [[random.randint(eq_range[0], eq_range[1])
            for _ in range(NVAR)] for _ in range(nfac)]
    rhs = []
    for eq in eqs:
        max_range = 0
        for c in eq:
            max_range += 2*abs(c)
        rh = sum([eqi*si % q for eqi, si in zip(eq, s)])
        rh = rh % q
        qs = rh
        r = []
        while qs <= max_range:
            r.append((qs, 1.0))
            qs += q
        qs = rh - q
        while qs >= -max_range:
            r.append((qs, 1.0))
            qs -= q
        r = to_probabilities(r)
        rhs.append(r)
    greedy = Greedy(eqs, rhs)
    graph = create_graph(eqs, rhs, eta=2)
    return greedy, graph, s, a, b


def lwe_from_mlwe(sample, q=3329):
    k = len(sample.pk.a)

    def a_map(p):
        return p.intt().montgomery_reduce().to_list()

    a = [[a_map(sample.pk.a[i].to_list()[j])
          for j in range(k)] for i in range(k)]
    b = sample.pk.pk.intt().montgomery_reduce().to_lists()
    a_s = (sample.sk.sk.apply_matrix_left_ntt(sample.pk.a)).intt()
    a_s_e = a_s + sample.e
    assert (
        a_s_e.reduce().to_lists() ==
        sample.pk.pk.intt().montgomery_reduce().to_lists()
    )
    #############
    a_mat = list(
        map(
            lambda ai: ai.intt().montgomery_reduce().to_lists(),
            sample.pk.a,
        )
    )
    b_vec = sample.pk.pk.intt().montgomery_reduce().to_lists()
    a, b = mlwe_to_lwe(a_mat, b_vec)
    return a, b


def rlwe_to_lwe(poly, q=3329):
    n = len(poly)
    mat = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        mat[i] = poly.copy()
        new_poly = [None for _ in range(n)]
        poly_last = poly[n - 1]
        for i, c in enumerate(poly[: n - 1]):
            new_poly[i + 1] = c
        new_poly[0] = (-poly_last) % q
        poly = new_poly
    return np.transpose(np.array(mat)).tolist()


def mlwe_to_lwe(A, b):
    assert len(A[0][0]) == len(b[0])
    poly_len = len(b[0])
    k = len(b)
    n = k * poly_len
    m = len(A) * poly_len
    mat = np.zeros(shape=(m, n), dtype=np.int64)
    for i, row_a in enumerate(A):
        for j, a in enumerate(row_a):
            mat[
                i * poly_len: i * poly_len + poly_len,
                j * poly_len: j * poly_len + poly_len,
            ] = rlwe_to_lwe(a)
    b = list(flatten(b))
    return mat.tolist(), b


def poly_to_matrix(poly):
    k = len(poly)

    def sign(i, j):
        return 1 if ((i % k) + (j % k)) < k else -1

    res = [[None for _ in range(k)] for _ in range(k)]
    for i in range(len(poly)):
        for j in range(len(poly)):
            idx = (k - i + j) % k
            res[i][j] = sign(i, idx) * poly[idx]
    return res

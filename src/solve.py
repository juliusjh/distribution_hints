import numpy as np


def from_compl(v, sz_msg):
    if v > sz_msg/2:
        return v - sz_msg
    else:
        return v


def to_likeliest(res):
    return from_compl(np.argmax(res), len(res))


def solve_from_greedy(a, b, guess, x, actions, recovered):
    a = np.array(a)
    b = np.array(b)
    x = np.array(x)
    n = a.shape[0]
    e_recovered = []
    s_recovered = []
    for (j, val) in recovered:
        if j < n:
            e_recovered.append((j, val))
        else:
            s_recovered.append((j-n, val))
    sorted_ents = []
    sorted_actions = sorted(actions, key=lambda x: x[1], reverse=True)
    for (_change, score, idx) in sorted_actions:
        sorted_ents.append((idx, score))
        
    return solve_lat(a, b, e_recovered, s_recovered, sorted_ents, x, dists=None, guess=guess)


def solve_from_bp(a, b, bp_out, x):
    dists, sorted_ents, recovered = bp_out[0], bp_out[1], bp_out[2]
    x = np.array(x)
    a = np.array(a)
    b = np.array(b)
    n = len(dists)//2
    e_recovered = []
    s_recovered = []
    for i in range(recovered):
        idx, _ = sorted_ents[i]
        d = dists[idx]
        if idx < n:
            e_recovered.append((idx, to_likeliest(d)))
        else:
            s_recovered.append((idx-n, to_likeliest(d)))
    return solve_lat(a, b, e_recovered, s_recovered, sorted_ents, x, dists, guess=None)


def solve_lat(a, b, e_recovered, s_recovered, sorted_ents, x, dists, guess, q=3329):
    assert (guess is None or dists is None) and (guess is not None or dists is not None)
    n = a.shape[0]
    assert x.shape[0] == 2*n
    assert a.shape[0] == n
    assert a.shape[1] == n
    assert b.shape[0] == n
    assert (((a@x[n:] + x[:n]) % q) == (b % q)).all(), "sk/pk mismatch"
    # Remove recovered s
    columns_to_delete = []
    for s_to_remove, sj in s_recovered:
        col = a[:, s_to_remove]
        v = col*sj
        b -= v
        columns_to_delete.append(s_to_remove)
        a[:, s_to_remove] = 0
    assert ((a@x[n:] + x[:n]) % q == b % q).all()

    # Use recovered e to remove more s
    saved_eqs = []
    rows_to_delete = []
    for i, ei in e_recovered:
        s_to_remove = None
        while len(sorted_ents) > 0 and s_to_remove is None:
            s_to_remove = sorted_ents.pop()[0]
            if s_to_remove < n or s_to_remove-n in columns_to_delete or a[i, s_to_remove-n] % q == 0:
                s_to_remove = None
        if s_to_remove is None:
            return 0, None, None # TODO
        s_to_remove -= n
        saved_eqs.append((a[i, :].copy(), s_to_remove))
        columns_to_delete.append(s_to_remove)
        ai_inv = pow(int(a[i, s_to_remove]), -1, q)
        sc_add = (ai_inv*a[i, :]) % q
        b_add = (ai_inv*(b[i]-ei)) % q
        for ip in range(n):
            aip = a[ip, s_to_remove]
            a[ip, :] -= aip * sc_add
            a[ip, :] %= q
            b[ip] = (b[ip] - aip*b_add) % q
            a[ip, s_to_remove] = 0
        a[:, s_to_remove] = 0
        a[i, :] = 0
        x[i] = 0
        b[i] = 0
        rows_to_delete.append(i)
    assert ((a@x[n:] + x[:n]) % q == b % q).all(), f"{(a@x[n:] + x[:n]) % q}\n{b % q=}\n{(a@x[n:] + x[:n]) % q - b % q=}"

    # TODO: Not necessary to carry this out
    rows_to_delete.sort(reverse=True)
    columns_to_delete.sort(reverse=True)
    for s_to_remove in columns_to_delete:
        a = np.delete(a, s_to_remove, axis=1)
        x = np.delete(x, s_to_remove+n, axis=0)
        if dists is not None:
            del dists[s_to_remove+n]
        if guess is not None:
            del guess[s_to_remove+n]
    for i in rows_to_delete:
        a = np.delete(a, i, axis=0)
        b = np.delete(b, i, axis=0)
        x = np.delete(x, i, axis=0)
        if dists is not None:
            del dists[i]
        if guess is not None:
            del guess[i]

    assert len(e_recovered) == len(rows_to_delete), f"{len(e_recovered)=} != {len(rows_to_delete)=}"
    assert len(s_recovered) + len(e_recovered) == len(columns_to_delete)
    assert a.shape[0] == n - len(e_recovered)
    assert a.shape[1] == n - len(e_recovered) - len(s_recovered)

    if dists is not None:
        best_guess = np.array(list(map(to_likeliest, dists)))
    else:
        best_guess = np.array(guess)
    lat_elem = x - best_guess
    lat_elem = np.append(lat_elem, 1)
    ln_volume = n * np.log(q)
    dim = a.shape[0] + a.shape[1] + 1
    bikz = get_bikz(dim, ln_volume, lat_elem)
    return bikz, a, saved_eqs


def get_bikz(dim, ln_volume, short, max_b=5000):
    norm_s = np.linalg.norm(short)
    for beta in range(50, max_b):
        if is_solveable(beta, dim, ln_volume, norm_s):
            if beta > 50:
                return beta
            else:
                return beta
    return None


def is_solveable(beta, dim, log_volume, norm_s):
    left = np.sqrt(beta / dim) * norm_s
    delta = delta_beta(beta)
    right = pow(delta, 2 * beta - dim - 1) * np.exp(log_volume / dim)
    return left <= right


def delta_beta(beta):
    delta = pow(np.pi * beta, 1 / beta)
    delta *= beta / (2 * np.pi * np.e)
    delta = pow(delta, 1 / (2 * beta - 2))
    return delta

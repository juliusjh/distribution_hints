from util import flatten, reduce_sym, reduce_sym_list
import python_kyber768 as python_kyber
import numpy as np


def compress_decompress(p):
    if type(p).__name__ == "Polyvec":
        return python_kyber.Polyvec.from_bytes_compressed(p.to_bytes_compressed())
    elif type(p).__name__ == "Poly":
        return python_kyber.Poly.from_bytes_compressed(p.to_bytes_compressed())
    raise ValueError("{} it not a poly or a polyvec".format(type(p)))


def calc_delta_u(sample):
    r = sample.r.ntt()
    u = r.apply_matrix_left_ntt(transpose(sample.pk.a)).intt() + sample.e1
    u = u.reduce()
    u_uncompressed = u
    u = compress_decompress(u)
    deltau = u - u_uncompressed
    assert sample.ct.b.to_lists() == u.to_lists()
    return deltau


def calc_delta_v(sample):
    v = python_kyber.Polyvec.scalar(sample.pk.pk, sample.r.ntt()).intt() + sample.e2
    v = v + python_kyber.Poly.from_msg(sample.nu)
    v = v.reduce()
    v_uncompressed = v
    v = compress_decompress(v)
    deltav = v - v_uncompressed
    return deltav


def calc_error_term(sample):
    msg = sample.get_msg()
    su = python_kyber.Polyvec.scalar(sample.sk.sk, sample.ct.b.ntt()).intt()
    rec = (sample.ct.v - su).reduce()
    assert sample.nu == rec.to_msg()
    r = rec - msg
    return r, rec


def transpose(A):
    Alist = [A[i].to_list() for i in range(len(A))]
    return [
        python_kyber.Polyvec.new_from_list([Alist[i][j] for i in range(len(A))])
        for j in range(len(A))
    ]


def sample_to_lwe(sample):
    k = len(sample.pk.a)

    def a_map(p):
        return p.intt().montgomery_reduce().to_list()

    a = [[a_map(sample.pk.a[i].to_list()[j]) for j in range(k)] for i in range(k)]
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


def extract_error_term_coefficients(sample, coeff_index=0, max_delta_v=None):
    delta_u = calc_delta_u(sample)
    delta_v = calc_delta_v(sample)

    if max_delta_v is not None and abs(sample.e2.to_list()[coeff_index] + delta_v.to_list()[coeff_index]) >= max_delta_v:
        return None, None

    def sign(i, j):
        return 1 if ((i % 256) + (j % 256)) < 256 else -1

    e_list = [
        sign(i, 256 - i + coeff_index) *
        sample.r.to_lists()[j][(256 - i + coeff_index) % 256]
        for j in range(python_kyber.KyberConstants.K())
        for i in range(256)
    ]
    e1_list = [
        sign(i, 256 - i + coeff_index) *
        sample.e1.to_lists()[j][(256 - i + coeff_index) % 256]
        for j in range(python_kyber.KyberConstants.K())
        for i in range(256)
    ]
    du_list = [
        sign(i, 256 - i + coeff_index) *
        delta_u.to_lists()[j][(256 - i + coeff_index) % 256]
        for j in range(python_kyber.KyberConstants.K())
        for i in range(256)
    ]
    s_list = [-(duj + e1j) for duj, e1j in zip(e1_list, du_list)]

    row = e_list + s_list
    b = reduce_sym(sample.e2.to_list()[coeff_index] + delta_v.to_list()[coeff_index])
    row = reduce_sym_list(row)
    return row, b

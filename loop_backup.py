import numpy as np
# from scipy import linalg
from numpy import linalg
from time import time
# import torch
from numba import jit


def _eig_inv(x, rank):
    """Compute a pseudoinverse with smallest component set to zero."""
    U, s, V = linalg.svd(x)

    # pseudoinverse is computed by setting eigenvalues not included in
    # signalspace to zero
    s_inv = np.zeros(s.shape)
    s_inv[:rank] = 1. / s[:rank]

    x_inv = np.dot(V.T, s_inv[:, np.newaxis] * U.T)
    return x_inv


def stacked_power_iteration(A):
    # Ideally choose a random vector
    # To decrease the chance that our vector
    # Is orthogonal to the eigenvector
    n_src = int(A.shape[0] / 3)

    b1 = np.random.rand(n_src)
    b2 = np.random.rand(n_src)
    b3 = np.random.rand(n_src)

    A1 = np.ascontiguousarray(A[:, 0].reshape([n_src, 3]).T)
    A2 = np.ascontiguousarray(A[:, 1].reshape([n_src, 3]).T)
    A3 = np.ascontiguousarray(A[:, 2].reshape([n_src, 3]).T)

    temp = np.zeros_like(A1)
    # temp_prev = np.ones_like(A1)

    # ABS_TOL = 1e-6


    # while np.linalg.norm(temp_prev - temp, ord='fro') > ABS_TOL:
    for _ in range(100):
        # calculate the matrix-by-vector product Ab
        # temp_prev = temp
        temp = A1 * b1 + A2 * b2 + A3 * b3

        # calculate the norm
        b_norm = np.linalg.norm(temp, axis=0)
        temp /= b_norm

        b1 = temp[0, :]
        b2 = temp[1, :]
        b3 = temp[2, :]

    return temp.flatten('F')


# @profile
def beam_loop(W, G, Cm_inv_sq, Cm_inv, is_free_ori, pick_ori, weight_norm, reduce_rank, noise):
    # Compute spatial filters
    # tprev = time()
    n_orient = 3 if is_free_ori else 1
    n_sources = G.shape[1] // n_orient
    # tprev = time()
    # TMP = np.dot(G.T, np.dot(Cm_inv_sq, G))
    TMP = np.dot(G.T, Cm_inv_sq)
    # print('big product: {:.3f}'.format((time() - tprev) * 1000))
    G = np.asfortranarray(G)
    # W = np.asfortranarray(W)
    print(W.shape)
    max_ori = np.empty(n_orient * n_sources, order='F')
    pwr = np.empty(n_sources, order='F')
    tmp_prod = np.empty((3 * n_sources, 3))

    # tstart = time()
    # print('{:.3f}'.format((tstart - tprev) * 1000))
    for k in range(n_sources):
        # tprev = time()
        Wk = W[n_orient * k: n_orient * k + n_orient, :]
        # print(Wk.shape)
        # print('29: {:.3f}'.format((time() - tprev) * 1000))

        # tprev = time()
        Gk = G[:, n_orient * k: n_orient * k + n_orient]
        # print('32: {:.3f}'.format((time() - tprev) * 1000))

        # tprev = time()
        # tmp = np.dot(Gk.T, np.dot(Cm_inv_sq, Gk))
        tmp = TMP[n_orient * k: n_orient * k + n_orient, :] @ Gk
        # print('35: {:.3f}'.format((time() - tprev) * 1000))

        # tprev = time()
        tmp_1 = np.dot(Wk, Gk)
        # print('37: {:.3f}'.format((time() - tprev) * 1000))

        # tprev = time()
        tmp_prod_temp = linalg._umath_linalg.solve(tmp, tmp_1)
        tmp_prod[n_orient * k: n_orient * (k + 1), :] = tmp_prod_temp
        # tmp_prod = linalg._umath_linalg.solve(tmp, tmp_1)
        # print('38: {:.3f}'.format((time() - tprev) * 1000))

        # tprev = time()
        # eig_vals, eig_vecs = linalg._umath_linalg.eig(tmp_prod_temp)
        # eig_vals, eig_vecs = eig_vals.real, eig_vecs.real
        # import ipdb; ipdb.set_trace()
        # # print('42: {:.3f}'.format((time() - tprev) * 1000))

        # # tprev = time()
        # idx_max = eig_vals.argmax()
        # # print('45: {:.3f}'.format((time() - tprev) * 1000))

        # # tprev = time()
        # max_ori_temp = eig_vecs[:, idx_max]
        # pwr[k] = np.dot(max_ori_temp.T, np.dot(tmp, max_ori_temp))
        # pwr[k] = 1
        # max_ori[n_orient * k : n_orient * (k + 1)] = max_ori_temp
        # print('48: {:.3f}'.format((time() - tprev) * 1000))

        # tprev = time()
        # print('51: {:.3f}'.format((time() - tprev) * 1000))

        # tprev = time()
        # Gk = np.dot(Gk, max_ori)
        # print('54: {:.3f}'.format((time() - tprev) * 1000))

        # tprev = time()
        # tmp = np.dot(Gk.T, np.dot(Cm_inv_sq, Gk))
        # print('58: {:.3f}'.format((time() - tprev) * 1000))

        # tprev = time()
        # print('61: {:.3f}'.format((time() - tprev) * 1000))

        # tprev = time()
        # print('74: {:.3f}'.format((time() - tprev) * 1000))
    # Wk[:] = np.dot(max_ori, Wk)
    max_ori = stacked_power_iteration(tmp_prod)
    # import ipdb; ipdb.set_trace()
    # W_tmp = np.expand_dims(max_ori, axis=1) * W
    # W = W_tmp[::3, :] + W_tmp[1::3, :] + W_tmp[2::3, :]
    W = multiply_by_orientations_rowwise(W, max_ori)
    G_or = multiply_by_orientations_columnwise(G, max_ori)
    TMP_or = multiply_by_orientations_rowwise(TMP, max_ori)
    # pwr_mat = TMP_or @ G_or
    pwr = np.array([TMP_or[k,:] @ G_or[:,k] for k in range(n_sources)])
    # pwr = np.diag(pwr_mat)

    denom = np.sqrt(pwr)
    print('Denom shape = ')
    print(denom.shape)
    W /= np.expand_dims(denom, axis=1)
    print(W.shape)

    is_free_ori = False
    # tend = time()
    # print('{:.3f}'.format((tend - tstart) * 1000))
    return W, is_free_ori


def multiply_by_orientations_rowwise(A, max_ori):
    A_tmp = np.expand_dims(max_ori, axis=1) * A
    A = A_tmp[::3, :] + A_tmp[1::3, :] + A_tmp[2::3, :]
    return A


def multiply_by_orientations_columnwise(A, max_ori):
    # A_tmp = A * np.expand_dims(max_ori, axis=1)
    A_tmp = A * max_ori
    A = A_tmp[:, ::3] + A_tmp[:, 1::3] + A_tmp[:, 2::3]
    return A

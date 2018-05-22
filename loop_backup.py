import numpy as np
# from scipy import linalg
from numpy import linalg
from time import time
import torch
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


@jit
def beam_loop(W, G, Cm_inv_sq, Cm_inv, is_free_ori, pick_ori, weight_norm, reduce_rank, noise):
    # Compute spatial filters
    # tprev = time()
    n_orient = 3 if is_free_ori else 1
    n_sources = G.shape[1] // n_orient
    # tprev = time()
    # TMP = np.dot(G.T, np.dot(Cm_inv_sq, G))
    TMP = np.dot(G.T, Cm_inv_sq)
    # print('big product: {:.3f}'.format((time() - tprev) * 1000))

    # tstart = time()
    # print('{:.3f}'.format((tstart - tprev) * 1000))
    for k in range(n_sources):
        # tprev = time()
        Wk = W[n_orient * k: n_orient * k + n_orient]
        # print('29: {:.3f}'.format((time() - tprev) * 1000))

        # tprev = time()
        Gk = G[:, n_orient * k: n_orient * k + n_orient]
        # print('32: {:.3f}'.format((time() - tprev) * 1000))

        # tprev = time()
        # tmp = np.dot(Gk.T, np.dot(Cm_inv_sq, Gk))
        tmp = TMP[n_orient * k: n_orient * k + n_orient, :] @ Gk
        # print('35: {:.3f}'.format((time() - tprev) * 1000))

        # tprev = time()
        tmp_inv = linalg.inv(tmp)
        # print('38: {:.3f}'.format((time() - tprev) * 1000))

        # tprev = time()
        eig_vals, eig_vecs = linalg.eig(np.dot(tmp_inv, np.dot(Wk, Gk)))
        # print('42: {:.3f}'.format((time() - tprev) * 1000))

        # tprev = time()
        idx_max = eig_vals.argmax()
        # print('45: {:.3f}'.format((time() - tprev) * 1000))

        # tprev = time()
        max_ori = eig_vecs[:, idx_max]
        # print('48: {:.3f}'.format((time() - tprev) * 1000))

        # tprev = time()
        Wk[:] = np.dot(max_ori, Wk)
        # print('51: {:.3f}'.format((time() - tprev) * 1000))

        # tprev = time()
        Gk = np.dot(Gk, max_ori)
        # print('54: {:.3f}'.format((time() - tprev) * 1000))

        # tprev = time()
        tmp = np.dot(Gk.T, np.dot(Cm_inv_sq, Gk))
        # print('58: {:.3f}'.format((time() - tprev) * 1000))

        # tprev = time()
        denom = np.sqrt(tmp)
        # print('61: {:.3f}'.format((time() - tprev) * 1000))

        # tprev = time()
        Wk /= denom
        # print('74: {:.3f}'.format((time() - tprev) * 1000))

    is_free_ori = False
    tend = time()
    # print('{:.3f}'.format((tend - tstart) * 1000))
    return is_free_ori

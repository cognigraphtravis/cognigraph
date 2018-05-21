import numpy as np
from scipy import linalg


def _eig_inv(x, rank):
    """Compute a pseudoinverse with smallest component set to zero."""
    U, s, V = linalg.svd(x)

    # pseudoinverse is computed by setting eigenvalues not included in
    # signalspace to zero
    s_inv = np.zeros(s.shape)
    s_inv[:rank] = 1. / s[:rank]

    x_inv = np.dot(V.T, s_inv[:, np.newaxis] * U.T)
    return x_inv


def beam_loop(W, G, Cm_inv_sq, Cm_inv, is_free_ori, pick_ori, weight_norm, reduce_rank, noise):
    # Compute spatial filters
    n_orient = 3 if is_free_ori else 1
    n_sources = G.shape[1] // n_orient
    for k in range(n_sources):
        Wk = W[n_orient * k: n_orient * k + n_orient]
        Gk = G[:, n_orient * k: n_orient * k + n_orient]
        if np.all(Gk == 0.):
            continue
        Ck = np.dot(Wk, Gk)

        # Compute scalar beamformer by finding the source orientation which
        # maximizes output source power
        if pick_ori == 'max-power':
            # weight normalization and orientation selection:
            if weight_norm is not None and pick_ori == 'max-power':
                # finding optimal orientation for NAI and unit-noise-gain
                # based on [2]_, Eq. 4.47
                tmp = np.dot(Gk.T, np.dot(Cm_inv_sq, Gk))

                if reduce_rank:
                    # use pseudo inverse computation setting smallest component
                    # to zero if the leadfield is not full rank
                    tmp_inv = _eig_inv(tmp, tmp.shape[0] - 1)
                else:
                    # use straight inverse with full rank leadfield
                    try:
                        tmp_inv = linalg.inv(tmp)
                    except np.linalg.linalg.LinAlgError:
                        raise ValueError('Singular matrix detected when '
                                         'estimating LCMV filters. Consider '
                                         'reducing the rank of the leadfield '
                                         'by using reduce_rank=True.')

                eig_vals, eig_vecs = linalg.eig(np.dot(tmp_inv,
                                                       np.dot(Wk, Gk)))

                if np.iscomplex(eig_vecs).any():
                    raise ValueError('The eigenspectrum of the leadfield at '
                                     'this voxel is complex. Consider '
                                     'reducing the rank of the leadfield by '
                                     'using reduce_rank=True.')

                idx_max = eig_vals.argmax()
                max_ori = eig_vecs[:, idx_max]
                Wk[:] = np.dot(max_ori, Wk)
                Gk = np.dot(Gk, max_ori)

                # compute spatial filter for NAI or unit-noise-gain
                tmp = np.dot(Gk.T, np.dot(Cm_inv_sq, Gk))
                denom = np.sqrt(tmp)
                Wk /= denom
                if weight_norm == 'nai':
                    Wk /= np.sqrt(noise)

                is_free_ori = False

            # no weight-normalization and max-power is not implemented yet:
            else:
                raise NotImplementedError('The max-power orientation '
                                          'selection is not yet implemented '
                                          'with weight_norm set to None.')

        else:  # do vector beamformer
            # compute the filters:
            if is_free_ori:
                # Free source orientation
                Wk[:] = np.dot(linalg.pinv(Ck, 0.1), Wk)
            else:
                # Fixed source orientation
                Wk /= Ck

            # handle noise normalization with free/normal source orientation:
            if weight_norm == 'nai':
                raise NotImplementedError('Weight normalization with neural '
                                          'activity index is not implemented '
                                          'yet with free or fixed '
                                          'orientation.')

            if weight_norm == 'unit-noise-gain':
                noise_norm = np.sum(Wk ** 2, axis=1)
                if is_free_ori:
                    noise_norm = np.sum(noise_norm)
                noise_norm = np.sqrt(noise_norm)
                if noise_norm == 0.:
                    noise_norm_inv = 0  # avoid division by 0
                else:
                    noise_norm_inv = 1. / noise_norm
                Wk[:] *= noise_norm_inv
    return is_free_ori

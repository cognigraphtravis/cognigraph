import numpy as np
from copy import deepcopy
from numpy import linalg  # works faster than scipy on anaconda

from mne.io.pick import (pick_types, pick_channels_forward, pick_channels_cov, pick_info)
from mne.cov import compute_whitener
from mne.source_space import label_src_vertno_sel
from mne.io.constants import FIFF
from mne.minimum_norm.inverse import _get_vertno
from mne.io.proj import make_projector
from mne.channels.channels import _contains_ch_type
from mne.utils import warn, estimate_rank
from loop_backup import beam_loop


def _reg_pinv(x, reg):
    """Compute a regularized pseudoinverse of a square array."""
    if reg == 0:
        covrank = estimate_rank(x, tol='auto', norm=False,
                                return_singular=False)
        if covrank < x.shape[0]:
            warn('Covariance matrix is rank-deficient, but no regularization '
                 'is done.')

    # This adds it to the diagonal without using np.eye
    d = reg * np.trace(x) / len(x)
    x.flat[::x.shape[0] + 1] += d
    return linalg.pinv(x), d


def _eig_inv(x, rank):
    """Compute a pseudoinverse with smallest component set to zero."""
    U, s, V = linalg.svd(x)

    # pseudoinverse is computed by setting eigenvalues not included in
    # signalspace to zero
    s_inv = np.zeros(s.shape)
    s_inv[:rank] = 1. / s[:rank]

    x_inv = np.dot(V.T, s_inv[:, np.newaxis] * U.T)
    return x_inv


def _check_one_ch_type(info, picks, noise_cov):
    """Check number of sensor types and presence of noise covariance matrix."""
    info_pick = pick_info(info, sel=picks)
    ch_types =\
        [_contains_ch_type(info_pick, tt) for tt in ('mag', 'grad', 'eeg')]
    if sum(ch_types) > 1 and noise_cov is None:
        raise ValueError('Source reconstruction with several sensor types '
                         'requires a noise covariance matrix to be '
                         'able to apply whitening.')


def _prepare_beamformer_input(info, forward, label, picks, pick_ori):
    """Input preparation common for all beamformer functions.

    Check input values, prepare channel list and gain matrix. For documentation
    of parameters, please refer to _apply_lcmv.
    """
    is_free_ori = forward['source_ori'] == FIFF.FIFFV_MNE_FREE_ORI

    if pick_ori in ['normal', 'max-power'] and not is_free_ori:
        raise ValueError('Normal or max-power orientation can only be picked '
                         'when a forward operator with free orientation is '
                         'used.')
    if pick_ori == 'normal' and not forward['surf_ori']:
        # XXX eventually this could just call convert_forward_solution
        raise ValueError('Normal orientation can only be picked when a '
                         'forward operator oriented in surface coordinates is '
                         'used.')
    if pick_ori == 'normal' and not forward['src'][0]['type'] == 'surf':
        raise ValueError('Normal orientation can only be picked when a '
                         'forward operator with a surface-based source space '
                         'is used.')
    # Restrict forward solution to selected channels
    info_ch_names = [ch['ch_name'] for ch in info['chs']]
    ch_names = [info_ch_names[k] for k in picks]
    fwd_ch_names = forward['sol']['row_names']
    # Keep channels in forward present in info:
    fwd_ch_names = [ch for ch in fwd_ch_names if ch in info_ch_names]
    # This line takes ~48 milliseconds on kernprof
    # forward = pick_channels_forward(forward, fwd_ch_names, verbose='ERROR')
    picks_forward = [fwd_ch_names.index(ch) for ch in ch_names]

    # Get gain matrix (forward operator)
    if label is not None:
        vertno, src_sel = label_src_vertno_sel(label, forward['src'])

        if is_free_ori:
            src_sel = 3 * src_sel
            src_sel = np.c_[src_sel, src_sel + 1, src_sel + 2]
            src_sel = src_sel.ravel()

        G = forward['sol']['data'][:, src_sel]
    else:
        vertno = _get_vertno(forward['src'])
        G = forward['sol']['data']

    # Apply SSPs
    proj, ncomp, _ = make_projector(info['projs'], fwd_ch_names)

    if info['projs']:
        G = np.dot(proj, G)

    # Pick after applying the projections
    G = G[picks_forward]
    proj = proj[np.ix_(picks_forward, picks_forward)]

    return is_free_ori, ch_names, proj, vertno, G


def _compare_ch_names(names1, names2, bads):
    """Return channel names of common and good channels."""
    ch_names = [ch for ch in names1 if ch not in bads and ch in names2]
    return ch_names


def _setup_picks(info, forward, data_cov=None, noise_cov=None):
    """Return good channels common to forward model and covariance matrices."""
    # get a list of all channel names:
    fwd_ch_names = forward['info']['ch_names']

    # handle channels from forward model and info:
    ch_names = _compare_ch_names(info['ch_names'], fwd_ch_names, info['bads'])

    # inform about excluding channels:
    # if (data_cov is not None and set(info['bads']) != set(data_cov['bads']) and
    #         (len(set(ch_names).intersection(data_cov['bads'])) > 0)):
    #     logger.info('info["bads"] and data_cov["bads"] do not match, '
    #                 'excluding bad channels from both.')
    # if (noise_cov is not None and
    #         set(info['bads']) != set(noise_cov['bads']) and
    #         (len(set(ch_names).intersection(noise_cov['bads'])) > 0)):
    #     logger.info('info["bads"] and noise_cov["bads"] do not match, '
    #                 'excluding bad channels from both.')

    # handle channels from data cov if data cov is not None
    # Note: data cov is supposed to be None in tf_lcmv
    if data_cov is not None:
        ch_names = _compare_ch_names(ch_names, data_cov.ch_names,
                                     data_cov['bads'])

    # handle channels from noise cov if noise cov available:
    if noise_cov is not None:
        ch_names = _compare_ch_names(ch_names, noise_cov.ch_names,
                                     noise_cov['bads'])

    picks = [info['ch_names'].index(k) for k in ch_names if k in
             info['ch_names']]
    return picks


def make_lcmv(info, forward, data_cov, reg=0.05, noise_cov=None, label=None,
              pick_ori=None, rank=None, weight_norm='unit-noise-gain',
              reduce_rank=False, verbose=None):
    """Compute LCMV spatial filter.

    Parameters
    ----------
    info : dict
        The measurement info to specify the channels to include.
        Bad channels in info['bads'] are not used.
    forward : dict
        Forward operator.
    data_cov : Covariance
        The data covariance.
    reg : float
        The regularization for the whitened data covariance.
    noise_cov : Covariance
        The noise covariance. If provided, whitening will be done. Providing a
        noise covariance is mandatory if you mix sensor types, e.g.
        gradiometers with magnetometers or EEG with MEG.
    label : Label
        Restricts the LCMV solution to a given label.
    pick_ori : None | 'normal' | 'max-power'
        If 'normal', rather than pooling the orientations by taking the norm,
        only the radial component is kept. If 'max-power', the source
        orientation that maximizes output source power is chosen.
        If None, the solution depends on the forward model: if the orientation
        is fixed, a scalar beamformer is computed. If the forward model has
        free orientation, a vector beamformer is computed, combining the output
        for all source orientations.
    rank : None | int | dict
        Specified rank of the noise covariance matrix. If None, the rank is
        detected automatically. If int, the rank is specified for the MEG
        channels. A dictionary with entries 'eeg' and/or 'meg' can be used
        to specify the rank for each modality.
    weight_norm : 'unit-noise-gain' | 'nai' | None
        If 'unit-noise-gain', the unit-noise gain minimum variance beamformer
        will be computed (Borgiotti-Kaplan beamformer) [2]_,
        if 'nai', the Neural Activity Index [1]_ will be computed,
        if None, the unit-gain LCMV beamformer [2]_ will be computed.
    reduce_rank : bool
        If True, the rank of the leadfield will be reduced by 1 for each
        spatial location. Setting reduce_rank to True is typically necessary
        if you use a single sphere model for MEG.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    filters | dict
        Beamformer weights.

    Notes
    -----
    The original reference is [1]_.

    References
    ----------
    .. [1] Van Veen et al. Localization of brain electrical activity via
           linearly constrained minimum variance spatial filtering.
           Biomedical Engineering (1997) vol. 44 (9) pp. 867--880
    .. [2] Sekihara & Nagarajan. Adaptive spatial filters for electromagnetic
           brain imaging (2008) Springer Science & Business Media
    """
    picks = _setup_picks(info, forward, data_cov, noise_cov)

    is_free_ori, ch_names, proj, vertno, G = \
        _prepare_beamformer_input(info, forward, label, picks, pick_ori)

    data_cov = pick_channels_cov(data_cov, include=ch_names)
    Cm = data_cov['data']

    # check number of sensor types present in the data
    # This line takes ~23 ms on kernprof
    # _check_one_ch_type(info, picks, noise_cov)

    # apply SSPs
    is_ssp = False
    if info['projs']:
        Cm = np.dot(proj, np.dot(Cm, proj.T))
        is_ssp = True

    if noise_cov is not None:
        # Handle whitening + data covariance
        whitener, _ = compute_whitener(noise_cov, info, picks, rank=rank)
        # whiten the leadfield
        G = np.dot(whitener, G)
        # whiten  data covariance
        Cm = np.dot(whitener, np.dot(Cm, whitener.T))
    else:
        whitener = None

    # Tikhonov regularization using reg parameter d to control for
    # trade-off between spatial resolution and noise sensitivity
    Cm_inv, d = _reg_pinv(Cm.copy(), reg)

    if weight_norm is not None:
        # estimate noise level based on covariance matrix, taking the
        # smallest eigenvalue that is not zero
        noise, _ = linalg.eigh(Cm)
        if rank is not None:
            rank_Cm = rank
        else:
            rank_Cm = estimate_rank(Cm, tol='auto', norm=False,
                                    return_singular=False)
        noise = noise[len(noise) - rank_Cm]

        # use either noise floor or regularization parameter d
        noise = max(noise, d)

        # Compute square of Cm_inv used for weight normalization
        Cm_inv_sq = np.dot(Cm_inv, Cm_inv)

    del Cm

    # leadfield rank and optional rank reduction
    if reduce_rank:
        if not pick_ori == 'max-power':
            raise NotImplementedError('The computation of spatial filters '
                                      'with rank reduction using reduce_rank '
                                      'parameter is only implemented with '
                                      'pick_ori=="max-power".')
        if not isinstance(reduce_rank, bool):
            raise ValueError('reduce_rank has to be True or False '
                             ' (got %s).' % reduce_rank)

    # Compute spatial filters
    W = np.dot(G.T, Cm_inv)
    W, is_free_ori = beam_loop(W, G, Cm_inv_sq, Cm_inv, is_free_ori,
                               pick_ori, weight_norm, reduce_rank, noise)

    filters = dict(weights=W, data_cov=data_cov, noise_cov=noise_cov,
                   whitener=whitener, weight_norm=weight_norm,
                   pick_ori=pick_ori, ch_names=ch_names, proj=proj,
                   is_ssp=is_ssp, vertices=vertno, is_free_ori=is_free_ori,
                   nsource=forward['nsource'], src=deepcopy(forward['src']))

    return filters

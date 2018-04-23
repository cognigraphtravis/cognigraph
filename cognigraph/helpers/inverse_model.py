import os

import numpy as np
import mne
from mne.datasets import sample
from mne.minimum_norm import read_inverse_operator

from .. import MISC_CHANNEL_TYPE
from ..helpers.misc import all_upper

data_path = sample.data_path(verbose='ERROR')
sample_dir = os.path.join(data_path, 'MEG', 'sample')
neuromag_forward_file_path = os.path.join(sample_dir, 'sample_audvis-meg-oct-6-fwd.fif')
standard_1005_forward_file_path = os.path.join(sample_dir, 'sample_1005-eeg-oct-6-fwd.fif')


def _pick_columns_from_matrix(matrix: np.ndarray, output_column_labels: list, input_column_labels: list) -> np.ndarray:
    """
    From matrix take only the columns that correspond to output_column_labels - in the order of the latter.
    :param matrix: each column in matrix has a label (eg. EEG channel name)
    :param output_column_labels: labels that we need
    :param input_column_labels: labels that we have
    :return: np.ndarray with len(output_column_labels) columns and the same number of rows as matrix has.
    """

    # Choose the right columns, put zeros where label is missing
    row_count = matrix.shape[0]
    output_matrix = np.zeros((row_count, len(output_column_labels)))
    indices_in_input, indices_in_output = zip(*  # List of length-two arrays to two tuples
        [(input_column_labels.index(label), idx)
         for idx, label in enumerate(output_column_labels)
         if label in input_column_labels])
    output_matrix[:, indices_in_output] = matrix[:, indices_in_input]
    return output_matrix


def matrix_from_inverse_operator(inverse_operator, mne_info, snr, method) -> np.ndarray:
    # Create a dummy mne.Raw object
    channel_count = mne_info['nchan']
    I = np.identity(channel_count)
    dummy_raw = mne.io.RawArray(data=I, info=mne_info, verbose='ERROR')
    contains_eeg_channels = len(mne.pick_types(mne_info, meg=False, eeg=True)) > 0
    if contains_eeg_channels:
        dummy_raw.set_eeg_reference(verbose='ERROR')

    # Applying inverse modelling to an identity matrix gives us the inverse model matrix
    lambda2 = 1.0 / snr ** 2
    stc = mne.minimum_norm.apply_inverse_raw(dummy_raw, inverse_operator, lambda2, method,
                                             verbose='ERROR')

    return stc.data


def get_default_forward_file(mne_info: mne.Info):
    """
    Based on the labels of channels in mne_info return either neuromag or standard 1005 forward model file
    :param mne_info - mne.Info instance
    :return: str: path to the forward-model file
    """
    channel_labels_upper = all_upper(mne_info['ch_names'])

    if max(label.startswith('MEG ') for label in channel_labels_upper) is True:
        return neuromag_forward_file_path

    else:
        montage_1005 = mne.channels.read_montage(kind='standard_1005')
        montage_labels_upper = all_upper(montage_1005.ch_names)
        if any([label_upper in montage_labels_upper for label_upper in channel_labels_upper]):
            return standard_1005_forward_file_path


def assemble_gain_matrix(forward_model_path: str, mne_info: mne.Info, force_fixed=True, drop_missing=False):
    """
    Assemble the gain matrix from the forward model so that its rows correspond to channels in mne_info
    :param force_fixed: whether to return the gain matrix that uses fixed orientations of dipoles
    :param drop_missing: what to do with channels that are not in the forward solution? If False, zero vectors will be
    returned for them, if True, they will not be represented in the returned matrix.
    :param forward_model_path:
    :param mne_info:
    :return: np.ndarray with as many rows as there are dipoles in the forward model and as many rows as there are
    channels in mne_info (well, depending on drop_missing). It drop_missing is True, then also returns indices of
    channels that are both in the forward solution and mne_info
    """

    # Get the gain matrix from the forward solution
    forward = mne.read_forward_solution(forward_model_path, verbose='ERROR')
    if force_fixed is True:
        mne.convert_forward_solution(forward, force_fixed=force_fixed, copy=False, verbose='ERROR')
    G_forward = forward['sol']['data']

    # Take only the channels present in mne_info
    channel_labels_upper = all_upper(mne_info['ch_names'])
    channel_labels_forward = all_upper(forward['info']['ch_names'])
    if drop_missing is True:  # Take only the channels that are both in mne_info and the forward solution
        channel_labels_intersection = [label for label in channel_labels_upper if label in channel_labels_forward]
        channel_indices = [channel_labels_upper.index(label) for label in channel_labels_intersection]
        return (_pick_columns_from_matrix(G_forward.T, channel_labels_intersection, channel_labels_forward).T,
                channel_indices)

    else:
        return _pick_columns_from_matrix(G_forward.T, channel_labels_upper,
                                         channel_labels_forward).T


def make_inverse_operator(forward_model_file_path, mne_info, sigma2=1):
    # sigma2 is what will be used to scale the identity covariance matrix.
    # This will not affect MNE solution though.
    # The inverse operator will use channels common to forward_model_file_path and mne_info.
    forward = mne.read_forward_solution(forward_model_file_path, verbose='ERROR')
    cov = mne.Covariance(data=sigma2 * np.identity(mne_info['nchan']),
                         names=mne_info['ch_names'], bads=mne_info['bads'],
                         projs=mne_info['projs'], nfree=1)

    # return mne.minimum_norm.make_inverse_operator(mne_info, forward,
    #                                               cov, depth=None, loose=0,
    #                                               fixed=True, verbose='ERROR')
    return mne.minimum_norm.make_inverse_operator(
                mne_info, forward, cov, depth=0.8, loose=1, fixed=False)

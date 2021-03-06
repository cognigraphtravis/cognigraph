import time

from typing import Tuple
import math

from vendor.nfb.pynfb.protocols.ssd.topomap_selector_ica import ICADialog

import numpy as np
import mne
from numpy.linalg import svd
from scipy.optimize import linprog
from sklearn.preprocessing import normalize
from mne.preprocessing import find_outliers
from mne.minimum_norm import apply_inverse_raw  # , make_inverse_operator
from mne.minimum_norm import make_inverse_operator as mne_make_inverse_operator
from mne.beamformer import apply_lcmv_raw
from ..helpers.make_lcmv import make_lcmv

from .node import ProcessorNode
from ..helpers.matrix_functions import (make_time_dimension_second,
                                        put_time_dimension_back_from_second,
                                        last_sample)
from ..helpers.inverse_model import (get_default_forward_file,
                                     get_clean_forward,
                                     make_inverse_operator,
                                     matrix_from_inverse_operator)

from ..helpers.pynfb import (pynfb_ndarray_function_wrapper,
                             ExponentialMatrixSmoother)
from ..helpers.channels import channel_labels_saver
from ..helpers.aux_tools import nostdout
from .. import TIME_AXIS
from vendor.nfb.pynfb.signal_processing import filters


class Preprocessing(ProcessorNode):
    CHANGES_IN_THESE_REQUIRE_RESET = ('collect_for_x_seconds', )
    UPSTREAM_CHANGES_IN_THESE_REQUIRE_REINITIALIZATION = ('mne_info', )
    SAVERS_FOR_UPSTREAM_MUTABLE_OBJECTS = {'mne_info': channel_labels_saver}

    def __init__(self, collect_for_x_seconds: int=60):
        super().__init__()
        self.collect_for_x_seconds = collect_for_x_seconds  # type: int

        self._samples_collected = None  # type: int
        self._samples_to_be_collected = None  # type: int
        self._enough_collected = None  # type: bool
        self._means = None  # type: np.ndarray
        self._mean_sums_of_squares = None  # type: np.ndarray
        self._bad_channel_indices = None  # type: List[int]
        self._interpolation_matrix = None  # type: np.ndarray

        self._reset_statistics()

    def _initialize(self):
        self.mne_info = self.traverse_back_and_find('mne_info')
        frequency = self.mne_info['sfreq']
        self._samples_to_be_collected = int(math.ceil(
            self.collect_for_x_seconds * frequency))

    def _update(self):
        # Have we collected enough samples without the new input?
        enough_collected = self._samples_collected >=\
                self._samples_to_be_collected
        if not enough_collected:
            if self.input_node.output is not None and\
                    self.input_node.output.shape[TIME_AXIS] > 0:
                self._update_statistics()

        elif not self._enough_collected:  # We just got enough samples
            self._enough_collected = True
            standard_deviations = self._calculate_standard_deviations()
            self._bad_channel_indices = find_outliers(standard_deviations)
            if any(self._bad_channel_indices):
                # message = Message(there_has_been_a_change=True,
                #                   output_history_is_no_longer_valid=True)
                # self._deliver_a_message_to_receivers(message)
                # self.mne_info['bads'].append(self._bad_channel_indices)
                # self.mne_info['bads'] = self._bad_channel_indices

                # TODO: handle emergent bad channels on the go
                pass

        self.output = self.input_node.output

    def _reset(self) -> bool:
        self._reset_statistics()
        self._input_history_is_no_longer_valid = True
        return self._input_history_is_no_longer_valid

    def _reset_statistics(self):
        self._samples_collected = 0
        self._enough_collected = False
        self._means = 0
        self._mean_sums_of_squares = 0
        self._bad_channel_indices = []

    def _update_statistics(self):
        input_array = self.input_node.output.astype(np.dtype('float64'))
        # Using float64 is necessary because otherwise rounding error
        # in recursive formula accumulate
        n = self._samples_collected
        m = input_array.shape[TIME_AXIS]  # number of new samples
        self._samples_collected += m

        self._means = (
            self._means * n + np.sum(input_array, axis=TIME_AXIS)) / (n + m)
        self._mean_sums_of_squares = (
            self._mean_sums_of_squares * n +
            np.sum(input_array ** 2, axis=TIME_AXIS)) / (n + m)

    def _calculate_standard_deviations(self):
        n = self._samples_collected
        return np.sqrt(
            n / (n - 1) * (self._mean_sums_of_squares - self._means ** 2))

    def _on_input_history_invalidation(self):
        self._reset_statistics()

    def _check_value(self, key, value):
        pass


class InverseModel(ProcessorNode):
    SUPPORTED_METHODS = ['MNE', 'dSPM', 'sLORETA']
    UPSTREAM_CHANGES_IN_THESE_REQUIRE_REINITIALIZATION = ('mne_info', )
    CHANGES_IN_THESE_REQUIRE_RESET = ('mne_inverse_model_file_path',
                                      'mne_forward_model_file_path',
                                      'snr', 'method')
    SAVERS_FOR_UPSTREAM_MUTABLE_OBJECTS = {'mne_info': channel_labels_saver}

    def __init__(self, forward_model_path=None, snr=1.0, method='MNE'):
        super().__init__()

        self.snr = snr
        self._user_provided_forward_model_file_path = forward_model_path
        self._default_forward_model_file_path = None
        self.mne_info = None
        self.fwd = None

        self._inverse_model_matrix = None
        self.method = method

    def _initialize(self):
        mne_info = self.traverse_back_and_find('mne_info')
        self._bad_channels = mne_info['bads']

        if self._user_provided_forward_model_file_path is None:
            self._default_forward_model_file_path =\
                get_default_forward_file(mne_info)

        self.fwd, missing_ch_names = get_clean_forward(
            self.mne_forward_model_file_path, mne_info)
        mne_info['bads'] = list(set(mne_info['bads'] + missing_ch_names))

        inverse_operator = make_inverse_operator(self.fwd, mne_info)
        self._inverse_model_matrix = matrix_from_inverse_operator(
            inverse_operator=inverse_operator, mne_info=mne_info,
            snr=self.snr, method=self.method)

        frequency = mne_info['sfreq']
        # channel_count = self._inverse_model_matrix.shape[0]
        channel_count = self.fwd['nsource']
        channel_labels = ['vertex #{}'.format(i + 1)
                          for i in range(channel_count)]
        self.mne_info = mne.create_info(channel_labels, frequency)

    def _update(self):
        mne_info = self.traverse_back_and_find('mne_info')
        bads = mne_info['bads']
        if bads != self._bad_channels:
            inverse_operator = make_inverse_operator(self.fwd, mne_info)
            self._inverse_model_matrix = matrix_from_inverse_operator(
                inverse_operator=inverse_operator, mne_info=mne_info,
                snr=self.snr, method=self.method)
            self._bad_channels = bads

        input_array = self.input_node.output
        raw_array = mne.io.RawArray(input_array, mne_info, verbose='ERROR')
        raw_array.pick_types(eeg=True, meg=False, stim=False, exclude='bads')
        data = raw_array.get_data()
        self.output = self._apply_inverse_model_matrix(data)

    def _on_input_history_invalidation(self):
        # The methods implemented in this node do not rely on past inputs
        pass

    def _check_value(self, key, value):
        if key == 'method':
            if value not in self.SUPPORTED_METHODS:
                raise ValueError(
                    'Method {} is not supported.'.format(value) +
                    ' Use one of: {}'.format(self.SUPPORTED_METHODS))

        if key == 'snr':
            if value <= 0:
                raise ValueError(
                    'snr (signal-to-noise ratio) must be a positive number.')

    def _reset(self):
        self._should_reinitialize = True
        self.initialize()
        output_history_is_no_longer_valid = True
        return output_history_is_no_longer_valid

    @property
    def mne_forward_model_file_path(self):
        return self._user_provided_forward_model_file_path or\
                self._default_forward_model_file_path

    @mne_forward_model_file_path.setter
    def mne_forward_model_file_path(self, value):
        # This setter is for public use, hence the "user_provided"
        self._user_provided_forward_model_file_path = value

    def _apply_inverse_model_matrix(self, input_array: np.ndarray):
        W = self._inverse_model_matrix  # VERTICES x CHANNELS
        output_array = W.dot(make_time_dimension_second(input_array))
        return put_time_dimension_back_from_second(output_array)


class LinearFilter(ProcessorNode):
    UPSTREAM_CHANGES_IN_THESE_REQUIRE_REINITIALIZATION = ('mne_info', )
    CHANGES_IN_THESE_REQUIRE_RESET = ('lower_cutoff', 'upper_cutoff')
    SAVERS_FOR_UPSTREAM_MUTABLE_OBJECTS = {'mne_info':
                                           lambda info: (info['nchan'], )}

    def __init__(self, lower_cutoff, upper_cutoff):
        super().__init__()
        self.lower_cutoff = lower_cutoff
        self.upper_cutoff = upper_cutoff
        self._linear_filter = None  # type: filters.ButterFilter

    def _initialize(self):
        mne_info = self.traverse_back_and_find('mne_info')
        frequency = mne_info['sfreq']
        channel_count = mne_info['nchan']
        if not (self.lower_cutoff is None and self.upper_cutoff is None):
            band = (self.lower_cutoff, self.upper_cutoff)

            self._linear_filter = filters.ButterFilter(
                band, fs=frequency, n_channels=channel_count)

            self._linear_filter.apply = pynfb_ndarray_function_wrapper(
                self._linear_filter.apply)
        else:
            self._linear_filter = None

    def _update(self):
        input = self.input_node.output
        if self._linear_filter is not None:
            self.output = self._linear_filter.apply(input)
        else:
            self.output = input

    def _check_value(self, key, value):
        if value is None:
            pass

        elif key == 'lower_cutoff':
            if (hasattr(self, 'upper_cutoff') and
                    self.upper_cutoff is not None and
                    value > self.upper_cutoff):
                raise ValueError(
                    'Lower cutoff can`t be set higher that the upper cutoff')
            if value < 0:
                raise ValueError('Lower cutoff must be a positive number')

        elif key == 'upper_cutoff':
            if (hasattr(self, 'upper_cutoff') and
                    self.lower_cutoff is not None and
                    value < self.lower_cutoff):
                raise ValueError(
                    'Upper cutoff can`t be set lower that the lower cutoff')
            if value < 0:
                raise ValueError('Upper cutoff must be a positive number')

    def _on_input_history_invalidation(self):
        if self._linear_filter is not None:
            self._linear_filter.reset()

    def _reset(self):
        self._should_reinitialize = True
        self.initialize()
        output_history_is_no_longer_valid = True
        return output_history_is_no_longer_valid


class EnvelopeExtractor(ProcessorNode):
    def __init__(self, factor=0.9):
        super().__init__()
        self.method = 'Exponential smoothing'
        self.factor = factor
        self._envelope_extractor = None  # type: ExponentialMatrixSmoother

    def _initialize(self):
        channel_count = self.traverse_back_and_find('mne_info')['nchan']
        self._envelope_extractor = ExponentialMatrixSmoother(
            factor=self.factor, column_count=channel_count)
        self._envelope_extractor.apply = pynfb_ndarray_function_wrapper(
            self._envelope_extractor.apply)

    def _update(self):
        input = self.input_node.output
        self.output = self._envelope_extractor.apply(np.abs(input))

    def _check_value(self, key, value):
        if key == 'factor':
            if value <= 0 or value >= 1:
                raise ValueError('Factor must be a number between 0 and 1')

        if key == 'method':
            if value not in self.SUPPORTED_METHODS:
                raise ValueError(
                    'Method {} is not supported.' +
                    ' Use one of: {}'.format(value, self.SUPPORTED_METHODS))

    def _reset(self):
        self._should_reinitialize = True
        self.initialize()
        output_history_is_no_longer_valid = True
        return output_history_is_no_longer_valid

    def _on_input_history_invalidation(self):
        self._envelope_extractor.reset()

    UPSTREAM_CHANGES_IN_THESE_REQUIRE_REINITIALIZATION = ('mne_info', )
    CHANGES_IN_THESE_REQUIRE_RESET = ('method', 'factor')
    SUPPORTED_METHODS = ('Exponential smoothing', )
    SAVERS_FOR_UPSTREAM_MUTABLE_OBJECTS = {'mne_info':
                                           lambda info: (info['nchan'],)}


class Beamformer(ProcessorNode):

    SUPPORTED_OUTPUT_TYPES = ('power', 'activation')
    UPSTREAM_CHANGES_IN_THESE_REQUIRE_REINITIALIZATION = ('mne_info',)
    CHANGES_IN_THESE_REQUIRE_RESET = ('snr', 'output_type', 'is_adaptive',
                                      'fixed_orientation',
                                      'mne_forward_model_file_path')

    SAVERS_FOR_UPSTREAM_MUTABLE_OBJECTS = {'mne_info': channel_labels_saver}

    def __init__(self, snr: float=1.0, output_type: str='power',
                 is_adaptive: bool=False, fixed_orientation: bool=True,
                 forward_model_path: str=None,
                 forgetting_factor_per_second: float=0.99):
        super().__init__()

        self.snr = snr  # type: float
        self._user_provided_forward_model_file_path = forward_model_path
        self._default_forward_model_file_path = None  # type: str
        self.mne_info = None  # type: mne.Info

        self.output_type = output_type  # type: np.dtype
        self.is_adaptive = is_adaptive  # type: bool
        self._initialized_as_adaptive = None  # type: bool
        self.fixed_orientation = fixed_orientation  # type: bool
        self._initialized_as_fixed = None  # type: bool

        self._channel_indices = None  # type: list
        self._gain_matrix = None  # type: np.ndarray
        self._Rxx = None  # type: np.ndarray
        self.forgetting_factor_per_second = forgetting_factor_per_second
        self._forgetting_factor_per_sample = None  # type: float

    def _initialize(self):
        mne_info = self.traverse_back_and_find('mne_info')

        if self._user_provided_forward_model_file_path is None:
            self._default_forward_model_file_path = get_default_forward_file(
                    mne_info)

        try:
            fwd, missing_ch_names = get_clean_forward(
                self.mne_forward_model_file_path, mne_info)
        except ValueError:
            raise Exception('BAD FORWARD + DATA COMBINATION!')

        mne_info['bads'] = list(set(mne_info['bads'] + missing_ch_names))
        self._gain_matrix = fwd['sol']['data']
        G = self._gain_matrix
        if self.is_adaptive is False:
            Rxx = G.dot(G.T)
        elif self.is_adaptive is True:
            Rxx = np.zeros([G.shape[0], G.shape[0]])  # G.dot(G.T)

        goods = mne.pick_types(mne_info, eeg=True, meg=False, exclude='bads')
        ch_names = [mne_info['ch_names'][i] for i in goods]

        self._Rxx = mne.Covariance(Rxx, ch_names, mne_info['bads'],
                                   mne_info['projs'], nfree=1)

        self._mne_info = mne_info

        frequency = mne_info['sfreq']
        self._forgetting_factor_per_sample = np.power(
                self.forgetting_factor_per_second, 1 / frequency)

        n_vert = fwd['nsource']
        channel_labels = ['vertex #{}'.format(i + 1) for i in range(n_vert)]
        self.mne_info = mne.create_info(channel_labels, frequency)
        self._initialized_as_adaptive = self.is_adaptive
        self._initialized_as_fixed = self.fixed_orientation

        self.fwd_surf = mne.convert_forward_solution(
                    fwd, surf_ori=True, force_fixed=False)
        if not self.is_adaptive:
            self._filters = make_lcmv(
                    info=self._mne_info, forward=self.fwd_surf,
                    data_cov=self._Rxx, reg=0.05, pick_ori='max-power',
                    weight_norm='unit-noise-gain', reduce_rank=False)
        else:
            self._filters = None

    def _update(self):
        t1 = time.time()
        input_array = self.input_node.output
        raw_array = mne.io.RawArray(
            input_array, self._mne_info, verbose='ERROR')

        raw_array.pick_types(eeg=True, meg=False, stim=False, exclude='bads')
        raw_array.set_eeg_reference(ref_channels='average', projection=True)
        t2 = time.time()
        self.logger.debug('Prepare arrays in {:.1f} ms'.format(
                    (t2 - t1) * 1000))

        if self.is_adaptive:
            self._update_covariance_matrix(input_array)
            t1 = time.time()
            self._filters = make_lcmv(info=self._mne_info,
                                      forward=self.fwd_surf,
                                      data_cov=self._Rxx, reg=0.5,
                                      pick_ori='max-power',
                                      weight_norm='unit-noise-gain',
                                      reduce_rank=False)
            t2 = time.time()
            self.logger.debug('Assembled lcmv instance in {:.1f} ms'.format(
                        (t2 - t1) * 1000))

        self._filters['source_nn'] = []
        t1 = time.time()
        stc = apply_lcmv_raw(raw=raw_array, filters=self._filters,
                             max_ori_out='signed')
        t2 = time.time()
        self.logger.debug('Applied lcmv inverse in {:.1f} ms'.format(
                    (t2 - t1) * 1000))

        output = stc.data
        t1 = time.time()
        if self.fixed_orientation is True:
            if self.output_type == 'power':
                output = output ** 2
        else:
            vertex_count = self.fwd_surf['nsource']
            output = np.sum(
                np.power(output, 2).reshape((vertex_count, 3, -1)), axis=1)
            if self.output_type == 'activation':
                output = np.sqrt(output)

        self.output = output
        t2 = time.time()
        self.logger.debug(
                'Finalized in {:.1f} ms'.format(
                    (t2 - t1) * 1000))

    @property
    def mne_forward_model_file_path(self):
        # TODO: fix this
        return (self._user_provided_forward_model_file_path or
                self._default_forward_model_file_path)

    @mne_forward_model_file_path.setter
    def mne_forward_model_file_path(self, value):
        # This setter is for public use, hence the "user_provided"
        self._user_provided_forward_model_file_path = value

    def _reset(self) -> bool:

        # Only change adaptiveness or fixed_orientation requires reinit
        # if (self._initialized_as_adaptive is not self.is_adaptive
        #         or self._initialized_as_fixed is not self.fixed_orientation):
        self._should_reinitialize = True
        self.initialize()

        output_history_is_no_longer_valid = True
        return output_history_is_no_longer_valid

    def _on_input_history_invalidation(self):
        # Only adaptive version relies on history
        if self._initialized_as_adaptive is True:
            self._should_reinitialize = True
            self.initialize()

    def _check_value(self, key, value):
        if key == 'output_type':
            if value not in self.SUPPORTED_OUTPUT_TYPES:
                raise ValueError(
                    'Method {} is not supported.' +
                    ' Use one of: {}'.format(
                        value, self.SUPPORTED_OUTPUT_TYPES))

        if key == 'snr':
            if value <= 0:
                raise ValueError(
                    'snr (signal-to-noise ratio) must be a positive number')

        if key == 'is_adaptive':
            if not isinstance(value, bool):
                raise ValueError(
                    'Beamformer type (adaptive vs nonadaptive) is not set')

    def _update_covariance_matrix(self, input_array):
        t1 = time.time()
        alpha = self._forgetting_factor_per_sample
        sample_count = input_array.shape[TIME_AXIS]
        self.logger.debug('Number of samples: {}'.format(sample_count))
        new_Rxx_data = self._Rxx.data

        raw_array = mne.io.RawArray(
            input_array, self._mne_info, verbose='ERROR')
        raw_array.pick_types(eeg=True, meg=False, stim=False, exclude='bads')
        raw_array.set_eeg_reference(ref_channels='average', projection=True)
        input_array_nobads = raw_array.get_data()

        t2 = time.time()
        self.logger.debug(
            'Prepared covariance update in {:.2f} ms'.format((t2 - t1) * 1000))
        samples = make_time_dimension_second(input_array_nobads).T
        new_Rxx_data = (alpha * new_Rxx_data +
                        (1 - alpha) * samples.T.dot(samples))
        t3 = time.time()
        self.logger.debug(
            'Updated matrix data in {:.2f} ms'.format((t3 - t2) * 1000))

        self._Rxx = mne.Covariance(new_Rxx_data, self._Rxx.ch_names,
                                   raw_array.info['bads'],
                                   raw_array.info['projs'], nfree=1)
        t4 = time.time()
        self.logger.debug('Created instance of covariance' +
                          ' in {:.2f} ms'.format((t4 - t4) * 1000))


# TODO: implement this function
def pynfb_filter_based_processor_class(pynfb_filter_class):
    """
    Returns a ProcessorNode subclass with the functionality of
    pynfb_filter_class

    pynfb_filter_class: subclass of pynfb.signal_processing.filters.BaseFilter

    Sample usage 1:

    LinearFilter = pynfb_filter_based_processor_class(filters.ButterFilter)
    linear_filter = LinearFilter(band, fs, n_channels, order)

    Sample usage 2
    (this would correspond to a different implementation of this function):

    LinearFilter = pynfb_filter_based_processor_class(filters.ButterFilter)
    linear_filter = LinearFilter(band, order)

    In this case LinearFilter should provide
    fs and n_channels parameters to filters.ButterFilter automatically

    """
    class PynfbFilterBasedProcessorClass(ProcessorNode):
        def _on_input_history_invalidation(self):
            pass

        def _check_value(self, key, value):
            pass

        @property
        def CHANGES_IN_THESE_REQUIRE_RESET(self) -> Tuple[str]:
            pass

        @property
        def UPSTREAM_CHANGES_IN_THESE_REQUIRE_REINITIALIZATION(self) -> Tuple[str]:
            pass

        def _reset(self):
            pass

        def __init__(self):
            pass

        def _initialize(self):
            pass

        def _update(self):
            pass
    return PynfbFilterBasedProcessorClass


class MCE(ProcessorNode):
    input = []
    output = []

    UPSTREAM_CHANGES_IN_THESE_REQUIRE_REINITIALIZATION = ()
    CHANGES_IN_THESE_REQUIRE_RESET = ('mne_forward_model_file_path', 'snr')

    def __init__(self, snr=1.0, forward_model_path=None, n_comp=40):
        super().__init__()
        self.snr = snr
        self.mne_forward_model_file_path = forward_model_path
        self.n_comp = n_comp
        self.mne_info = None
        # pass

    def _initialize(self):
        print('INITIALIZING MCE NODE ...')
        mne_info = self.traverse_back_and_find('mne_info')
        # mne_info['custom_ref_applied'] = True
        # -------- truncated svd for fwd_opr operator -------- #
        fwd, missing_ch_names = get_clean_forward(
            self.mne_forward_model_file_path, mne_info)
        mne_info['bads'] = list(set(mne_info['bads'] + missing_ch_names))
        fwd_fix = mne.convert_forward_solution(
                fwd, surf_ori=True, force_fixed=False)

        self._gain_matrix = fwd_fix['sol']['data']

        print('MCE: COMPUTING SVD OF THE FORWARD OPERATOR')
        U, S, V = svd(self._gain_matrix)

        Sn = np.zeros([self.n_comp, V.shape[0]])
        Sn[:self.n_comp, :self.n_comp] = np.diag(S[:self.n_comp])

        self.Un = U[:, :self.n_comp]
        self.A_non_ori = Sn @ V
        # ---------------------------------------------------- #

        # -------- leadfield dims -------- #
        N_SEN = self._gain_matrix.shape[0]
        # -------------------------------- #

        # ------------------------ noise-covariance ------------------------ #
        cov_data = np.identity(N_SEN)
        ch_names = np.array(mne_info['ch_names'])[mne.pick_types(mne_info,
                                                                 eeg=True,
                                                                 meg=False)]
        ch_names = list(ch_names)
        noise_cov = mne.Covariance(
                cov_data, ch_names, mne_info['bads'],
                mne_info['projs'], nfree=1)
        # ------------------------------------------------------------------ #

        self.mne_inv = mne_make_inverse_operator(
                mne_info, fwd_fix, noise_cov, depth=0.8,
                loose=1, fixed=False, verbose='ERROR')
        self.mne_info = mne_info
        self.Sn = Sn
        self.V = V

    def _update(self):
        input_array = self.input_node.output
        last_slice = last_sample(input_array)
        n_src = self.mne_inv['nsource']
        n_times = input_array.shape[1]
        output_mce = np.empty([n_src, n_times])

        raw_slice = mne.io.RawArray(np.expand_dims(last_slice, axis=1),
                                    self.mne_info, verbose='ERROR')
        raw_slice.pick_types(eeg=True, meg=False, stim=False, exclude='bads')
        raw_slice.set_eeg_reference(ref_channels='average', projection=True)

        # ------------------- get dipole orientations --------------------- #
        stc_slice = apply_inverse_raw(raw_slice, self.mne_inv,
                                      pick_ori='vector',
                                      method='MNE', lambda2=1, verbose='ERROR')
        Q = normalize(stc_slice.data[:, :, 0])  # dipole orientations
        # ----------------------------------------------------------------- #

        # -------- setup linprog params -------- #
        n_sen = self.A_non_ori.shape[0]
        A_eq = np.empty([n_sen, n_src])
        for i in range(n_src):
            A_eq[:, i] = self.A_non_ori[:, i * 3: (i + 1) * 3] @ Q[i, :].T
        data_slice = raw_slice.get_data()[:, 0]
        b_eq = self.Un.T @ data_slice
        c = np.ones(A_eq.shape[1])
        # -------------------------------------- #

        with nostdout():
            sol = linprog(c, A_eq=A_eq, b_eq=b_eq,
                          method='interior-point', bounds=(0, None),
                          options={'disp': False})
        output_mce[:, :] = sol.x[:, np.newaxis]

        self.output = output_mce
        self.sol = sol
        return Q, A_eq, data_slice, b_eq, c

    def _on_input_history_invalidation(self):
        # The methods implemented in this node do not rely on past inputs
        pass

    def _reset(self):
        self._should_reinitialize = True
        self.initialize()
        output_history_is_no_longer_valid = True
        return output_history_is_no_longer_valid

    def _check_value(self, key, value):
        if key == 'snr':
            if value <= 0:
                raise ValueError(
                    'snr (signal-to-noise ratio) must be a positive number.')


class ICARejection(ProcessorNode):

    def __init__(self, collect_for_x_seconds: int=60):
        super().__init__()
        self.collect_for_x_seconds = collect_for_x_seconds  # type: int

        self._samples_collected = None  # type: int
        self._samples_to_be_collected = None  # type: int
        self._enough_collected = None  # type: bool

        self._reset_statistics()
        self._ica_rejector = None

    def _on_input_history_invalidation(self):
        self._reset_statistics()

    def _check_value(self, key, value):
        pass

    CHANGES_IN_THESE_REQUIRE_RESET = ('collect_for_x_seconds', )

    def _initialize(self):
        self._mne_info = self.traverse_back_and_find('mne_info')
        self._frequency = self._mne_info['sfreq']
        self._good_ch_inds = mne.pick_types(self._mne_info, eeg=True,
                                            meg=False, stim=False,
                                            exclude='bads')

        channels = self._mne_info['chs']
        self._ch_locs = np.array([ch['loc'] for ch in channels])

        n_ch = len(self._good_ch_inds)
        self._samples_to_be_collected = int(math.ceil(
            self.collect_for_x_seconds * self._frequency))
        self._collected_timeseries = np.zeros(
                [n_ch, self._samples_to_be_collected])
        self._linear_filter = filters.ButterFilter(
                [1, 200], fs=self._frequency,
                n_channels=len(self._good_ch_inds))
        self._linear_filter.apply = pynfb_ndarray_function_wrapper(
                self._linear_filter.apply)

    def _reset(self) -> bool:
        self._reset_statistics()
        self._input_history_is_no_longer_valid = True
        return self._input_history_is_no_longer_valid

    def _reset_statistics(self):
        self._samples_collected = 0
        self._enough_collected = False

    def _update(self):
        # Have we collected enough samples without the new input?
        self.output = self.input_node.output

        enough_collected = self._samples_collected >=\
            self._samples_to_be_collected
        if not enough_collected:
            if self.input_node.output is not None and\
                    self.input_node.output.shape[TIME_AXIS] > 0:
                self._update_statistics()

        elif not self._enough_collected:  # We just got enough samples
            self._enough_collected = True
            print('COLLECTED ENOUGH SAMPLES')
            ica = ICADialog(
                self._collected_timeseries.T,
                list(np.array(self._mne_info['ch_names'])[self._good_ch_inds]),
                self._ch_locs[self._good_ch_inds, :], self._frequency)

            ica.exec_()
            self._ica_rejector = ica.rejection.val.T
        else:
            self.output[self._good_ch_inds, :] = np.dot(
                    self._ica_rejector,
                    self.input_node.output[self._good_ch_inds, :])

    def _update_statistics(self):
        input_array = self.input_node.output.astype(np.dtype('float64'))
        n = self._samples_collected
        m = input_array.shape[TIME_AXIS]  # number of new samples
        self._samples_collected += m
        self._collected_timeseries[:, n:n + m] = self._linear_filter.apply(
                input_array[self._good_ch_inds, :])
        # Using float64 is necessary because otherwise rounding error
        # in recursive formula accumulate
        pass

    UPSTREAM_CHANGES_IN_THESE_REQUIRE_REINITIALIZATION = ('mne_info', )
    SAVERS_FOR_UPSTREAM_MUTABLE_OBJECTS = {'mne_info': channel_labels_saver}

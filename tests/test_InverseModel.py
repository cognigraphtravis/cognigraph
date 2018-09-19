import pytest
# from nose.tools import assert_equals, raises
from cognigraph.nodes.processors import InverseModel
from cognigraph.nodes.sources import FifSource
import os.path as op
import numpy as np
from mne.io import read_info

test_data_path = op.join(op.dirname(__file__),  'data')


@pytest.fixture
def inv_model():
    snr = 1
    fwd_model_path = op.join(test_data_path, 'dmalt_custom_lr-fwd.fif')
    info_src_path = op.join(test_data_path, 'Koleno.fif')
    info = read_info(info_src_path)
    method = 'MNE'
    inv_model = InverseModel(
        snr=snr, forward_model_path=fwd_model_path, method=method)
    inv_model.mne_info = info
    N_SEN = len(info['ch_names'])
    inv_model.input = np.random.rand(N_SEN)
    input_node = FifSource()
    input_node.output = np.random.rand(info['nchan'], 1)
    input_node.mne_info = info
    inv_model.input_node = input_node
    return inv_model


@pytest.fixture
def inv_model_def():
    info_src_path = op.join(test_data_path, 'Koleno.fif')
    info = read_info(info_src_path)
    inv_model_def = InverseModel()
    input_node = FifSource()
    input_node.mne_info = info
    input_node.output = np.random.rand(info['nchan'], 1)
    inv_model_def.input_node = input_node
    return inv_model_def


def test_defaults(inv_model_def):
    assert(inv_model_def.mne_forward_model_file_path is None)
    assert(inv_model_def.mne_info is None)


def test_initialize(inv_model):
    inv_model.initialize()


def test_reset(inv_model):
    out_hist = inv_model._reset()
    # assert(self.inv_model._should_reinitialize == True)
    assert(out_hist == True)


def test_update(inv_model):
    inv_model._initialize()
    inv_model._update()


def test_check_value(inv_model):
    with pytest.raises(ValueError):
        inv_model.snr = -1

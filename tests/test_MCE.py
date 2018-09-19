import pytest
# from nose.tools import assert_equals, raises
# from scripts.mce import MCE
from cognigraph.nodes.processors import MCE
from cognigraph.nodes.sources import FileSource
import os.path as op
import numpy as np
from mne.io import read_info

test_data_path = op.join(op.dirname(__file__),  'data')


@pytest.fixture
def mce():
    snr = 1
    fwd_model_path = op.join(test_data_path, 'dmalt_custom_lr-fwd.fif')
    info_src_path = op.join(test_data_path, 'Koleno.fif')
    info = read_info(info_src_path)
    n_comp = 10
    mce = MCE(snr, fwd_model_path, n_comp)
    mce.mne_info = info
    N_SEN = len(info['ch_names'])
    mce.input = np.random.rand(N_SEN)
    input_node = FileSource()
    input_node.output = np.random.rand(info['nchan'], 1)
    input_node.mne_info = info
    mce.input_node = input_node
    return mce


@pytest.fixture
def mce_def():
    info_src_path = op.join(test_data_path, 'Koleno.fif')
    info = read_info(info_src_path)
    mce_def = MCE()
    input_node = FileSource()
    input_node.mne_info = info
    input_node.output = np.random.rand(info['nchan'], 1)
    mce_def.input_node = input_node
    return mce_def


def test_defaults(mce_def):
    assert(mce_def.mne_forward_model_file_path is None)
    assert(mce_def.mne_info is None)


def test_initialize(mce):
    mce.initialize()


def test_reset(mce):
    out_hist = mce._reset()
    # assert(self.mce._should_reinitialize == True)
    assert(out_hist == True)


def test_update(mce):
    mce._initialize()
    mce._update()


def test_check_value(mce):
    with pytest.raises(ValueError):
        mce.snr = -1

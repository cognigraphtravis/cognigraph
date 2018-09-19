import pytest
# from nose.tools import assert_equals, raises
# from scripts.beamformer import MCE
from cognigraph.nodes.processors import Beamformer
from cognigraph.nodes.sources import FileSource
import os.path as op
import numpy as np
from mne.io import read_info

test_data_path = op.join(op.dirname(__file__),  'data')


@pytest.fixture
def beamformer():
    snr = 1
    fwd_model_path = op.join(test_data_path, 'dmalt_custom_lr-fwd.fif')
    info_src_path = op.join(test_data_path, 'Koleno.fif')
    info = read_info(info_src_path)
    is_adaptive = True
    beamformer = Beamformer(
        snr=snr, forward_model_path=fwd_model_path, is_adaptive=is_adaptive)
    beamformer.mne_info = info
    N_SEN = len(info['ch_names'])
    beamformer.input = np.random.rand(N_SEN)
    input_node = FileSource()
    input_node.output = np.random.rand(info['nchan'], 1)
    input_node.mne_info = info
    beamformer.input_node = input_node
    return beamformer


@pytest.fixture
def beamformer_def():
    info_src_path = op.join(test_data_path, 'Koleno.fif')
    info = read_info(info_src_path)
    beamformer_def = Beamformer()
    input_node = FileSource()
    input_node.mne_info = info
    input_node.output = np.random.rand(info['nchan'], 1)
    beamformer_def.input_node = input_node
    return beamformer_def


def test_defaults(beamformer_def):
    assert(beamformer_def.mne_forward_model_file_path is None)
    assert(beamformer_def.mne_info is None)


def test_initialize(beamformer):
    beamformer.initialize()


def test_reset(beamformer):
    out_hist = beamformer._reset()
    # assert(self.beamformer._should_reinitialize == True)
    assert(out_hist == True)


def test_update(beamformer):
    beamformer._initialize()
    beamformer._update()


def test_check_value(beamformer):
    with pytest.raises(ValueError):
        beamformer.snr = -1

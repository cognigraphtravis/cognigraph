from nose.tools import assert_equals, raises
# from scripts.mce import MCE
from cognigraph.nodes.processors import MCE
from cognigraph.nodes.sources import FileSource
import os.path as op
import numpy as np
from mne.io import Raw

test_data_path = op.join(op.dirname(__file__), 'tests', 'data')


class TestMCE:
    def setup(self):
        self.snr = 1
        self.fwd_model_path = op.join(
                test_data_path, 'dmalt_custom_lr-fwd.fif')
        info_src_path = op.join(
                test_data_path, 'Koleno.fif')

        raw = Raw(info_src_path)
        raw.set_eeg_reference(ref_channels='average')
        self.info = raw.info

        # self.info = read_info(info_src_path)

        print(self.fwd_model_path)
        self.n_comp = 10
        self.mce = MCE(self.snr, self.fwd_model_path, self.n_comp)
        self.mce.mne_info = self.info
        N_SEN = len(self.info['ch_names'])
        self.mce.input = np.random.rand(N_SEN)

        assert_equals(self.mce.snr, self.snr)
        assert_equals(self.mce.mne_forward_model_file_path, self.fwd_model_path)
        assert_equals(self.mce.n_comp, self.n_comp)

        # assert defaults
        self.mce_def = MCE()

        assert_equals(self.mce_def.snr, 1.0)
        assert(self.mce_def.mne_forward_model_file_path is None)
        assert_equals(self.mce_def.n_comp, 40)
        assert(self.mce_def.info is None)

        input_node = FileSource()
        input_node.mne_info = self.info
        self.mce.input_node = input_node
        self.mce_def.input_node = input_node

    def test_initialize(self):
        self.mce.initialize()


    def test_reset(self):
        out_hist = self.mce._reset()
        # assert(self.mce._should_reinitialize == True)
        assert(out_hist == True)

    def test_update(self):
        self.mce._initialize()
        self.mce._update()

    @raises(ValueError)
    def test_check_value(self):
        self.mce.snr = -1
        self.mce._check_value()


    # def test_reset(self):
    #     self.mce._reset()


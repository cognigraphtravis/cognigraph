""" Compute coregistration and forward model for EDF data
AUTHOR: dmalt
DATE: 2018-05-15

"""

import numpy as np
from scipy.signal import hilbert, lfilter
import matplotlib.pyplot as plt
import os.path as op

import mne
from mne.minimum_norm import make_inverse_operator, apply_inverse_raw
from mne import make_forward_solution
from mne import pick_types
from mne.viz import plot_alignment
from mne.viz import plot_topomap
from mne.io.edf import read_raw_edf
from mne.channels import Montage
from os.path import splitext
from scipy.io import loadmat

data_path = '/home/dmalt/Code/python/cogni_submodules/tests/data'
subjects_dir = '/home/dmalt/mne_data/MNE-sample-data/subjects'
montage_path = '/home/dmalt/Data/cognigraph/data/nvx136_sens_locations.npy'
subject = 'sample'

fname_edf =  'DF_2018-03-02_11-34-38.edf'
edf_path = op.join(data_path, fname_edf)
basename_edf, ext = op.splitext(fname_edf)

# #  set montage {{{ # 
# raw = read_raw_edf(edf_path, preload=True, stim_channel=-1, misc=[128,129,130])
# del raw._cals  # fixes bug with pick_types

# ch_locs = np.load(montage_path)
# ch_locs[:, :2] = ch_locs[:, -2:-4:-1]
# ch_locs[:, 0] = -ch_locs[:, 0]
# aux_ch_locs = np.zeros([4, 3])
# ch_locs_all = np.concatenate([ch_locs, aux_ch_locs])
# ch_names = raw.ch_names
# kind = 'nvx_136'
# selection = np.arange(len(ch_locs_all))
# montage = Montage(ch_locs_all, ch_names, kind, selection)
# montage.nasion = np.array([0, 0.1, 0])
# montage.lpa = np.array([-0.075, 0, 0])
# montage.rpa = np.array([0.075, 0, 0])

# raw.set_montage(montage)
# raw.pick_types(meg=False, eeg=True)

# basename, ext =  op.splitext(edf_path)
# raw.save(basename + '.fif', overwrite=True)
# #  }}} set montage # 

#  compute forward {{{ # 
raw = mne.io.Raw(op.join(data_path, basename_edf + '.fif'),
                 preload=True)
trans_path = op.join(data_path, basename_edf + '-trans.fif')

src = mne.setup_source_space(subject, spacing='ico4',
                             subjects_dir=subjects_dir, add_dist=False)

conductivity = (0.3, 0.006, 0.3)  # for three layers
model = mne.make_bem_model(subject='sample', ico=4,
                           conductivity=conductivity,
                           subjects_dir=subjects_dir)

bem = mne.make_bem_solution(model)

raw.set_eeg_reference(ref_channels='average')
raw.apply_proj()
raw.pick_types(eeg=True, stim=False)
fwd = make_forward_solution(raw.info, trans=trans_path, src=src, bem=bem,
                            meg=False, eeg=True, mindist=7, n_jobs=2)

mne.write_forward_solution(basename_edf + '-fwd.fif', fwd, overwrite=True)
# fwd = mne.read_forward_solution('dmalt_custom_lr-fwd.fif')
fwd = mne.read_forward_solution(basename_edf + '-fwd.fif')

print(fwd)
leadfield = fwd['sol']['data']
print("Leadfield size : %d sensors x %d dipoles" % leadfield.shape)
n_sen = leadfield.shape[0]
#  }}} compute forward # 

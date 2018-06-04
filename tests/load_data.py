import numpy as np
from scipy.signal import hilbert, lfilter
import matplotlib.pyplot as plt
import os.path as op

import mne
from mne.minimum_norm import make_inverse_operator, apply_inverse_raw
from mne import make_forward_solution
from mne.beamformer import lcmv_raw
from mne.beamformer import  apply_lcmv_raw
# from make_lcmv_cython import make_lcmv
from mne import pick_types
from mne.viz import plot_alignment
from mne.viz import plot_topomap
from mne.io import read_raw_brainvision as Raw
from mne.channels import Montage
from os.path import splitext
from scipy.io import loadmat

import sys
sys.path.append('/home/dmalt/Code/python/cogni_submodules/')
from cognigraph.helpers.make_lcmv import make_lcmv

data_path = '/home/dmalt/Data/cognigraph/data'
subjects_dir = '/home/dmalt/mne_data/MNE-sample-data/subjects'
trans_file = '/home/dmalt/Data/cognigraph/data/sample_aligned-trans.fif'
subject = 'sample'
fname_eeg = op.join(data_path, 'Koleno.vhdr')

# #  {{{prepare channels and load data #
# ch_path  = '/home/dmalt/Data/cognigraph/channel_BrainProducts_ActiCap_128.mat'
# ch_struct = loadmat(ch_path)
# kind = ch_struct['Comment'][0]
# chans = ch_struct['Channel'][0]
# ch_locs = np.empty([len(chans), 3])
# ch_types = [None] * len(chans)
# ch_names = [None] * len(chans)
# selection = np.arange(len(chans))

# for i_ch, chan in enumerate(chans):
#     ch_names[i_ch] = chan[0][0]
#     ch_types[i_ch] = chan[2][0]
#     ch_locs[i_ch] = chan[4][:, 0]


# ch_locs[:,0:2] = ch_locs[:,-2:-4:-1]
# ch_locs[:,0] = -ch_locs[:,0]
# ch_names[ch_names.index('GND')] = 'AFz'
# ch_names[ch_names.index('REF')] = 'FCz'

# montage = Montage(ch_locs, ch_names, kind, selection)
# # montage.plot()

# raw = Raw(fname_eeg, preload=True)
# raw.set_montage(montage)
# raw.info['bads'] = ['F5', 'PPO10h', 'C5', 'FCC2h', 'F2', 'VEOG']
# #  prepare channels and load data}}} #
raw = mne.io.Raw('/home/dmalt/Data/cognigraph/data/raw_sim.fif', preload=True)

# #  plot timestamp {{{topo #
# raw_c = raw.copy()

# raw_c.pick_types(eeg=True, eog=False, stim=False, exclude='bads')
# raw_c.filter(l_freq = 8, h_freq=12)
# picks = pick_types(raw_c.info,  eeg=True, eog=False,
#                            stim=False, exclude='bads')
# raw_c.apply_hilbert(picks=picks, envelope=True)

# start, stop = raw_c.time_as_index(81)

# topo_data = raw_c[:,start:end][0]

# plot_topomap(topo_data.mean(axis=1), raw_c.info)
# plot_topomap(topo_data[:,360], raw_c.info, cmap='viridis')
# #  topo}}} #

# #  setup {{{sources #

# plot_alignment(raw.info, trans=trans_file, dig=True, subject=subject,
#                subjects_dir=subjects_dir, surfaces=['brain', 'head'])

# src = mne.setup_source_space(subject, spacing='oct6',
#                              subjects_dir=subjects_dir, add_dist=False)

# print(src)
# # ------------- plot source space points ------------- #

# from mayavi import mlab  # noqa
# from surfer import Brain  # noqa

# brain = Brain('sample', 'lh', 'inflated', subjects_dir=subjects_dir)
# surf = brain.geo['lh']

# vertidx = np.where(src[0]['inuse'])[0]

# mlab.points3d(surf.x[vertidx], surf.y[vertidx],
#               surf.z[vertidx], color=(1, 1, 0), scale_factor=1.5)

# # ---------------------------------------------------- #
# #  sources}}} #

#  compute {{{forward #
# conductivity = (0.3,)  # for single layer
# conductivity = (0.3, 0.006, 0.3)  # for three layers
# model = mne.make_bem_model(subject='sample', ico=4,
#                            conductivity=conductivity,
#                            subjects_dir=subjects_dir)

# bem = mne.make_bem_solution(model)


raw.set_eeg_reference(ref_channels='average')
raw.apply_proj()
raw.pick_types(eeg=True, stim=False)
# fwd = make_forward_solution(raw.info, trans=trans_file, src=src, bem=bem,
#                             meg=False, eeg=True, mindist=5.0, n_jobs=2)

# mne.write_forward_solution('dmalt_custom_lr-fwd.fif', fwd)
fwd = mne.read_forward_solution('./tests/data/dmalt_custom_mr-fwd.fif')

print(fwd)
leadfield = fwd['sol']['data']
print("Leadfield size : %d sensors x %d dipoles" % leadfield.shape)
n_sen = leadfield.shape[0]
#  forward}}} #

#  compute {{{inverse #
snr = 1           # use smaller SNR for raw data
inv_method = 'MNE'  # sLORETA, MNE, dSPM
# inv_method = 'dSPM'  # sLORETA, MNE, dSPM
parc = 'aparc'      # the parcellation to use, e.g., 'aparc' 'aparc.a2009s'

lambda2 = 1.0 / snr ** 2
# lambda2 = 0.1

# cov_data = np.eye(n_sen) * 1e-12
cov_data = np.identity(n_sen)
noise_cov = mne.Covariance(cov_data, raw.info['ch_names'], raw.info['bads'], raw.info['projs'], nfree=1)
# noise_cov =  mne.compute_raw_covariance(raw, tmin=0, tmax=29, method='shrunk')
inverse_operator = make_inverse_operator(raw.info, fwd, noise_cov, depth=0.8, loose=1, fixed=False)
# inverse_operator = make_inverse_operator(raw.info, fwd, noise_cov, depth=None, loose=1, fixed=False)

#  inverse {{{cogni #
# inv_cogni_path = op.join(data_path, 'inv_mat.npy')
# inv_cogni = np.load(file=inv_cogni_path)
#  cogni}}} #
#  inverse}}} #

# start, stop = raw.time_as_index([75,85])
start, stop = raw.time_as_index([0,29.9])

#  {{{ apply inverse #
raw_c = raw.copy()
raw_c.filter(l_freq = 8, h_freq=12)
raw_c.set_eeg_reference(ref_channels='average')
raw_c.apply_proj()
# stc = apply_inverse_raw(raw_c, inverse_operator, pick_ori='vector', method=inv_method, lambda2=lambda2, start=start, stop=stop)
# stc = apply_inverse_raw(raw_c, inverse_operator,  method=inv_method, lambda2=lambda2, start=start, stop=stop)
# raw.set_reference_channels
# stc
#  {{{beamformer #
fwd_fix = mne.convert_forward_solution(fwd, surf_ori=True,  force_fixed=False)
data_cov = mne.compute_raw_covariance(raw_c, tmin=0, tmax=5, method='shrunk')
import time
t1 = time.time()
# stc = lcmv_raw(raw_c, fwd_fix, None, data_cov, reg=0.05, start=start, stop=stop,
#                pick_ori='max-power', weight_norm='unit_noise_gain', max_ori_out='signed')
filters = make_lcmv(info=raw_c.info, forward=fwd_fix, data_cov=data_cov,
                    reg=0.5, pick_ori='max-power',
                    weight_norm='unit-noise-gain', reduce_rank=False)
t2 = time.time()
print('{:.2f}'.format((t2 - t1) * 1000))
# stc = apply_lcmv_raw(raw=raw_c, filters=filters, max_ori_out='abs')
t3 = time.time()
print('{:.2f}'.format((t3 - t2) * 1000))

#  beamformer}}} #

# raw_cc = raw_c.copy()
# raw_cc[:,:] = np.random.rand(raw_cc[:,:][0].shape[0], raw_cc[:,:][0].shape[1]) - 1

# data_cogni, times = raw_c[:,start:stop]
# src_cogni = np.dot(inv_cogni, data_cogni)

# data_me = raw_c.get_data()[:,start:stop]
# id_mat = np.identity(data_me.shape[0])
# dummy_raw = mne.io.RawArray(data=id_mat, info=raw_c.info, verbose='ERROR')
# stc_me = mne.minimum_norm.apply_inverse_raw(dummy_raw, inverse_operator, lambda2, method=inv_method, verbose='ERROR')
# src_me = np.dot(stc_me.data, data_me)

#  apply inverse}}} #

# {{{hilbert-transform sources #
# stc_hilb = stc.copy()
# hilb = hilbert(stc.data)
# hilb = hilbert(src_me)
# hilb = np.abs(hilb)
# stc_hilb.data = hilb

# stc_smooth = stc.copy()
# factor = 0.99
# a = [1, -factor]
# b = [1 - factor]

# smooth = lfilter(b, a, np.abs(stc.data), axis=1)
# stc_smooth.data =  smooth

# stc_hilb_cogni = stc.copy()
# hilb_cogni = hilbert(src_cogni)
# hilb_cogni = np.abs(hilb_cogni)
# stc_hilb_cogni.data = hilb_cogni

#  hilbert-transform sources}}} #

#  {{{plot src data #
# stc_hilb.plot(hemi='both', initial_time=82, time_viewer=True,
#               subjects_dir=subjects_dir, transparent=True,  colormap='bwr')

# stc.plot(hemi='rh', initial_time=0, time_viewer=True,
#          subjects_dir=subjects_dir, transparent=True)

# stc.plot(hemi='rh', subjects_dir=subjects_dir,
#              initial_time=0.21, time_unit='s', time_viewer=True)

# stc_smooth.plot(hemi='split', initial_time=0, time_viewer=True,
#         # clim=dict(kind='value', lims =[1.e-8, 1.5e-8, 2e-8]),
#                 subjects_dir=subjects_dir, transparent=True,  colormap='bwr')

# stc_hilb_cogni.plot(hemi='both', initial_time=82, subjects_dir=subjects_dir, time_viewer=True, transparent=True, colormap='bwr')
#  plot src data}}} #

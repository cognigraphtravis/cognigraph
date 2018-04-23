"""Minimum current estimate"""
from load_data import *
# def mce(fwd, n_svd):
from numpy.linalg import svd
from scipy.linalg import block_diag
from scipy.optimize import linprog
from sklearn.preprocessing import normalize

fwd_fix = mne.convert_forward_solution(fwd, surf_ori=True,  force_fixed=False)
leadfield = fwd_fix['sol']['data']
print("Leadfield size : %d sensors x %d dipoles" % leadfield.shape)

n_comp = 40

n_dipoles = leadfield.shape[1]
n_src = fwd['nsource']

U, S, V = svd(leadfield)

Sn = np.zeros([n_comp, V.shape[0]])
Sn[:n_comp,:n_comp] = np.diag(S[:n_comp])

Un = U[:,:n_comp]
A_temp = Sn @ V
# Sn = S[:n_comp]

# data = raw_c.get_data(start=start, stop=stop)
# t_start, t_stop= 80, 83
t_start, t_stop= 0, 2
t_step = 1 / raw_c.info['sfreq']
start, stop = raw.time_as_index([t_start, t_stop])
times = np.arange(t_start, t_stop, t_step)
times = times[:13]
data_mce = np.empty([n_src, len(times)])

inv_method='MNE'
for i, time in enumerate(times):
    slice_ind = raw_c.time_as_index(time)[0]
    stc_slice = apply_inverse_raw(raw_c, inverse_operator, pick_ori='vector',
                                  method=inv_method, lambda2=lambda2,
                                  start=slice_ind, stop=slice_ind + 1)

    dd = stc_slice.data[:,:,0]

    Q = normalize(dd)
    QQ = block_diag(*Q).T

    A_eq = A_temp @ QQ
    data_slice = raw_c.get_data()[:,slice_ind]
    b_eq = Un.T @ data_slice
    c = np.ones(A_eq.shape[1])

    sol = linprog(c, A_eq=A_eq, b_eq=b_eq, method='interior-point', bounds=(0,None), options={'disp':True})
    # sol = linprog(c, A_eq=A_eq, b_eq=b_eq, method='simplex', bounds=(0,None), options={'disp':True})
    data_mce[:,i] = sol.x
data_mce[:,0] = sol.x


stc_mce_scal = apply_inverse_raw(raw_c, inverse_operator,  method=inv_method,
                                 lambda2=lambda2, start=start, stop=stop)
stc_mce = stc_mce_scal.copy()

factor = 0.999
a = [1, -factor]
b = [1 - factor]

smooth = lfilter(b, a, np.abs(data_mce), axis=1)
stc_mce.data =  smooth
# stc_mce.data = data_mce

stc_mce.plot(hemi='split', initial_time=0, time_viewer=True,
        clim=dict(kind='value', lims =[1.e-8, 1.5e-8, 2e-8]),
         subjects_dir=subjects_dir, transparent=True,  colormap='bwr')

# sol = linprog(c,  options={'disp':True})

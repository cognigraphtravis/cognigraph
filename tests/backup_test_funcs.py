
def make_inverse_operator(forward_model_file_path, mne_info, sigma2=1):
    import mne
    import numpy as np
    # sigma2 is what will be used to scale the identity covariance matrix.
    # This will not affect MNE solution though.
    # The inverse operator will use channels common to forward_model_file_path and mne_info.
    forward = mne.read_forward_solution(forward_model_file_path, verbose='ERROR')
    cov = mne.Covariance(data=sigma2 * np.identity(mne_info['nchan']),
                         names=mne_info['ch_names'], bads=mne_info['bads'],
                         projs=mne_info['projs'], nfree=1)

    return mne.minimum_norm.make_inverse_operator(mne_info, forward,
                                                  cov, depth=None, loose=0,
                                                  fixed=True, verbose='ERROR')

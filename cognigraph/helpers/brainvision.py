"""Functions to read input from file"""
import os
import mne
import mne.io.brainvision.brainvision as brainvision


BRAINVISION_TIME_AXIS = 1


# Monkey-patch to accommodate for spelling mistakes in the header file
# if not hasattr(brainvision, '_check_version_monkey_patched'):  # Do not repeat upon reimporting
#     def _check_mrk_version(header):
#         bobe_mrk_header = 'BrainVision Data Exchange Marker File, Version 1.0'
#         if header == bobe_mrk_header:
#             return True
#         else:
#             return brainvision._check_mrk_version_original(header)


#     def _check_hdr_version(header):
#         bobe_hdr_header = 'BrainVision Data Exchange Header File Version 1.0'
#         if header == bobe_hdr_header:
#             return True
#         else:
#             return brainvision._check_hdr_version_original(header)


#     brainvision._check_hdr_version_original = brainvision._check_hdr_version
#     brainvision._check_hdr_version = _check_hdr_version

#     brainvision._check_mrk_version_original = brainvision._check_mrk_version
#     brainvision._check_mrk_version = _check_mrk_version

#     brainvision._check_version_monkey_patched = True


def read_brain_vision_data(file_path, time_axis, start_s: int=0, stop_s: int=None):
    vhdr_file_path = os.path.splitext(file_path)[0] + '.vhdr'
    raw = mne.io.read_raw_brainvision(vhdr_fname=vhdr_file_path,
                                      verbose='ERROR')  # type: mne.io.Raw

    # Get the required time slice.
    # mne.io.Raw.get_data takes array indices, not time
    start = 0 if start_s is None else raw.time_as_index(start_s)[0]
    stop = min(raw.n_times if stop_s is None else raw.time_as_index(stop_s)[0],
               raw.n_times)
    data = raw.get_data(start=start, stop=stop)

    mne_info = raw.info.copy()

    if time_axis != BRAINVISION_TIME_AXIS:
        data = data.T

    return data, mne_info


def read_fif_data(file_path, time_axis, start_s: int=0, stop_s: int=None):
    raw = mne.io.Raw(fname=file_path, verbose='ERROR')  # type: mne.io.Raw

    # Get the required time slice.
    # mne.io.Raw.get_data takes array indices, not time
    start = 0 if start_s is None else raw.time_as_index(start_s)[0]
    stop = min(raw.n_times if stop_s is None else raw.time_as_index(stop_s)[0],
               raw.n_times)
    data = raw.get_data(start=start, stop=stop)

    mne_info = raw.info.copy()

    if time_axis != BRAINVISION_TIME_AXIS:
        data = data.T

    return data, mne_info


def read_edf_data(file_path, time_axis, start_s: int=0, stop_s: int=None):
    raw = mne.io.edf.read_raw_edf(input_fname=file_path, preload=True,
                                  verbose='ERROR', stim_channel=-1,
                                  misc=[129,130,131])  # type: mne.io.Raw
    try:
        del raw._cals  # fixes bug with pick_types in edf data
    except:
        pass
    raw.pick_types(meg=False, eeg=True)
    # Get the required time slice.
    # mne.io.Raw.get_data takes array indices, not time
    start = 0 if start_s is None else raw.time_as_index(start_s)[0]
    stop = min(raw.n_times if stop_s is None else raw.time_as_index(stop_s)[0], raw.n_times)
    data = raw.get_data(start=start, stop=stop)

    mne_info = raw.info.copy()

    if time_axis != BRAINVISION_TIME_AXIS:
        data = data.T

    return data, mne_info

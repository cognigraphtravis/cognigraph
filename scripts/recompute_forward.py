import argparse
import os
import sys
import time
import csv
import numpy as np
import mne
import logging
from builtins import input
logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s: %(message)s')
# ------------------------ define argparse arguments ------------------------ #
parser = argparse.ArgumentParser(
    description= 'Recompute forward model for new channel locations ' +
    'based on coregistration ')

parser.add_argument('channels', type=argparse.FileType('r'),
                    help='File with tab-separated channel locations')
parser.add_argument('forward', type=argparse.FileType('r'),
                    help='Path to forward solution')
parser.add_argument('trans', type=argparse.FileType('r'),
                    help='Path to trans file')
parser.add_argument('subject_name',# required=True,
                    help='Subject name in FreeSurfer folder')
parser.add_argument('-s', '--subjects-dir', default=os.getenv('SUBJECTS_DIR'),
                    help='Path to Freesurfer`s SUBJECTS_DIR.' +
                    ' Defaults to env variable with the same name')
parser.add_argument('-d', '--dest',
                    default='{}-fwd.fif'.format(time.strftime("%d_%m_%y")),
                    help='Destination file')
parser.add_argument('-p', '--padding', type=int, default=0,
                    help='Skip p columnts from the left in channels file')
parser.add_argument('-j', '--n_jobs', type=int, default=1,
                    help='Number of jobs for forward solution computation')
parser.add_argument('-f', '--force', action='store_true',
                    help='If specified, overwrite destination file' +
                    ' without prompting')
# --------------------------------------------------------------------------- #


def parse_channels_file(channels_file, padding):
    p = padding
    ch_names = []
    ch_locs = np.empty([0, 3])
    n_good_reads = 0
    with open(channels_file, 'r') as f:
        tsvin = csv.reader(f, delimiter='\t')
        for i, line in enumerate(tsvin):
            try:
                ch_names.append(line[p])
                ch_loc = np.array([float(loc) for loc in line[p + 1: p + 4]])
                assert np.any(ch_loc)  # don't include chans with zero coords
                ch_locs = np.vstack((ch_locs, ch_loc))
                n_good_reads += 1
            except (IndexError, ValueError):
                logging.warning(
                    'parse_channels_file: '
                    'Bad format for line number {} ({}).'.format(i + 1, line) +
                    ' Skipping.')
            except AssertionError:
                logging.warning(
                    'parse_channels_file: '
                    'All coordinates are zero for line number {}.'.format(i) +
                    ' Skipping')
    if n_good_reads:
        logging.info('Read {} channels from {}'.format(n_good_reads,
                                                       channels_file))
    else:
        logging.error('parse_channels_file:' +
                      'no channels were found in {}\n'.format(channels_file) +
                      'Aborting.')
        sys.exit()
    return ch_names, ch_locs


def write_forward_prompted(fwd_savename, fwd, overwrite):
    try:
        mne.write_forward_solution(fwd_savename, fwd, overwrite=overwrite)
    except OSError:
        while True:
            answer = input('Destination file exists. Rewrite? [y/n] ')
            answer = answer.upper()
            if answer == 'Y' or answer == 'YES':
                mne.write_forward_solution(
                    fwd_savename, fwd, overwrite=True)
                break
            elif answer == 'N' or answer == 'NO':
                new_dest_answer = input('Specify another destination? [y/n] ')
                new_dest_answer = new_dest_answer.upper()
                if new_dest_answer == 'Y' or new_dest_answer == 'YES':
                    new_dest = input('Enter new destination path: ')
                    write_forward_prompted(new_dest, fwd, overwrite=False)
                    break
                elif new_dest_answer == 'N' or new_dest_answer == 'NO':
                    break
            else:
                continue


if __name__ == '__main__':
    args = parser.parse_args()

    # ---------------------- read channels locations ---------------------- #
    logging.info('Reading channels locations ...')
    ch_names, ch_locs = parse_channels_file(args.channels.name, args.padding)
    n_chans = len(ch_names)
    kind = 'custom EEG {}'.format(n_chans)
    selection = np.arange(n_chans)
    montage = mne.channels.Montage(ch_locs, ch_names, kind, selection)
    # --------------------------------------------------------------------- #

    fwd = mne.read_forward_solution(args.forward.name, verbose='WARNING')
    src = fwd['src']
    info = fwd['info']
    info['comps'] = []  # otherwise make_forward_solution complains
    logging.info('Setting montage ...')
    mne.channels.montage._set_montage(info, montage, update_ch_names=True)

    # conductivity = (0.3, 0.006, 0.3)  # for three layers
    subjects_dir = args.subjects_dir
    logging.info('SUBJECTS_DIR is set to {}'.format(subjects_dir))
    logging.info('Creating bem model (be patient) ...')
    model = mne.make_bem_model(subject=args.subject_name, ico=4,
                               # conductivity=conductivity,  # use default
                               subjects_dir=subjects_dir,
                               verbose='WARNING')
    bem = mne.make_bem_solution(model, verbose='WARNING')
    trans_file = args.trans.name
    n_jobs = args.jobs
    logging.info('Computing forward solution (be patient) ...')
    fwd = mne.make_forward_solution(info, trans=trans_file, src=src,
                                    bem=bem, meg=False, eeg=True,
                                    mindist=5.0, n_jobs=n_jobs,
                                    verbose='WARNING')

    fwd_savename = args.dest
    write_forward_prompted(fwd_savename, fwd, args.force)

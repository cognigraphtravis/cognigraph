import argparse
import sys
import time
from PyQt5 import QtCore, QtGui
import mne
import os.path as op

from cognigraph.pipeline import Pipeline
from cognigraph.nodes import sources, processors, outputs
from cognigraph.gui.window import GUIWindow

# Убираем предупреждения numpy, иначе в iPython некрасиво как-то Ж)
import numpy as np
np.warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('-d','--data', type=argparse.FileType('r'),
                    help='data path')
parser.add_argument('-f', '--forward', type=argparse.FileType('r'),
                    help='forward model path')
args = parser.parse_args()

sys.path.append('../vendor/nfb')  # For nfb submodule

app = QtGui.QApplication(sys.argv)

SURF_DIR = op.join(mne.datasets.sample.data_path(), 'subjects/sample/surf')
DATA_DIR = '/home/dmalt/Code/python/cogni_submodules/tests/data'
FWD_MODEL_NAME = 'dmalt_custom_mr-fwd.fif'
# Собираем узлы в пайплайн

pipeline = Pipeline()

if not args.data:
        file_tuple = QtGui.QFileDialog.getOpenFileName(
                caption="Select Data",
                filter="Brainvision (*.eeg *.vhdr *.vmrk);;" +
                       "MNE-python (*.fif);;" +
                       "European Data Format (*.edf)")
        print(file_tuple)
        file_path = file_tuple[0]
else:
    file_path = args.data.name

if not file_path:
    raise Exception("DATA PATH IS MANDATORY!")

if not args.forward:
    try:
        fwd_tuple = QtGui.QFileDialog.getOpenFileName(
                caption="Select forward model",
                filter= "MNE-python forward (*-fwd.fif)")
        fwd_path = fwd_tuple[0]
    except:
        print("DATA FILE IS MANDATORY!")
else:
    fwd_path = args.forward.name

if not fwd_path:
    raise Exception("FORWARD SOLUTION IS MANDATORY!")

source = sources.FileSource(file_path=file_path)
# source = sources.FileSource()
source.loop_the_file = True
source.MAX_SAMPLES_IN_CHUNK = 10000
pipeline.source = source


# Processors
preprocessing = processors.Preprocessing(collect_for_x_seconds=120)
pipeline.add_processor(preprocessing)

linear_filter = processors.LinearFilter(lower_cutoff=8.0, upper_cutoff=12.0)
pipeline.add_processor(linear_filter)

inverse_model = processors.InverseModel(
        method='MNE', snr=1.0,
        forward_model_path=fwd_path)
pipeline.add_processor(inverse_model)

envelope_extractor = processors.EnvelopeExtractor(0.99)
pipeline.add_processor(envelope_extractor)

# Outputs
global_mode = outputs.ThreeDeeBrain.LIMITS_MODES.GLOBAL
three_dee_brain = outputs.ThreeDeeBrain(
        limits_mode=global_mode, buffer_length=6, surfaces_dir=SURF_DIR)
pipeline.add_output(three_dee_brain)
# pipeline.add_output(outputs.LSLStreamOutput())

signal_viewer = outputs.SignalViewer()
pipeline.add_output(signal_viewer, input_node=linear_filter)


# Создаем окно

window = GUIWindow(pipeline=pipeline)
window.init_ui()
window.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
# window.show() # Will show after init


# Инициализируем все узлы
window.initialize()


# Симулируем работу препроцессинга по отлову шумных каналов

# Set bad channels manually
# bad_channel_labels = ['Fp2', 'F5', 'C5', 'F2', 'PPO10h', 'POO1', 'FCC2h', 'VEOG']
# preprocessing._bad_channel_indices = mne.pick_channels(
#     source.mne_info['ch_names'], include=bad_channel_labels)
# preprocessing.mne_info['bads'] = bad_channel_labels
# # preprocessing._samples_to_be_collected = 0
# preprocessing._enough_collected = True

# message = node.Message(there_has_been_a_change=True,
#                        output_history_is_no_longer_valid=True)
# preprocessing._deliver_a_message_to_receivers(message)

# fif_file_path = source.file_path
# start_s, stop_s = 80, 100
# with source.not_triggering_reset():
#     source.data, _ = read_fif_data(
#         fif_file_path, time_axis=TIME_AXIS, start_s=start_s, stop_s=stop_s)
# Подключаем таймер окна к обновлению пайплайна
class AsyncUpdater(QtCore.QRunnable):
    _stop_flag = False

    def __init__(self):
        super(AsyncUpdater, self).__init__()
        self.setAutoDelete(False)

    def run(self):
        self._stop_flag = False

        while self._stop_flag is False:
            start = time.time()
            pipeline.update_all_nodes()
            end = time.time()

            # Force sleep to update at 10Hz
            if end - start < 0.1:
                time.sleep(0.1 - (end - start))

    def stop(self):
        self._stop_flag = True


pool = QtCore.QThreadPool.globalInstance()
updater = AsyncUpdater()
is_paused = True


def toggle_updater():
    global pool
    global updater
    global is_paused

    if is_paused:
        is_paused = False
        pool.start(updater)
    else:
        is_paused = True
        updater.stop()
        pool.waitForDone()


window.run_button.clicked.connect(toggle_updater)


# Show window and exit on close
window.show()
updater.stop()
pool.waitForDone()
sys.exit(app.exec_())

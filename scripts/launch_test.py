import sys
import time


from PyQt5 import QtCore, QtGui
import mne
import os

from cognigraph.helpers.brainvision import read_brain_vision_data, read_fif_data
from cognigraph.pipeline import Pipeline
from cognigraph.nodes import sources, processors, outputs, node
from cognigraph import TIME_AXIS
from cognigraph.gui.window import GUIWindow

# Убираем предупреждения numpy, иначе в iPython некрасиво как-то Ж)
import numpy as np
np.warnings.filterwarnings('ignore')

sys.path.append('../vendor/nfb')  # For nfb submodule

app = QtGui.QApplication(sys.argv)

import os.path as op
cur_dir =  '/home/dmalt/Code/python/cogni_submodules'
test_data_path = cur_dir + '/tests/data/'
print(test_data_path)
# sim_data_fname = 'raw_sim.fif'
sim_data_fname = 'Koleno.fif'
# fwd_fname = 'dmalt_custom_lr-fwd.fif'
fwd_fname = 'dmalt_custom_mr-fwd.fif'
fwd_path = op.join(test_data_path, fwd_fname)

surf_dir = '/home/dmalt/mne_data/MNE-sample-data/subjects/sample/surf'
# Собираем узлы в пайплайн

pipeline = Pipeline()

# launch_test_filepath = QtGui.QFileDialog.getOpenFileName(
#     caption="Select Data", filter="Brainvision (*.eeg *.vhdr *.vmrk)")[0]

launch_test_filepath = QtGui.QFileDialog.getOpenFileName(
    caption="Select Data", filter="Fif (*.fif)")[0]

# source = sources.BrainvisionSource(file_path=launch_test_filepath)
source = sources.FifSource(file_path=launch_test_filepath)

source.loop_the_file = True
source.MAX_SAMPLES_IN_CHUNK = 10000
pipeline.source = source


# Processors
preprocessing = processors.Preprocessing(collect_for_x_seconds=120)
pipeline.add_processor(preprocessing)

linear_filter = processors.LinearFilter(lower_cutoff=8.0, upper_cutoff=12.0)
pipeline.add_processor(linear_filter)

inverse_model = processors.InverseModel(method='MNE', snr=1.0,
                                        forward_model_path=fwd_path)
pipeline.add_processor(inverse_model)

envelope_extractor = processors.EnvelopeExtractor(0.99)
pipeline.add_processor(envelope_extractor)

# Outputs
global_mode = outputs.ThreeDeeBrain.LIMITS_MODES.GLOBAL
three_dee_brain = outputs.ThreeDeeBrain(limits_mode=global_mode,
                                        buffer_length=6, surfaces_dir=surf_dir)
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

# Обрезаем данные в диапазоне с приличной записью
# vhdr_file_path = os.path.splitext(source.file_path)[0] + '.vhdr'
# start_s, stop_s = 80, 100
# with source.not_triggering_reset():
#     source.data, _ = read_brain_vision_data(
#         vhdr_file_path, time_axis=TIME_AXIS, start_s=start_s, stop_s=stop_s)

fif_file_path = source.file_path
start_s, stop_s = 80, 100
with source.not_triggering_reset():
    source.data, _ = read_fif_data(
        fif_file_path, time_axis=TIME_AXIS, start_s=start_s, stop_s=stop_s)
# Подключаем таймер окна к обновлению пайплайна
class AsyncUpdater(QtCore.QRunnable):
    _stop_flag = False

    def __init__(self):
        super(AsyncUpdater, self).__init__()
        self.setAutoDelete(False)

    def run(self):
        self._stop_flag = False

        while self._stop_flag == False:
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

    if is_paused == True:
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

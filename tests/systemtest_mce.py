"""Integration test to check mce performance"""
import numpy as np
np.warnings.filterwarnings('ignore')  # noqa
import os
os.environ['PYQTGRAPH_QT_LIB'] = 'PyQt5' # noqa

import time
import sys
import os.path as op

from PyQt5 import QtCore, QtGui

from cognigraph.helpers.brainvision import read_fif_data
from cognigraph.pipeline import Pipeline
from cognigraph.nodes import sources, processors, outputs
from cognigraph import TIME_AXIS
from cognigraph.gui.window import GUIWindow


app = QtGui.QApplication(sys.argv)

pipeline = Pipeline()

cur_dir =  '/home/dmalt/Code/python/cogni_submodules'
test_data_path = cur_dir + '/tests/data/'
print(test_data_path)
sim_data_fname = 'raw_sim.fif'
sim_data_fname = 'Koleno.fif'
# fwd_fname = 'dmalt_custom_lr-fwd.fif'
fwd_fname = 'dmalt_custom_mr-fwd.fif'

surf_dir = '/home/dmalt/mne_data/MNE-sample-data/subjects/sample/surf'

fwd_path = op.join(test_data_path, fwd_fname)
sim_data_path = op.join(test_data_path, sim_data_fname)

source = sources.FileSource(file_path=sim_data_path)
source.loop_the_file = True
source.MAX_SAMPLES_IN_CHUNK = 10000
pipeline.source = source

# Processors
preprocessing = processors.Preprocessing(collect_for_x_seconds=30)
pipeline.add_processor(preprocessing)

linear_filter = processors.LinearFilter(lower_cutoff=8.0, upper_cutoff=12.0)
pipeline.add_processor(linear_filter)

inverse_model = processors.MCE(forward_model_path=fwd_path, snr=1.0)
# inverse_model = processors.InverseModel(method='MNE', forward_model_path=fwd_path, snr=1.0)
pipeline.add_processor(inverse_model)


envelope_extractor = processors.EnvelopeExtractor()
# pipeline.add_processor(envelope_extractor)

# Outputs
global_mode = outputs.ThreeDeeBrain.LIMITS_MODES.GLOBAL
three_dee_brain = outputs.ThreeDeeBrain(
        limits_mode=global_mode, buffer_length=10, surfaces_dir=surf_dir)
pipeline.add_output(three_dee_brain)
# pipeline.add_output(outputs.LSLStreamOutput())
# pipeline.initialize_all_nodes()

signal_viewer = outputs.SignalViewer()
pipeline.add_output(signal_viewer, input_node=linear_filter)

window = GUIWindow(pipeline=pipeline)
window.init_ui()
window.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
window.show()


base_controls = window._controls._base_controls
source_controls = base_controls.source_controls
processors_controls = base_controls.processors_controls
outputs_controls = base_controls.outputs_controls

source_controls.source_type_combo.setValue(source_controls.SOURCE_TYPE_PLACEHOLDER)


linear_filter_controls = processors_controls.children()[0]

envelope_controls = processors_controls.children()[2]
# envelope_controls.disabled.setValue(True)


three_dee_brain_controls = outputs_controls.children()[0]
three_dee_brain_controls.threshold_slider.setValue(50)
# three_dee_brain_controls.limits_mode_combo.setValue('Local')

window.initialize()

start_s, stop_s = 80, 100
with source.not_triggering_reset():
    source.data, _ = read_fif_data(sim_data_path, time_axis=TIME_AXIS, start_s=start_s, stop_s=stop_s)

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
window.show()
updater.stop()
pool.waitForDone()
sys.exit(app.exec_())
# def run():
#     pipeline.update_all_nodes()
#     # print(pipeline.source._samples_already_read / 500)


# timer = QtCore.QTimer()
# timer.timeout.connect(run)
# frequency = pipeline.frequency
# output_frequency = 10
# # timer.setInterval(1000. / frequency * 500)
# timer.setInterval(1000. / output_frequency)

# source.loop_the_file = False
# source.MAX_SAMPLES_IN_CHUNK = 10000
# # envelope.disabled = True


# if __name__ == '__main__':
#     import sys

#     timer.start()
    # timer.stop()

    # TODO: this runs when in iPython. It should not.
    # Start Qt event loop unless running in interactive mode or using pyside.
    # if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
    #     sys.exit(QtGui.QApplication.instance().exec_())

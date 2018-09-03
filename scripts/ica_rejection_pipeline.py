"""Integration test to check mce performance"""

import sys
import os.path as op
import mne

from pyqtgraph import QtCore, QtGui
import pyqtgraph

pyqtgraph.setConfigOption('useOpenGL',True)

from cognigraph.pipeline import Pipeline
from cognigraph.nodes import sources, processors, outputs
from cognigraph import TIME_AXIS
from cognigraph.gui.window import GUIWindow
import logging

app = QtGui.QApplication(sys.argv)

pipeline = Pipeline()

cur_dir =  '/home/dmalt/Code/python/cogni_submodules'
test_data_path = cur_dir + '/tests/data/'
print(test_data_path)
# sim_data_fname = 'raw_sim.fif'
data_fname = 'Koleno.fif'

data_path = op.join(test_data_path, data_fname)


source = sources.FifSource(file_path=data_path)
pipeline.source = source


# Processors
preprocessing = processors.Preprocessing(collect_for_x_seconds=0)
pipeline.add_processor(preprocessing)

ica_rejection = processors.ICARejection(collect_for_x_seconds=10)
pipeline.add_processor(ica_rejection)

linear_filter = processors.LinearFilter(lower_cutoff=8, upper_cutoff=12)
pipeline.add_processor(linear_filter)

inverse_model = processors.InverseModel(method='dSPM', snr=1.0)
pipeline.add_processor(inverse_model)

envelope_extractor = processors.EnvelopeExtractor(0.99)
pipeline.add_processor(envelope_extractor)


# Outputs
signal_viewer = outputs.SignalViewer()
pipeline.add_output(signal_viewer, input_node=linear_filter)

global_mode = outputs.ThreeDeeBrain.LIMITS_MODES.GLOBAL
three_dee_brain = outputs.ThreeDeeBrain(limits_mode=global_mode,
                                        buffer_length=6)
pipeline.add_output(three_dee_brain)

window = GUIWindow(pipeline=pipeline)
window.init_ui()
window.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
window.show()


# base_controls = window._controls._base_controls
# processors_controls = base_controls.processors_controls

window.initialize()

bad_channel_labels = ['Fp2', 'F5', 'C5', 'F2', 'PPO10h', 'POO1', 'FCC2h']
preprocessing.mne_info['bads'] = bad_channel_labels
preprocessing._samples_to_be_collected = 0
preprocessing._enough_collected = True


def run():
    logging.debug('Start iteration')
    pipeline.update_all_nodes()
    logging.debug('End iteration')


timer = QtCore.QTimer()
timer.timeout.connect(run)
frequency = pipeline.frequency
output_frequency = 1000


timer.setInterval(0)

source.loop_the_file = True
# source.MAX_SAMPLES_IN_CHUNK = int(frequency / output_frequency)

source.MAX_SAMPLES_IN_CHUNK = 10000

import numpy as np
np.warnings.filterwarnings('ignore')
# source.MAX_SAMPLES_IN_CHUNK = 5
# envelope.disabled = True


if __name__ == '__main__':
    import sys

    timer.start()

    # while True:
    #     pipeline.update_all_nodes()
    # timer.start()
    # timer.stop()

    # TODO: this runs when in iPython. It should not.
    # Start Qt event loop unless running in interactive mode or using pyside.
    # if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
    #     sys.exit(QtGui.QApplication.instance().exec_())

"""Integration test to check mce performance"""

import sys
import os.path as op

from pyqtgraph import QtCore, QtGui

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
# sim_data_fname = 'Koleno.fif'
fwd_fname = 'dmalt_custom_lr-fwd.fif'

surf_dir = '/home/dmalt/mne_data/MNE-sample-data/subjects/sample/surf'

fwd_path = op.join(test_data_path, fwd_fname)
sim_data_path = op.join(test_data_path, sim_data_fname)

source = sources.FifSource(file_path=sim_data_path)
pipeline.source = source

# Processors
preprocessing = processors.Preprocessing(collect_for_x_seconds=12)
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
        limits_mode=global_mode, buffer_length=6, surfaces_dir=surf_dir)
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
three_dee_brain_controls.limits_mode_combo.setValue('Global')
three_dee_brain_controls.limits_mode_combo.setValue('Local')

window.initialize()


def run():
    pipeline.update_all_nodes()
    # print(pipeline.source._samples_already_read / 500)


timer = QtCore.QTimer()
timer.timeout.connect(run)
frequency = pipeline.frequency
output_frequency = 3.
# timer.setInterval(1000. / frequency * 500)
timer.setInterval(1000. / output_frequency)

source.loop_the_file = False
source.MAX_SAMPLES_IN_CHUNK = int(frequency / output_frequency)
# envelope.disabled = True


if __name__ == '__main__':
    import sys

    timer.start()
    # timer.stop()

    # TODO: this runs when in iPython. It should not.
    # Start Qt event loop unless running in interactive mode or using pyside.
    # if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
    #     sys.exit(QtGui.QApplication.instance().exec_())

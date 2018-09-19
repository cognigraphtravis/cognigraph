from pyqtgraph.parametertree import parameterTypes
from PyQt5 import QtCore, QtGui
import pylsl

from ...helpers.pyqtgraph import MyGroupParameter
from ...nodes.sources import LSLStreamSource, FileSource


class SourceControls(MyGroupParameter):
    @property
    def SOURCE_CLASS(self):
        raise NotImplementedError

    def __init__(self, pipeline, **kwargs):
        self._pipeline = pipeline
        self.source_node = pipeline.source  # type: self.SOURCE_CLASS
        super().__init__(**kwargs)

    def create_node(self):
        self.source_node = self.SOURCE_CLASS()
        return self.source_node


class LSLStreamSourceControls(SourceControls):
    SOURCE_CLASS = LSLStreamSource

    STREAM_NAME_PLACEHOLDER = 'Click here to choose a stream'
    STREAM_NAMES_COMBO_NAME = 'Choose a stream: '

    def __init__(self, pipeline, **kwargs):

        kwargs['title'] = 'LSL stream'
        super().__init__(pipeline, **kwargs)

        stream_names = [info.name() for info in pylsl.resolve_streams()]
        values = [self.STREAM_NAME_PLACEHOLDER] + stream_names
        try:
            value = self.source_node.stream_name
        except AttributeError:
            value = self.STREAM_NAME_PLACEHOLDER
        stream_names_combo = parameterTypes.ListParameter(name=self.STREAM_NAMES_COMBO_NAME,
                                                          values=values, value=value)
        stream_names_combo.sigValueChanged.connect(self._on_stream_name_picked)
        self.stream_names_combo = self.addChild(stream_names_combo)

    def _on_stream_name_picked(self, param, value):
        # Update if needed
        if self.source_node.stream_name != value:
            self.source_node.stream_name = value

    def _remove_placeholder_option(self, default):
        stream_names_combo = self.param(self.STREAM_NAMES_COMBO_NAME)
        values = stream_names_combo.opts['values']  # type: list
        try:
            values.remove(self.STREAM_NAME_PLACEHOLDER)
            self.setLimits(values)
        except ValueError:  # The placeholder option has already been removed
            pass


class FileSourceControls(SourceControls):
    SOURCE_CLASS = FileSource

    FILE_PATH_STR_NAME = 'Path to file: '

    def __init__(self, pipeline, **kwargs):

        kwargs['title'] = 'Input file'
        super().__init__(pipeline, **kwargs)

        try:
            file_path = pipeline.source.file_path
        except:
            file_path = ''

        # Add LineEdit for choosing file
        file_path_str = parameterTypes.SimpleParameter(
                type='str', name=self.FILE_PATH_STR_NAME, value=file_path)

        file_path_str.sigValueChanged.connect(self._on_file_path_changed)

        self.file_path_str = self.addChild(file_path_str)

        # Add PushButton for choosing file
        file_path_button = parameterTypes.ActionParameter(
                type='action', name="Select data...")

        file_path_button.sigActivated.connect(self._choose_file)

        self.file_path_button = self.addChild(file_path_button)

    def _choose_file(self):
        filter_string = ''
        supported_exts = self._pipeline.source.SUPPORTED_EXTENSIONS
        for i, key in enumerate(supported_exts.keys()):
            if i > 0:
                filter_string += ';;'
            exts = supported_exts[key]
            ext_wildcards = ['*' + ext for ext in exts]
            ext_wildcards_str = ''
            for i, ext_wildcard in enumerate(ext_wildcards):
                if i == 0:
                    ext_wildcards_str += '('
                else:
                    ext_wildcards_str += ' '
                ext_wildcards_str += ext_wildcard
            ext_wildcards_str += ')'

            filter_string += '{} {}'.format(key, ext_wildcards_str)

        file_path = QtGui.QFileDialog.getOpenFileName(
                caption="Select Data",
                # filter="Brainvision (*.eeg *.vhdr *.vmrk)")
                filter=filter_string)

        if file_path != "":
            self.file_path_str.setValue(file_path[0])

    def _on_file_path_changed(self, param, value):
        self._pipeline.source.file_path = value

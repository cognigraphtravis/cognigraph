import platform
import time
from PyQt5.QtCore import QTimer
from vispy.gloo.util import _screenshot
from PIL import Image as im

# Import screenshoting software
# if platform.system() == 'Linux':
#     import pyscreenshot as ImageGrab  # ImageGrab replacement for Pillow
# else:
#     from PIL import ImageGrab


class ScreenRecorder:
    def __init__(self):
        self.is_recording = False
        self.sector = None

        self._start_time = None
        self._display_time = None  # Time in ms between switching images

        self._timer = QTimer()
        self._timer.timeout.connect(self._append_screenshot)

        self._images = []

    def start(self):
        self._images = []
        self._start_time = time.time()

        self.is_recording = True
        self._timer.start()

    def stop(self):
        self.is_recording = False
        self._timer.stop()

        duration = time.time() - self._start_time
        self._display_time = (duration * 1000) / len(self._images)

    def save(self, path):
        self._images[0].save(
            path,
            save_all=True,
            append_images=self._images[1:],
            duration=self._display_time,
            loop=0)

    def _append_screenshot(self):
        if self.sector is None:
            # self._images.append(ImageGrab.grab())
            self._images.append(im.fromarray(_screenshot()))
        else:
            # self._images.append(ImageGrab.grab(bbox=self.sector))
            self._images.append(im.fromarray(_screenshot()))

''' Config file for setuptools '''
from setuptools import setup

import os
VERSION = None

with open(os.path.join('cognigraph', '__init__.py'), 'r') as fid:
    for line in (line.strip() for line in fid):
        if line.startswith('__version__'):
            VERSION = line.split('=')[1].strip().strip('\'')
            break

setup(
    name='cognigraph',
    version=VERSION,
    install_requires=[
        'pyqtgraph',
        'pyqt5',
        'mne',
        'scipy>=1.0.0',
        'matplotlib',
        'pylsl',
        'h5py',
        'sympy',
        'sklearn',
        'pandas',
        'nibabel',
        'pyopengl',
        'pillow',
        'pyscreenshot',
        'numba',
        'vispy',
        'PyOpenGL',
        'PyOpenGL_accelerate'
    ]
)

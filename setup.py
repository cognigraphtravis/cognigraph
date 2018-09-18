''' Config file for setuptools '''
from setuptools import setup

setup(
    name='cognigraph',
    version='0.1.1',
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
        'numba',
        'nibabel',
        'pyopengl',
        'pillow',
        'pyscreenshot'
    ]
)

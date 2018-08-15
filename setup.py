''' Config file for setuptools '''
from setuptools import setup

setup(
    name='cognigraph',
    version='0.1',
    install_requires=[
        'pyqt5',
        'pyqtgraph',
        'mne==0.15',
        'scipy',
        'matplotlib',
        'pylsl',
        'h5py',
        'sympy',
        'sklearn',
        'pandas',
        'numba',
        'nibabel',
        'pyopengl'
    ]
)

# try:
#         import PyQt4
#         raise RuntimeError('Cognigraph requires that PyQt4 is NOT installed.')
# except ImportError:
#         pass

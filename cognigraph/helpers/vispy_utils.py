"""Group functions for color managment.

This file contains a bundle of functions that can be used to have a more
flexible control of diffrent problem involving colors (like turn an array /
string / faces into RBGA colors, defining the basic colormap object...)
"""
import logging

import numpy as np

from vispy.color.colormap import Colormap as VispyColormap

from matplotlib import cm
import matplotlib.colors as mplcol
from warnings import warn

# from .sigproc import normalize
# from .mesh import vispy_array
from functools import wraps

import logging

import numpy as np
from scipy.signal import fftconvolve

from vispy.visuals.transforms import STTransform, NullTransform
__all__ = ('Colormap', 'color2vb', 'array2colormap', 'cmap_to_glsl',
           'dynamic_color', 'color2faces', 'type_coloring', 'mpl_cmap',
           'color2tuple', 'mpl_cmap_index')

"""Surfaces (mesh) and volume utility functions."""
from scipy.spatial.distance import cdist

from vispy.geometry import MeshData
from vispy.geometry.isosurface import isosurface

# from .sigproc import smooth_3d


__all__ = ('vispy_array', 'convert_meshdata', 'volume_to_mesh',
           'smoothing_matrix', 'mesh_edges', 'laplacian_smoothing')

logger = logging.getLogger('cognigraph')


class Colormap(object):
    """Main colormap class.

    Parameters
    ----------
    cmap : string | inferno
        Matplotlib colormap
    clim : tuple/list | None
        Limit of the colormap. The clim parameter must be a tuple / list
        of two float number each one describing respectively the (min, max)
        of the colormap. Every values under clim[0] or over clim[1] will
        peaked.
    alpha : float | 1.0
        The opacity to use. The alpha parameter must be between 0 and 1.
    vmin : float | None
        Threshold from which every color will have the color defined using
        the under parameter bellow.
    under : tuple/string | 'dimgray'
        Matplotlib color for values under vmin.
    vmax : float | None
        Threshold from which every color will have the color defined using
        the over parameter bellow.
    over : tuple/string | 'darkred'
        Matplotlib color for values over vmax.
    translucent : tuple | None
        Set a specific range translucent. With f_1 and f_2 two floats, if
        translucent is :

            * (f_1, f_2) : values between f_1 and f_2 are set to translucent
            * (None, f_2) x <= f_2 are set to translucent
            * (f_1, None) f_1 <= x are set to translucent
    lut_len : int | 1024
        Number of levels for the colormap.
    interpolation : {None, 'linear', 'cubic'}
        Interpolation type. Default is None.

    Attributes
    ----------
    data : array_like
        Color data of shape (n_data, 4)
    shape : tuple
        Shape of the data.
    r : array_like
        Red levels.
    g : array_like
        Green levels.
    b : array_like
        Blue levels.
    rgb : array_like
        RGB levels.
    alpha : array_like
        Transparency level.
    glsl : vispy.colors.Colormap
        GL colormap version.
    """

    def __init__(self, cmap='viridis', clim=None, vmin=None, under=None,
                 vmax=None, over=None, translucent=None, alpha=1.,
                 lut_len=1024, interpolation=None):
        """Init."""
        # Keep color parameters into a dict :
        self._kw = dict(cmap=cmap, clim=clim, vmin=vmin, vmax=vmax,
                        under=under, over=over, translucent=translucent,
                        alpha=alpha)
        # Color conversion :
        if isinstance(cmap, np.ndarray):
            assert (cmap.ndim == 2) and (cmap.shape[-1] in (3, 4))
            # cmap = single color :
            if (cmap.shape[0] == 1) and isinstance(interpolation, str):
                logger.debug("Colormap : unique color repeated.")
                data = np.tile(cmap, (lut_len, 1))
            elif (cmap.shape[0] == lut_len) or (interpolation is None):
                logger.debug("Colormap : Unique repeated.")
                data = cmap
            else:
                from scipy.interpolate import interp2d
                n_ = cmap.shape[1]
                x, y = np.linspace(0, 1, n_), np.linspace(0, 1, cmap.shape[0])
                f = interp2d(x, y, cmap, kind=interpolation)
                # Interpolate colormap :
                data = f(x, np.linspace(0, 1, lut_len))
        elif isinstance(cmap, str):
            data = array2colormap(np.linspace(0., 1., lut_len), **self._kw)
        # Alpha correction :
        if data.shape[-1] == 3:
            data = np.c_[data, np.full((data.shape[0],), alpha)]
        # NumPy float32 conversion :
        self._data = vispy_array(data)

    def to_rgba(self, data):
        """Turn a data vector into colors using colormap properties.

        Parameters
        ----------
        data : array_like
            Vector of data of shape (n_data,).

        Returns
        -------
        color : array_like
            Array of colors of shape (n_data, 4)
        """
        if isinstance(self._kw['cmap'], np.ndarray):
            return self._data
        else:
            return array2colormap(data, **self._kw)

    def __len__(self):
        """Get the number of colors in the colormap."""
        return self._data.shape[0]

    def __getitem__(self, name):
        """Get a color item."""
        return self._kw[name]

    @property
    def data(self):
        """Get colormap data."""
        return self._data

    @property
    def shape(self):
        """Get the shape of the data."""
        return self._data.shape

    @property
    def glsl(self):
        """Get a glsl version of the colormap."""
        return cmap_to_glsl(lut_len=len(self), **self._kw)

    @property
    def r(self):
        """Get red levels."""
        return self._data[:, 0]

    @property
    def g(self):
        """Get green levels."""
        return self._data[:, 1]

    @property
    def b(self):
        """Get blue levels."""
        return self._data[:, 2]

    @property
    def rgb(self):
        """Get rgb levels."""
        return self._data[:, 0:3]

    @property
    def alpha(self):
        """Get transparency level."""
        return self._data[:, -1]


def color2vb(color=None, default=(1., 1., 1.), length=1, alpha=1.0,
             faces_index=False):
    """Turn into a RGBA compatible color format.

    This function can tranform a tuple of RGB, a matplotlib color or an
    hexadecimal color into an array of RGBA colors.

    Parameters
    ----------
    color : None/tuple/string | None
        The color to use. Can either be None, or a tuple (R, G, B),
        a matplotlib color or an hexadecimal color '#...'.
    default : tuple | (1,1,1)
        The default color to use instead.
    length : int | 1
        The length of the output array.
    alpha : float | 1
        The opacity (Last digit of the RGBA tuple).
    faces_index : bool | False
        Specify if the returned color have to be compatible with faces index
        (e.g a (n_color, 3, 4) array).

    Return
    ------
    vcolor : array_like
        Array of RGBA colors of shape (length, 4).
    """
    # Default or static color :
    if (color is None) or isinstance(color, (str, tuple, list, np.ndarray)):
        if color is None:  # Default
            coltuple = default
        elif isinstance(color, (tuple, list, np.ndarray)):  # Static
            color = np.squeeze(color).ravel()
            if len(color) == 4:
                alpha = color[-1]
                color = color[0:-1]
            coltuple = color
        elif isinstance(color, str) and (color[0] is not '#'):  # Matplotlib
            # Check if the name is in the Matplotlib database :
            if color in mplcol.cnames.keys():
                coltuple = mplcol.hex2color(mplcol.cnames[color])
            else:
                warn("The color name " + color + " is not in the matplotlib "
                     "database. Default color will be used instead.")
                coltuple = default
        elif isinstance(color, str) and (color[0] is '#'):  # Hexadecimal
            try:
                coltuple = mplcol.hex2color(color)
            except:
                warn("The hexadecimal color " + color + " is not valid. "
                     "Default color will be used instead.")
                coltuple = default
        # Set the color :
        vcolor = np.concatenate((np.array([list(coltuple)] * length),
                                 alpha * np.ones((length, 1),
                                                 dtype=np.float32)), axis=1)

        # Faces index :
        if faces_index:
            vcolor = np.tile(vcolor[:, np.newaxis, :], (1, 3, 1))

        return vcolor.astype(np.float32)
    else:
        raise ValueError(str(type(color)) + " is not a recognized type of "
                         "color. Use None, tuple or string")


def color2tuple(color, astype=np.float32, rmalpha=True, roundto=2):
    """Return a RGB tuple of the color.

    Parameters
    ----------
    color : None/tuple/string | None
        The color to use. Can either be None, or a tuple (R, G, B),
        a matplotlib color or an hexadecimal color '#...'.
    astype : type | np.float32
        The final color type.
    rmalpha : bool | True
        Specify if the alpha component have to be deleted.
    roundto : int | 2
        Number of digits per RGB.

    Returns
    -------
    coltuple: tuple
        Tuple of colors.
    """
    # Get the converted color :
    ccol = color2vb(color).ravel().astype(astype)
    # Round it :
    ccol = np.ndarray.tolist(np.around(ccol, roundto))
    if rmalpha:
        return tuple(ccol[0:-1])
    else:
        return tuple(ccol)


def array2colormap(x, cmap='inferno', clim=None, alpha=1.0, vmin=None,
                   vmax=None, under='dimgray', over='darkred',
                   translucent=None, faces_render=False):
    """Transform an array of data into colormap (array of RGBA).

    Parameters
    ----------
    x: array
        Array of data
    cmap : string | inferno
        Matplotlib colormap
    clim : tuple/list | None
        Limit of the colormap. The clim parameter must be a tuple / list
        of two float number each one describing respectively the (min, max)
        of the colormap. Every values under clim[0] or over clim[1] will
        peaked.
    alpha : float | 1.0
        The opacity to use. The alpha parameter must be between 0 and 1.
    vmin : float | None
        Threshold from which every color will have the color defined using
        the under parameter bellow.
    under : tuple/string | 'dimgray'
        Matplotlib color for values under vmin.
    vmax : float | None
        Threshold from which every color will have the color defined using
        the over parameter bellow.
    over : tuple/string | 'darkred'
        Matplotlib color for values over vmax.
    translucent : tuple | None
        Set a specific range translucent. With f_1 and f_2 two floats, if
        translucent is :

            * (f_1, f_2) : values between f_1 and f_2 are set to translucent
            * (None, f_2) x <= f_2 are set to translucent
            * (f_1, None) f_1 <= x are set to translucent
    faces_render : boll | False
        Precise if the render should be applied to faces

    Returns
    -------
    color : array_like
        Array of RGBA colors
    """
    # ================== Check input argument types ==================
    # Force data to be an array :
    x = np.asarray(x)

    # Check clim :
    clim = (None, None) if clim is None else list(clim)
    assert len(clim) == 2

    # ---------------------------
    # Check alpha :
    if (alpha < 0) or (alpha > 1):
        warn("The alpha parameter must be >= 0 and <= 1.")

    # ================== Define colormap ==================
    sc = cm.ScalarMappable(cmap=cmap)

    # Fix limits :
    norm = mplcol.Normalize(vmin=clim[0], vmax=clim[1])
    sc.set_norm(norm)

    # ================== Apply colormap ==================
    # Apply colormap to x :
    x_cmap = np.array(sc.to_rgba(x, alpha=alpha))

    # ================== Colormap (under, over) ==================
    if (vmin is not None) and (under is not None):
        under = color2vb(under)  # if isinstance(under, str) else under
        x_cmap[x < vmin, :] = under
    if (vmax is not None) and (over is not None):
        over = color2vb(over)  # if isinstance(over, str) else over
        x_cmap[x > vmax, :] = over

    # ================== Transparency ==================
    x_cmap = _transclucent_cmap(x, x_cmap, translucent)

    # Faces render (repeat the color to other dimensions):
    if faces_render:
        x_cmap = np.transpose(np.tile(x_cmap[..., np.newaxis],
                                      (1, 1, 3)), (0, 2, 1))

    return x_cmap.astype(np.float32)


def _transclucent_cmap(x, x_cmap, translucent, smooth=None):
    """Sub function to define transparency."""
    if translucent is not None:
        is_num = [isinstance(k, (int, float)) for k in translucent]
        assert len(translucent) == 2 and any(is_num)
        if all(is_num):                # (f_1, f_2)
            trans_x = np.logical_and(translucent[0] <= x, x <= translucent[1])
        elif is_num == [True, False]:  # (f_1, None)
            trans_x = translucent[0] <= x
        elif is_num == [False, True]:  # (None, f_2)
            trans_x = x <= translucent[1]
        x_cmap[..., -1] = np.invert(trans_x)
        if isinstance(smooth, int):
            alphas = x_cmap[:, -1]
            alphas = np.convolve(alphas, np.hanning(smooth), 'valid')
            alphas /= max(alphas.max(), 1.)
            x_cmap[smooth - 1::, -1] = alphas
    return x_cmap


def cmap_to_glsl(limits=None, lut_len=1024, color=None, **kwargs):
    """Get a glsl colormap.

    Parameters
    ----------
    limits : tuple | None
        Color limits for the object. Must be a tuple of two floats.
    lut_len : int | 1024
        Number of levels for the colormap.
    color : string | None
        Use a unique color for the colormap.
    kwarg : dict | None
        Additional inputs to pass to the array2colormap function.

    Returns
    -------
    cmap : vispy.color.Colormap
        VisPy colormap instance.
    """
    if limits is None:
        limits = (0., 1.)
    assert len(limits) == 2
    # Color transform :
    vec = np.linspace(limits[0], limits[1], lut_len)
    if color is None:  # colormap
        cmap = VispyColormap(array2colormap(vec, **kwargs))
    else:              # uniform color
        translucent = kwargs.get('translucent', None)
        rep_col = color2vb(color, length=lut_len)
        cmap_trans = _transclucent_cmap(vec, rep_col, translucent)
        cmap = VispyColormap(cmap_trans)

    return cmap


def dynamic_color(color, x, dynamic=(0., 1.)):
    """Dynamic color changing.

    Parameters
    ----------
    color : array_like
        The color to dynamic change. color must have a shape
        of (N, 4) RGBA colors
    x : array_like
        Dynamic values for color. x must have a shape of (N,)
    dynamic : tuple | (0.0, 1.0)
        Control the dynamic of color.

    Returns
    -------
    colordyn : array_like
        Dynamic color with a shape of (N, 4)
    """
    x = x.ravel()
    # Check inputs types :
    if color.shape[1] != 4:
        raise ValueError("Color must be RGBA")
    if color.shape[0] != len(x):
        raise ValueError("The length of color must be the same as"
                         " x: " + str(len(x)))
    # Normalise x :
    if dynamic[0] < dynamic[1]:
        x_norm = normalize(x, tomin=dynamic[0], tomax=dynamic[1])
    else:
        x_norm = np.full((len(x),), dynamic[0], dtype=np.float)
    # Update color :
    color[:, 3] = x_norm
    return color


def color2faces(color, length):
    """Pass a simple color to faces shape.

    Parameters
    ----------
    color : RGBA tuple
        Tuple of RGBA colors
    length : tuple
        Length of faces

    Returns
    -------
    color_face : array_like
        The color adapted for faces
    """
    color = np.asarray(color).ravel()
    colort = np.tile(np.array(color)[..., np.newaxis, np.newaxis],
                     (1, length, 3))
    return np.transpose(colort, (1, 2, 0))


def colorclip(x, th, kind='under'):
    """Force an array to have clipping values.

    Parameters
    ----------
    x : array_like
        Array of data.
    th : float
        The threshold to use.
    kind : string | 'under'
        Use eiher 'under' or 'over' for fore the array to clip for every
        values respectively under or over th.

    Returns
    -------
    x : array_like
        The clipping array.
    """
    if kind is 'under':
        idx = x < th
    elif kind is 'over':
        idx = x > th
    x[idx] = th
    return x


def type_coloring(color=None, n=1, data=None, rnd_dyn=(0.3, 0.9), clim=None,
                  cmap='viridis', vmin=None, under=None, vmax=None, over=None,
                  unicolor='gray'):
    """Switch between different coloring types.

    This function can be used to color a signal using random, uniform or
    dynamic colors.

    Parameters
    ----------
    color : string/tuple/array | None
        Choose how to color signals. Use None (or 'rnd', 'random') to
        generate random colors. Use 'uniform' (see the unicolor
        parameter) to define the same color for all signals. Use
        'dynamic' to have a dynamic color according to data values.
    n : int | 1
        The number of colors to generate in case of random or uniform
        colors.
    data : array_like | None
        The data to convert into color if the color type is dynamic.
        If this parameter is ignored, a default linear spaced vector of
        length n will be used instead.
    rnd_dyn : tuple | (.3, .9)
        Define the dynamic of random color. This parameter is active
        only if the color parameter is turned to None (or 'rnd' /
        'random').
    cmap : string | 'inferno'
        Matplotlib colormap (parameter active for 'dyn_minmax' and
        'dyn_time' color).
    clim : tuple/list | None
        Limit of the colormap. The clim parameter must be a tuple /
        list of two float number each one describing respectively the
        (min, max) of the colormap. Every values under clim[0] or over
        clim[1] will peaked (parameter active for 'dyn_minmax' and
        'dyn_time' color).
    alpha : float | 1.0
        The opacity to use. The alpha parameter must be between 0 and 1
        (parameter active for 'dyn_minmax' and 'dyn_time' color).
    vmin : float | None
        Threshold from which every color will have the color defined
        using the under parameter bellow (parameter active for
        'dyn_minmax' and 'dyn_time' color).
    under : tuple/string | 'gray'
        Matplotlib color for values under vmin (parameter active for
        'dyn_minmax' and 'dyn_time' color).
    vmax : float | None
        Threshold from which every color will have the color defined
        using the over parameter bellow (parameter active for
        'dyn_minmax' and 'dyn_time' color).
    over : tuple/string | 'red'
        Matplotlib color for values over vmax (parameter active for
        'dyn_minmax' and 'dyn_time' color).
    unicolor : tuple/string | 'gray'
        The color to use in case of uniform color.
    """
    # ---------------------------------------------------------------------
    # Random color :
    if color in [None, 'rnd', 'random']:
        # Create a (m, 3) color array :
        colout = np.random.uniform(size=(n, 3), low=rnd_dyn[0], high=rnd_dyn[1]
                                   )

    # ---------------------------------------------------------------------
    # Dynamic color :
    elif color == 'dynamic':
        # Generate a linearly spaced vector for None data :
        if data is None:
            data = np.arange(n)
        # Get colormap as (n, 3):
        colout = array2colormap(data.ravel(), cmap=cmap, clim=clim, vmin=vmin,
                                vmax=vmax, under=under, over=over)[:, 0:3]

    # ---------------------------------------------------------------------
    # Uniform color :
    elif color == 'uniform':
        # Create a (m, 3) color array :
        colout = color2vb(unicolor, length=n)[:, 0:3]

    # ---------------------------------------------------------------------
    # Not found color :
    else:
        raise ValueError("The color parameter is not recognized.")

    return colout.astype(np.float32)


def mpl_cmap(invert=False):
    """Get the list of matplotlib colormaps.

    Parameters
    ----------
    invert : bool | False
        Get the list of inverted colormaps.

    Returns
    -------
    cmap_lst: list
        list of available matplotlib colormaps.
    """
    # Full list of colormaps :
    fullmpl = list(cm.datad.keys()) + list(cm.cmaps_listed.keys())
    # Get the list of cmaps (inverted or not) :
    if invert:
        cmap_lst = [k for k in fullmpl if k.find('_r') + 1]
    else:
        cmap_lst = [k for k in fullmpl if not k.find('_r') + 1]

    # Sort the list :
    cmap_lst.sort()

    return cmap_lst


def mpl_cmap_index(cmap, cmaps=None):
    """Find the index of a colormap.

    Parameters
    ----------
    cmap : string
        Colormap name.
    cmaps : list | None
        List of colormaps.

    Returns
    -------
    idx : int
        Index of the colormap.
    invert : bool
        Boolean value indicating if it's a reversed colormap.
    """
    # Find if it's a reversed colormap :
    invert = bool(cmap.find('_r') + 1)
    # Get list of colormaps :
    if cmaps is None:
        cmap = cmap.replace('_r', '')
        cmaps = mpl_cmap()
        return np.where(np.char.find(cmaps, cmap) + 1)[0][0], invert
    else:
        return cmaps.index(cmap), invert


"""Wrappers."""


__all__ = ['wrap_properties']


def wrap_properties(fn):
    """Run properties if not None."""
    @wraps(fn)
    def wrapper(self, value):
        if value is not None:
            fn(self, value)
    return wrapper


"""This script contains some usefull signal processing functions."""


__all__ = ('normalize', 'derivative', 'tkeo', 'zerocrossing', 'power_of_ten',
           'averaging', 'normalization', 'smoothing', 'smooth_3d')

logger = logging.getLogger('cognigraph')


def normalize(x, tomin=0., tomax=1.):
    """Normalize the array x between tomin and tomax.

    Parameters
    ----------
    x : array_like
        The array to normalize
    tomin : int/float | 0.
        Minimum of returned array

    tomax : int/float | 1.
        Maximum of returned array

    Returns
    -------
    xn : array_like
        The normalized array
    """
    if x.size:
        x = np.float32(x)
        xm, xh = np.float32(x.min()), np.float32(x.max())
        if xm != xh:
            coef = (tomax - tomin) / (xh - xm)
            np.subtract(x, xh, out=x)
            np.multiply(x, coef, out=x)
            np.add(x, tomax, out=x)
            return x
            # return tomax - (((tomax - tomin) * (xh - x)) / (xh-xm))
        else:
            logger.debug("Normalization has been ignored because minimum and "
                         "maximum are both equal to " + str(xm))
            np.multiply(x, tomax, out=x)
            np.divide(x, xh, out=x)
            return x
    else:
        return x


def derivative(x, window, sf):
    """Compute first derivative of signal.

    Equivalent to np.gradient function

    Parameters
    ----------
    x : array_like
        Signal
    window : int
        Time (ms) window to compute first derivative
    sf : int
        Downsampling frequency
    """
    length = x.size
    step = int(window / (1000 / sf))
    tail = np.zeros(shape=(int(step / 2),))
    deriv = np.r_[tail, x[step:length] - x[0:length - step], tail]
    deriv = np.abs(deriv)
    # Check size
    if deriv.size < length:
        missing_pts = length - deriv.size
        tail = np.zeros(missing_pts)
        deriv = np.r_[deriv, tail]

    return deriv


def tkeo(x):
    """Calculate the TKEO of a given recording by using 2 samples.

    github.com/lvanderlinden/OnsetDetective/blob/master/OnsetDetective/tkeo.py

    Parameters
    ----------
    x : array_like
        Row vector of data.

    Returns
    -------
    a_tkeo : array_like
        Row vector containing the tkeo per sample.
    """
    # Create two temporary arrays of equal length, shifted 1 sample to the
    # right and left and squared:
    i = x[1:-1] * x[1:-1]
    j = x[2:] * x[:-2]

    # Calculate the difference between the two temporary arrays:
    a_tkeo = i - j
    return a_tkeo


def zerocrossing(data):
    """Find zero-crossings index of a signal.

    Parameters
    ----------
    x: array_like
        Data

    Returns
    -------
    index : array_like
        Row vector containing zero-crossing index.
    """
    pos = data > 0
    npos = ~pos
    return ((pos[:-1] & npos[1:]) | (npos[:-1] & pos[1:])).nonzero()[0] + 1


def power_of_ten(x, e=3):
    """Power of ten format.

    Parameters
    ----------
    x : float
        The floating point to transform.
    e : int | 2
        If x is over 10 ** -e and bellow 10 ** e, this function doesn't
        change the format.

    Returns
    -------
    xtronc: float
        The troncate version of x.
    power: int
        The power of ten to retrieve x.
    """
    sign = np.sign(x)
    x = np.abs(x)
    stx = str(x)
    if 0 < x <= 10 ** -e:  # x is a power of e- :
        if stx.find('e-') + 1:  # Format : 'xe-y'
            sp = stx.split('e-')
            return float(sp[0]), -int(sp[1])
        else:  # Format : 0.000x
            sp = stx.split('.')[1]
            id_l = 0
            while sp[id_l] == '0':
                id_l += 1
            id_l += 1
            return (sign * x) * (10 ** id_l), -id_l
    elif x >= 10 ** e:  # x is a power of e :
        if stx.find('e') + 1:  # Format : 'xey'
            sp = stx.split('e')
            return float(sp[0]), -int(sp[1])
        else:
            k = e
            while x % (10 ** k) != x:
                k += 1
            return (sign * x) / (10 ** (k - 1)), k - 1
    else:
        return sign * x, 0


def averaging(ts, n_window, axis=-1, overlap=0., window='flat'):
    """Take the mean of an ndarray.

    Parameters
    ----------
    ts : array_like
        Array of data to take the mean.
    n_window : int
        Number of sample per window.
    axis : int | -1
        Axis along which take the mean. By default, the last axis.
    overlap : float | None
        Overlap of successive window (0 <= overlap < 1). By default, no overlap
        is performed.
    window : {'flat', 'hanning', 'hamming', 'bartlett', 'blackman'}
        Windowing method.

    Returns
    -------
    average : array_like
        The averaged signal.
    """
    # Checking :
    assert isinstance(ts, np.ndarray)
    assert isinstance(axis, int) and axis <= ts.ndim - 1
    assert isinstance(n_window, int) and n_window < ts.shape[axis]
    assert isinstance(overlap, (float, int)) and 0. <= overlap < 1.
    assert window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']

    # Get axis :
    npts = ts.shape[axis]
    axis = ts.ndim - 1 if axis == -1 else axis

    # Get overlap step in samples :
    n_overlap = int(np.round(n_window * (1. - overlap)))

    # Build the index vector :
    ind = np.c_[np.arange(0, npts - n_window, n_overlap),
                np.arange(n_window, npts, n_overlap)]
    ind = np.vstack((ind, [npts - 1 - n_window, npts - 1]))  # add last window
    n_ind = ind.shape[0]

    # Get the window :
    if window == 'flat':  # moving average
        win = np.ones(n_window, 'd')
    else:
        win = eval('np.' + window + '(n_window)')

    rsh = tuple(1 if i != axis else -1 for i in range(ts.ndim))
    win = win.reshape(*rsh)

    # Define the averaging array :
    av_shape = tuple(k if i != axis else n_ind for i, k in enumerate(ts.shape))
    average = np.zeros(av_shape, dtype=float)

    # Compute averaging :
    sl_ts = [slice(None)] * ts.ndim
    sl_av = sl_ts.copy()
    for k in range(n_ind):
        sl_ts[axis], sl_av[axis] = slice(ind[k, 0], ind[k, 1]), slice(k, k + 1)
        average[tuple(sl_av)] += (ts[tuple(sl_ts)] * win).mean(axis=axis,
                                                               keepdims=True)

    return average


def normalization(data, axis=-1, norm=None, baseline=None):
    """Data normalization.

    Parameters
    ----------
    data : array_like
        Array of data.
    axis : int | -1
        Array along which to perform the normalization.
    norm : int | None
        The normalization type. Use :
            * 0 : no normalization
            * 1 : subtract the mean
            * 2 : divide by the mean
            * 3 : subtract then divide by the mean
            * 4 : subtract the mean then divide by deviation
    baseline : tuple | None
        Baseline period to consider. If None, the entire signal is used.

    Returns
    -------
    data_n : array_like
        The normalized array.
    """
    assert isinstance(data, np.ndarray)
    # assert norm in [None, ]

    # Take data in baseline (if defined) :
    if (baseline is not None) and (len(baseline) == 2):
        sl = [slice(None)] * data.ndim
        sl[axis] = slice(baseline[0], baseline[1])
        _data = data[tuple(sl)]
    else:
        _data = None

    if norm in [0, None]:  # don't normalize
        return data
    elif norm in [1, 2, 3, 4]:
        kw = {'axis': axis, 'keepdims': True}
        d_m = _data.mean(**kw) if _data is not None else data.mean(**kw)
        if norm == 1:  # subtract the mean
            data -= d_m
        elif norm == 2:  # divide by the mean
            d_m[d_m == 0] = 1.
            data /= d_m
        elif norm == 3:  # subtract then divide by the mean
            data -= d_m
            d_m[d_m == 0] = 1.
            data /= d_m
        elif norm == 4:  # z-score
            d_std = _data.mean(**kw) if _data is not None else data.mean(**kw)
            d_std[d_std == 0] = 1.
            data -= d_m
            data /= d_std


def smoothing(x, n_window=10, window='hanning'):
    """Smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    Parameters
    ----------
    x : array_like
        1-D array to smooth.
    n_window : int | 10
        Window length.
    window : string, array_like | 'hanning'
        Use either 'flat', 'hanning', 'hamming', 'bartlett', 'blackman' or pass
        a numpy array of length n_window.

    Returns
    -------
        The smoothed signal
    """
    n_window = int(n_window)
    assert isinstance(x, np.ndarray) and x.ndim == 1
    assert len(x) > n_window
    assert isinstance(window, (str, np.ndarray))
    if isinstance(window, str):
        assert window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']
    elif isinstance(window, np.ndarray):
        assert len(window) == n_window

    if n_window < 3:
        return x

    s = np.r_[2 * x[0] - x[n_window:1:-1], x, 2 * x[-1] - x[-1:-n_window:-1]]
    if window == 'flat':  # Moving average
        w = np.ones((n_window,))
    else:
        w = eval('np.' + window + '(n_window)')

    y = np.convolve(w / w.sum(), s, mode='same')
    return y[n_window - 1:-n_window + 1]


def smooth_3d(vol, smooth_factor=3, correct=True):
    """Smooth a 3-D volume.

    Parameters
    ----------
    vol : array_like
        The volume of shape (N, M, P)
    smooth_factor : int | 3
        The smoothing factor.

    Returns
    -------
    vol_smooth : array_like
        The smooth volume with the same shape as vol.
    """
    tf = NullTransform()
    # No smoothing :
    if (not isinstance(smooth_factor, int)) or (smooth_factor < 3):
        return vol, tf
    # Smoothing array :
    sz = np.full((3,), smooth_factor, dtype=int)
    smooth = np.ones([smooth_factor] * 3) / np.prod(sz)
    # Apply smoothing :
    sm = fftconvolve(vol, smooth, mode='same')
    if correct:
        # Get the shape of the vol and the one with 'full' convolution :
        vx, vy, vz = vol.shape
        vcx, vcy, vcz = np.array([vx, vy, vz]) + smooth_factor - 1
        # Define transform :
        sc = [vx / vcx, vy / vcy, vz / vcz]
        tr = .5 * np.array([smooth_factor] * 3)
        tf = STTransform(scale=sc, translate=tr)
    return sm, tf




logger = logging.getLogger('cognigraph')


def vispy_array(data, dtype=np.float32):
    """Check and convert array to be compatible with buffers.

    Parameters
    ----------
    data : array_like
        Array of data.
    dtype : type | np.float32
        Futur type of the array.

    Returns
    -------
    data : array_like
        Contiguous array of type dtype.
    """
    if not data.flags['C_CONTIGUOUS']:
        data = np.ascontiguousarray(data, dtype=dtype)
    if data.dtype != dtype:
        data = data.astype(dtype, copy=False)
    return data


def convert_meshdata(vertices=None, faces=None, normals=None, meshdata=None,
                     invert_normals=False, transform=None):
    """Convert mesh data to be compatible with vispy.

    Parameters
    ----------
    vertices : array_like | None
        Vertices of the template of shape (N, 3) or (N, 3, 3) if indexed by
        faces.
    faces : array_like | None
        Faces of the template of shape (M, 3)
    normals : array_like | None
        The normals to each vertex, with the same shape as vertices.
    meshdata : VisPy.MeshData | None
        VisPy MeshData object.
    invert_normals : bool | False
        If the brain appear to be black, use this parameter to invert normals.
    transform : visPy.transform | None
        VisPy transformation to apply to vertices ans normals.

    Returns
    -------
    vertices : array_like
        Vertices of shape (N, 3)
    faces : array_like
        Faces of the template of shape (M, 3)
    normals : array_like
        The normals of shape (M, 3, 3).
    """
    # Priority to meshdata :
    if meshdata is not None:
        vertices = meshdata.get_vertices()
        faces = meshdata.get_faces()
        normals = meshdata.get_vertex_normals()
        logger.debug('Indexed faces normals converted // extracted')
    else:
        # Check if faces index start at zero (Matlab like):
        if faces.min() != 0:
            faces -= faces.min()
        # Get normals if None :
        if (normals is None) or (normals.ndim != 2):
            md = MeshData(vertices=vertices, faces=faces)
            normals = md.get_vertex_normals()
            logger.debug('Indexed faces normals converted // extracted')
    assert vertices.ndim == 2

    # Invert normals :
    norm_coef = -1. if invert_normals else 1.
    normals *= norm_coef

    # Apply transformation :
    if transform is not None:
        vertices = transform.map(vertices)[..., 0:-1]
        normals = transform.map(normals)[..., 0:-1]

    # Type checking :
    vertices = vispy_array(vertices)
    faces = vispy_array(faces, np.uint32)
    normals = vispy_array(normals)

    return vertices, faces, normals


def volume_to_mesh(vol, smooth_factor=3, level=None, **kwargs):
    """Convert a volume into a mesh with vertices, faces and normals.

    Parameters
    ----------
    vol : array_like
        The volume of shape (N, M, P)
    smooth_factor : int | 3
        The smoothing factor to apply to the volume.
    level : int | None
        Level to extract.
    kwargs : dict | {}
        Optional arguments to pass to convert_meshdata.

    Returns
    -------
    vertices : array_like
        Mesh vertices.
    faces : array_like
        Mesh faces.
    normals : array_like
        Mesh normals.
    """
    # Smooth the volume :
    vol_s, tf = smooth_3d(vol, smooth_factor, correct=True)
    # Extract vertices and faces :
    if level is None:
        level = .5
    elif isinstance(level, int):
        vol_s[vol_s != level] = 0
        level = .5
    vert_n, faces_n = isosurface(vol_s, level=level)
    # Smoothing compensation :
    vert_n = tf.map(vert_n)[:, 0:-1]
    # Convert to meshdata :
    vertices, faces, normals = convert_meshdata(vert_n, faces_n, **kwargs)
    return vertices, faces, normals


def smoothing_matrix(vertices, adj_mat, smoothing_steps=20):
    """Create a smoothing matrix.

    This function  can be used to interpolate data defined for a subset of
    vertices onto mesh with an adjancency matrix given by adj_mat.

    This function is a copy from the PySurfer package. See :
    https://github.com/nipy/PySurfer/blob/master/surfer/utils.py

    Parameters
    ----------
    vertices : array_like
        Vertex indices of shape (N,)
    adj_mat : sparse matrix
        N x N adjacency matrix of the full mesh.
    smoothing_steps : int
        Number of smoothing steps. If smoothing_steps is None, as many
        smoothing steps are applied until the whole mesh is filled with
        with non-zeros. Only use this option if the vertices correspond to a
        subsampled version of the mesh.
    Returns
    -------
    smooth_mat : sparse matrix
        smoothing matrix with size N x len(vertices)
    """
    from scipy import sparse

    e = adj_mat.copy()
    e.data[e.data == 2] = 1
    n_vertices = e.shape[0]
    e = e + sparse.eye(n_vertices, n_vertices)
    idx_use = vertices
    smooth_mat = 1.0
    n_iter = smoothing_steps if smoothing_steps is not None else 1000
    for k in range(n_iter):
        e_use = e[:, idx_use]

        data1 = e_use * np.ones(len(idx_use))
        idx_use = np.where(data1)[0]
        scale_mat = sparse.dia_matrix((1 / data1[idx_use], 0),
                                      shape=(len(idx_use), len(idx_use)))

        smooth_mat = scale_mat * e_use[idx_use, :] * smooth_mat

        if smoothing_steps is None and len(idx_use) >= n_vertices:
            break

    # Make sure the smoothing matrix has the right number of rows
    # and is in COO format
    smooth_mat = smooth_mat.tocoo()
    smooth_mat = sparse.coo_matrix((smooth_mat.data,
                                    (idx_use[smooth_mat.row],
                                     smooth_mat.col)),
                                   shape=(n_vertices,
                                          len(vertices)))

    return smooth_mat


def mesh_edges(faces):
    """Get sparse matrix with edges as an adjacency matrix.

    This function is a copy from the PySurfer package. See :
    https://github.com/nipy/PySurfer/blob/master/surfer/utils.py

    Parameters
    ----------
    faces : array_like
        The mesh faces of shape (n_faces, 3).
    Returns
    -------
    edges : sparse matrix
        The adjacency matrix.
    """
    from scipy import sparse
    npoints = np.max(faces) + 1
    nfaces = len(faces)
    a, b, c = faces.T
    edges = sparse.coo_matrix((np.ones(nfaces), (a, b)),
                              shape=(npoints, npoints))
    edges = edges + sparse.coo_matrix((np.ones(nfaces), (b, c)),
                                      shape=(npoints, npoints))
    edges = edges + sparse.coo_matrix((np.ones(nfaces), (c, a)),
                                      shape=(npoints, npoints))
    edges = edges + edges.T
    edges = edges.tocoo()
    return edges


def laplacian_smoothing(vertices, faces, n_neighbors=-1):
    """Apply a laplacian smoothing to vertices.

    Parameters
    ----------
    vertices : array_like
        Array of vertices.
    vertices : array_like
        Array of faces.
    n_neighbors : int | -1
        Specify maximum number of closest neighbors to take into account in the
        mean.

    Returns
    -------
    new_vertices : array_like
        New smoothed vertices.
    """
    assert vertices.ndim == 2 and vertices.shape[1] == 3
    assert faces.ndim == 2 and faces.shape[1] == 3
    assert n_neighbors >= -1 and isinstance(n_neighbors, int)
    n_vertices = vertices.shape[0]
    new_vertices = np.zeros_like(vertices)
    for k in range(n_vertices):
        # Find connected vertices :
        faces_idx = np.where(faces == k)[0]
        u_faces_idx = np.unique(np.ravel(faces[faces_idx, :])).tolist()
        u_faces_idx.remove(k)
        # Select closest connected vertices :
        if n_neighbors == -1:
            to_smooth = u_faces_idx
        else:
            norms = cdist(vertices[[k], :], vertices[u_faces_idx, :]).ravel()
            n_norm = min(n_neighbors, len(norms))
            to_smooth = np.array(u_faces_idx)[np.argsort(norms)[0:n_norm]]
        # Take the mean of selected vertices :
        new_vertices[k, :] = vertices[to_smooth, :].mean(0).reshape(1, -1)
    return new_vertices

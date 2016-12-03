from __future__ import division, print_function, absolute_import
from .extraction import (extract_sparse, extract, load_classifier,
                         load_default_classifier, colorize)
from .color import match_lightness
from . import image
from .datasets import load_image_list
from .checker import checker_main


VERSION = (0, 2, 1)
ISRELEASED = True
__version__ = '{0}.{1}.{2}'.format(*VERSION)
if not ISRELEASED:
    __version__ += '.git'

__all__ = ['extract_sparse',
           'extract',
           'load_classifier',
           'load_default_classifier',
           'colorize',
           'match_lightness',
           'image',
           'load_image_list',
           'checker_main']

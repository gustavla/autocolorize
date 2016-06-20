import os

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
RES_DIR = os.path.join(SCRIPT_DIR, 'res')
WEIGHTS_DIR = os.path.expanduser('~/.autocolorize')
WEIGHTS_URL = 'http://people.cs.uchicago.edu/~larsson/colorization/res/autocolorize.caffemodel.h5'
TOTAL_SIZE = 588203256

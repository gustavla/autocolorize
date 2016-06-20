from __future__ import division, print_function, absolute_import
from . import config
import numpy as np
import os


def download_file(url, fn):
    import requests
    r = requests.get(url, stream=True)
    with open(fn, 'wb') as f:
        BUFFER = 1024 * 32
        BARS = 40
        cur_size = 0
        for i, chunk in enumerate(r.iter_content(chunk_size=BUFFER)):
            if chunk:
                cur_size += BUFFER
                if i % 20 == 0:
                    progress = cur_size / config.TOTAL_SIZE
                    n = int(np.ceil(progress * BARS))
                    bar = '#' * n + '.' * (BARS - n)
                    mib = cur_size / 1024**2
                    print('\r[{}] {:.1f} MiB'.format(bar, mib), end='')
                f.write(chunk)
        print()
        print('Saved weights to', fn)


def weights_filename_with_download(weights):
    if not os.path.isdir(config.WEIGHTS_DIR):
        os.mkdir(config.WEIGHTS_DIR)

    if weights:
        weights_fn = weights
    else:
        # Try downloading it
        weights_fn = os.path.join(config.WEIGHTS_DIR,
                                  'autocolorize.caffemodel.h5')
        if not os.path.isfile(weights_fn):
            print('Downloading weights file...')
            download_file(config.WEIGHTS_URL, weights_fn)
    return weights_fn

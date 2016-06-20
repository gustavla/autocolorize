from __future__ import division, print_function, absolute_import
import autocolorize
import numpy as np
from skimage import color, exposure
from scipy.stats import norm
from . import image
import time
import sys
import os
import tempfile
from string import Template

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
RES_DIR = os.path.join(SCRIPT_DIR, 'res')
WEIGHTS_DIR = os.path.expanduser('~/.autocolorize')
WEIGHTS_URL = 'http://people.cs.uchicago.edu/~larsson/colorization/res/autocolorize.caffemodel.h5'
TOTAL_SIZE = 588203256


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
                    progress = cur_size / TOTAL_SIZE
                    n = int(np.ceil(progress * BARS))
                    bar = '#' * n + '.' * (BARS - n)
                    print('\r[{}] {:.1f} MiB'.format(bar, cur_size / 1024**2), end='')
                f.write(chunk)
        print()
        print('Saved weights to', fn)


def weights_filename_with_download(weights):
    if not os.path.isdir(WEIGHTS_DIR):
        os.mkdir(WEIGHTS_DIR)

    if weights:
        weights_fn = weights
    else:
        # Try downloading it
        weights_fn = os.path.join(WEIGHTS_DIR, 'autocolorize.caffemodel.h5')
        if not os.path.isfile(weights_fn):
            print('Downloading weights file...')
            download_file(WEIGHTS_URL, weights_fn)
    return weights_fn


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs='+', type=str, help='Input images')
    parser.add_argument('-o', '--output', type=str,
                        help='Output image or directory')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-w', '--weights', type=str, help='Weights file')
    parser.add_argument('-g', '--gpu', type=int),
    parser.add_argument('-d', '--device', choices=['cpu', 'gpu'],
                        default='gpu')
    parser.add_argument('-p', '--param', type=str)
    parser.add_argument('-s', '--size', type=int, default=512, help='Processing size of input')
    parser.add_argument('--install', action='store_true',
                        help='Download weights file')
    args = parser.parse_args()

    input_size = args.size

    # Not all input sizes will produce even sizes for all layers, so we have
    # to restrict the choices to multiples of 32
    if input_size % 32 != 0:
        print('Invalid processing size (--size {}): Must be a multiple of 32.'.format(input_size), file=sys.stderr)
        sys.exit(1)

    template_prototxt_fn = os.path.join(RES_DIR, 'autocolorize.prototxt.template')
    with open(template_prototxt_fn) as f:
        template_content = f.read()
    weights_fn = weights_filename_with_download(args.weights)
    if args.install:
        return

    min_side = input_size // 2
    max_side = input_size - 12

    import caffe

    if args.device == 'gpu':
        caffe.set_mode_gpu()
    elif args.device == 'cpu':
        caffe.set_mode_cpu()

    if args.gpu is not None:
        if args.device == 'cpu':
            raise ValueError('Cannot specify GPU when using CPU mode')
        caffe.set_device(args.gpu)

    with tempfile.NamedTemporaryFile(mode='w+', suffix='prototxt', delete=True) as f:
        # Make prototxt file
        content = Template(template_content).substitute(dict(INPUT_SIZE=input_size))
        f.write(content)
        f.seek(0)
        print(f.name)

        classifier = autocolorize.load_classifier(f.name,
                                                  weights=weights_fn)

    img_list = args.input
    output_fn0 = None
    if args.output is None:
        output_dir = 'colorization-output'
    elif os.path.splitext(os.path.basename(args.output))[1]:
        output_fn0 = args.output
        output_dir = None
        if len(img_list) > 1:
            raise ValueError("Cannot output to a single file if "
                             "multiple files are input")
    else:
        output_dir = args.output

    for img_i, img_fn in enumerate(img_list):
        name = os.path.splitext(os.path.basename(img_fn))[0]
        if output_fn0:
            output_fn = output_fn0
        else:
            if not os.path.isdir(output_dir):
                os.mkdir(output_dir)

            output_fn = os.path.join(output_dir, name + '.png')

        raw_img = image.load(img_fn)
        if raw_img.ndim == 3:
            grayscale = raw_img[..., :3].mean(-1)
        else:
            grayscale = raw_img

        orig_grayscale = grayscale.copy()

        rgb, info = calc_rgb(classifier, grayscale,
                             param=args.param, name=output_fn,
                             min_side=min_side, max_side=max_side,
                             return_info=True)

        # Correct the lightness
        rgb = autocolorize.match_lightness(rgb, grayscale)
        image.save(output_fn, rgb)

        print('Colorized: {} -> {}'.format(img_fn, output_fn))

        if args.verbose:
            print('Min side:', info['min_side'])
            print('Max side:', info['max_side'])
            print('Input shape', info['input_shape'])
            print('Scaled shape', info['scaled_shape'])
            print('Padded shape', info['padded_shape'])
            print('Output shape:', info['output_shape'])



def calc_rgb(classifier, grayscale, param=None, name=None,
             min_side=256, max_side=500, return_info=False):
    img_h, img_c, info = autocolorize.extract(classifier,
                                              grayscale,
                                              ['prediction_h', 'prediction_c'],
                                              min_side=min_side,
                                              max_side=max_side)


    bins = img_h.shape[-1]
    hist_h = img_h
    hist_c = img_c

    spaced = (np.arange(bins) + 0.5) / bins

    if param is None:
        c_method = 'median'
        h_method = 'expectation-cf'
    else:
        vv = param.split(':')
        c_method = vv[0]
        h_method = vv[1]

    factor = 1.0

    # Hue
    if h_method == 'mode':
        hsv_h = (hist_h.argmax(-1) + 0.5) / bins
    elif h_method == 'expectation':
        tau = 2 * np.pi
        a = hist_h * np.exp(1j * tau * spaced)
        hsv_h = (np.angle(a.sum(-1)) / tau) % 1.0
    elif h_method == 'expectation-cf':  # with chromatic fading
        tau = 2 * np.pi
        a = hist_h * np.exp(1j * tau * spaced)
        cc = abs(a.mean(-1))
        factor = cc.clip(0, 0.03) / 0.03
        hsv_h = (np.angle(a.sum(-1)) / tau) % 1.0
    elif h_method == 'pixelwise':
        cum_h = hist_h.cumsum(-1)
        z_h = ((np.random.uniform(size=cum_h.shape[:2]+(1,)) > cum_h).sum(-1) + 0.5) / bins
        hsv_h = z_h
    elif h_method == 'median':
        cum_h = hist_h.cumsum(-1)
        z_h = ((0.5 > cum_h).sum(-1) + 0.5) / bins
        hsv_h = z_h
    elif h_method == 'once':
        cum_h = hist_h.cumsum(-1)
        z_h = ((np.random.uniform() > cum_h).sum(-1) + 0.5) / bins
        hsv_h = z_h

    # Chroma
    if c_method == 'mode': # mode
        hsv_c = hist_c.argmax(-1) / bins
    elif c_method == 'expectation': # expectation
        hsv_c = (hist_c * spaced).sum(-1)
    elif c_method == 'pixelwise':
        cum_c = hist_c.cumsum(-1)
        z_c = ((np.random.uniform(size=cum_c.shape) > cum_c).sum(-1) + 0.5) / bins
        hsv_c = z_c
    elif c_method == 'once':
        cum_c = hist_c.cumsum(-1)
        z_c = ((np.random.uniform() > cum_c).sum(-1) + 0.5) / bins
        hsv_c = z_c
    elif c_method == 'median':
        cum_c = hist_c.cumsum(-1)
        z_c = ((0.5 > cum_c).sum(-1) + 0.5) / bins
        hsv_c = z_c
    elif c_method == 'q75':
        cum_c = hist_c.cumsum(-1)
        z_c = ((0.75 > cum_c).sum(-1) + 0.5) / bins
        hsv_c = z_c
    else:
        raise ValueError('Unknown chroma method')

    hsv_c *= factor

    hsv_v = grayscale + hsv_c / 2
    hsv_s = 2 * hsv_c / (2 * grayscale + hsv_c)

    hsv = np.concatenate([hsv_h[..., np.newaxis], hsv_s[..., np.newaxis], hsv_v[..., np.newaxis]], axis=-1)
    rgb = color.hsv2rgb(hsv.clip(0, 1)).clip(0, 1)

    if return_info:
        return rgb, info
    else:
        return rgb


if __name__ == '__main__':
    main()

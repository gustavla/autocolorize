from __future__ import division, print_function, absolute_import
import autocolorize
import numpy as np
from skimage import color, exposure
from scipy.stats import norm
from . import image
import time
import os

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
RES_DIR = os.path.join(SCRIPT_DIR, 'res')
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


def main():
    import argparse
    import caffe
    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs='+', type=str, help='Input images')
    parser.add_argument('-o', '--output', type=str, help='Output image or directory')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-w', '--weights', type=str, help='Weights file')
    parser.add_argument('-g', '--gpu', type=int)
    parser.add_argument('-p', '--param', type=str)
    args = parser.parse_args()

    start = time.time()

    if args.gpu is not None:
        caffe.set_device(args.gpu)
    caffe.set_mode_gpu()

    prototxt_fn = os.path.join(RES_DIR, 'autocolorize.prototxt')
    if args.weights:
        weights_fn = args.weights
    else:
        # Try downloading it
        weights_fn = os.path.join(RES_DIR, 'autocolorize.caffemodel.h5')
        if not os.path.isfile(weights_fn):
            print('Downloading weights file...')
            download_file(WEIGHTS_URL, weights_fn)

    classifier = autocolorize.load_classifier(prototxt_fn,
                                              weights=weights_fn)

    img_list = args.input
    output_fn0 = None
    if args.output is None:
        output_dir = 'colorization-output'
    elif os.path.splitext(os.path.basename(args.output))[1]:
        output_fn0 = args.output
        output_dir = None
        if len(img_list) > 1:
            raise ValueError("Cannot output to a single file if multiple files are input")
    else:
        output_dir = args.output

    for img_i, img_fn in enumerate(img_list):
        name = os.path.splitext(os.path.basename(img_fn))[0]
        if output_fn0:
            output_fn = output_fn0
        else:
            output_fn = os.path.join(output_dir, name + '.png')

        raw_img = image.load(img_fn)
        if raw_img.ndim == 3:
            grayscale = raw_img[..., :3].mean(-1)
        else:
            grayscale = raw_img

        orig_grayscale = grayscale.copy()

        rgb = calc_rgb(classifier, grayscale,
                       param=args.param, name=output_fn)

        # Correct the lightness
        rgb = autocolorize.match_lightness(rgb, grayscale)
        image.save(output_fn, rgb)

        print('Colorized: {} -> {}'.format(img_fn, output_fn))


def calc_rgb(classifier, grayscale, param=None, name=None):
    img_h, img_c = autocolorize.extract(classifier,
                                        grayscale,
                                        'prediction_h',
                                        'prediction_c')


    bins = img_h.shape[-1]
    hist_h = img_h
    hist_c = img_c

    spaced = (np.arange(bins) + 0.5) / bins

    if param is None:
        c_method = 'median'
        h_method = 'expectation-cf'
    else:
        vv = param.split('-')
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

    return rgb


if __name__ == '__main__':
    main()

import os
import time
import re
import glob
import caffe
import sys
import numpy as np
import itertools as itr
from skimage import color
import autocolorize
import argparse
from autocolorize import image


def checker_main(inner):
    parser = argparse.ArgumentParser()
    parser.add_argument('out_dir', type=str, help='Output directory')
    parser.add_argument('-d', '--dataset', default='sun')
    parser.add_argument('-o', '--offset', type=int, default=0)
    parser.add_argument('-n', '--count', type=int, default=10000)
    parser.add_argument('-gto', '--ground-truth-output')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-s', '--seed', type=int)
    parser.add_argument('-g', '--gpu', type=int)
    parser.add_argument('-p', '--param', type=str)
    args = parser.parse_args()

    start = time.time()

    if args.gpu is not None:
        caffe.set_device(args.gpu)
    caffe.set_mode_gpu()

    classifier = autocolorize.load_classifier('bare.prototxt')
    img_list = autocolorize.load_image_list(args.dataset, offset=args.offset, count=args.count, seed=args.seed)

    for img_i, img_fn in enumerate(img_list):
        if time.time() - start > 3.75 * 3600:
            sys.exit(0)

        name = os.path.splitext(os.path.basename(img_fn))[0]
        output_fn = os.path.join(args.out_dir, name + '.png')

        if os.path.exists(output_fn):
            continue

        raw_img = image.load(img_fn)#[..., :3][..., ::-1]
        if raw_img.ndim == 3:
            raw_img = raw_img[..., :3][..., ::-1]
        else:
            raw_img = np.tile(raw_img[..., np.newaxis], 3)

        orig_img = raw_img.copy()
        print('Original image size:', raw_img.shape)

        rgb = inner(classifier, raw_img,
                    param=args.param, name=output_fn)

        # Correct the lightness
        #rgb = autocolorize.match_lightness(rgb, orig_img)
        rgb = rgb.clip(0, 1)
        image.save(output_fn, rgb)

        if args.ground_truth_output:
            orig_rgb = orig_img[..., ::-1]
            image.save(os.path.join(args.ground_truth_output, name + '.png'), orig_rgb)

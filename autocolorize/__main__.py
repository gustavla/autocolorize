from __future__ import division, print_function, absolute_import
import autocolorize
from . import image, download
import sys
import os


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs='*', type=str, help='Input images')
    parser.add_argument('-o', '--output', type=str,
                        help='Output image or directory')
    parser.add_argument('-V', '--version', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-w', '--weights', type=str, help='Weights file')
    parser.add_argument('-g', '--gpu', type=int),
    parser.add_argument('-d', '--device', choices=['cpu', 'gpu'],
                        default='gpu')
    parser.add_argument('-p', '--param', type=str)
    parser.add_argument('-s', '--size', type=int, default=512,
                        help='Processing size of input')
    parser.add_argument('--install', action='store_true',
                        help='Download weights file')
    args = parser.parse_args()

    if args.version:
        print('autocolorize', autocolorize.__version__)
        return

    input_size = args.size

    # Not all input sizes will produce even sizes for all layers, so we have
    # to restrict the choices to multiples of 32
    if input_size % 32 != 0:
        s = 'Invalid processing size (--size {}): Must be a multiple of 32.'
        print(s.format(input_size), file=sys.stderr)
        sys.exit(1)

    if args.install:
        download.weights_filename_with_download(args.weights)
        return

    import caffe

    if args.device == 'gpu':
        caffe.set_mode_gpu()
    elif args.device == 'cpu':
        caffe.set_mode_cpu()

    if args.gpu is not None:
        if args.device == 'cpu':
            raise ValueError('Cannot specify GPU when using CPU mode')
        caffe.set_device(args.gpu)

    classifier = autocolorize.load_default_classifier(input_size=input_size,
                                                      weights=args.weights)

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
        rgb, info = autocolorize.colorize(raw_img, classifier=classifier,
                                          param=args.param, return_info=True)

        image.save(output_fn, rgb)

        print('Colorized: {} -> {}'.format(img_fn, output_fn))

        if args.verbose:
            print('Min side:', info['min_side'])
            print('Max side:', info['max_side'])
            print('Input shape', info['input_shape'])
            print('Scaled shape', info['scaled_shape'])
            print('Padded shape', info['padded_shape'])
            print('Output shape:', info['output_shape'])


if __name__ == '__main__':
    main()

from __future__ import division, print_function, absolute_import
import os
import glob
import itertools as itr
import numpy as np


def load_image_list(dataset, offset=0, count=None, seed=None):
    if dataset == 'legacy':
        root_dir = "/share/data/vision-greg/larsson/data/legacy"
        img_list = glob.glob(os.path.join(root_dir, '*'))
        #name_list = [os.path.splitext(os.path.basename(fn))[0] for fn in sorted(glob.glob(os.path.join(root_dir, '*')))]
        #img_list = [os.path.join(root_dir, '{}.png'.format(fn)) for fn in name_list]
    elif dataset == 'charpiat':
        root_dir = "/share/data/vision-greg/larsson/data/charpiat"
        img_list = glob.glob(os.path.join(root_dir, '*'))
        #name_list = [os.path.splitext(os.path.basename(fn))[0] for fn in sorted(glob.glob(os.path.join(root_dir, '*')))]
        #img_list = [os.path.join(root_dir, '{}.png'.format(fn)) for fn in name_list]
    elif dataset == 'deep-sun':
        root_dir = "/share/data/vision-greg/larsson/data/SUN397"
        with open(os.path.join(root_dir, 'list.txt')) as f:
            img_list = [os.path.join(root_dir, fn.strip()) for fn in f.readlines()]
    elif dataset == 'sun':
        root_dir = "/share/data/vision-greg/larsson/data/uiuc-color-data/ground-truth"

        name_list = [os.path.splitext(os.path.basename(fn))[0] for fn in sorted(glob.glob(os.path.join(root_dir, '*.png')))]
        img_list = [os.path.join(root_dir, '{}.png'.format(fn)) for fn in name_list]

    elif dataset == 'pascal':
        #root_dir = "/share/data/vision-greg/Pascal/VOCdevkit/VOC2012"
        #name_list = [os.path.splitext(os.path.basename(fn))[0] for fn in sorted(glob.glob(os.path.join(ucm_dir, '*.mat')))]
        #img_list = ['/JPEGImages/{}.jpg'.format(fn) for fn in name_list]
        raise NotImplemented('Fix if needed')
    elif dataset == 'imagenet-val':
        root_dir = "/share/data/vision-greg/ImageNet/clsloc/256/images/val"
        with open("/share/data/vision-greg/larsson/ImageNet/imagenet_val.txt") as f:
            img_list = [os.path.join(root_dir, x.split()[0][1:]) for x in f.readlines()]
    elif dataset == 'imagenet':
        root_dir = "/share/data/vision-greg/ImageNet/clsloc/256/images/train"
        with open("/share/data/vision-greg/larsson/ImageNet/imagenet_train.txt") as f:
            img_list = [os.path.join(root_dir, x.split()[0][1:]) for x in f.readlines()]
    elif dataset == 'imagenet-cval1k':
        root_dir = "/share/data/vision-greg/ImageNet/clsloc/256/images/val"
        with open("/share/data/vision-greg/larsson/ImageNet/colorization/imagenet_cval1k.txt") as f:
            img_list = [os.path.join(root_dir, x.split()[0][1:]) for x in f.readlines()]
    elif dataset == 'imagenet-example':
        root_dir = "/share/data/vision-greg/ImageNet/clsloc/256/images/val"
        img_list = [os.path.join(root_dir, 'n12620546/ILSVRC2012_val_00039107.JPEG')]
    elif dataset == 'imagenet-ctest1k':
        root_dir = "/share/data/vision-greg/ImageNet/clsloc/256/images/val"
        with open("/share/data/vision-greg/larsson/ImageNet/colorization/imagenet_ctest1k.txt") as f:
            img_list = [os.path.join(root_dir, x.split()[0][1:]) for x in f.readlines()]
    elif dataset == 'imagenet-ctest10k':
        root_dir = "/share/data/vision-greg/ImageNet/clsloc/256/images/val"
        with open("/share/data/vision-greg/larsson/ImageNet/colorization/imagenet_ctest10k.txt") as f:
            img_list = [os.path.join(root_dir, x.split()[0][1:]) for x in f.readlines()]
    else:
        raise ValueError("Unknown dataset")

    if seed is not None:
        rs = np.random.RandomState(seed)
        rs.shuffle(img_list)

    if count is None:
        return img_list[offset:]
    else:
        return img_list[offset:offset + count]

from __future__ import division, print_function, absolute_import
import glob
import re
import caffe
import os
import numpy as np
from . import image
from scipy.ndimage import zoom
from skimage.transform import rescale


# Could be moved to a more configurable place
MAX_SIDE = 500
MIN_SIDE = 256
HOLE = 2


def load_classifier(bare_fn, snap_dir='snapshots', iteration=None, weights=None):
    """
    Loads classifier with latest snapshot
    """
    fn = None
    if weights is not None:
        fn = weights
    else:
        if iteration is None:
            snap_fns = glob.glob(os.path.join(snap_dir, '*.caffemodel.h5'))
            snapshots = sorted([(int(re.sub('[^0-9]', '', os.path.splitext(fn)[0])), fn) for fn in snap_fns])
        else:
            snapshots = [(iteration, os.path.join(snap_dir, 'snapshot_iter_{}.caffemodel.h5'.format(iteration)))]
        fn = snapshots[-1][1]

    if fn:
        print('Loading', fn)
        classifier = caffe.Classifier(bare_fn, fn)
    else:
        raise ValueError('No snapshot')
    return classifier


def resize_by_factor(img, factor):
    if factor != 1:
        return rescale(img, factor, mode='constant', cval=0)
    else:
        return img


def extract_sparse(classifier, grayscale, *chs):
    """
    Extract using sparse model (slower).

    classifier: Caffe Classifier object

    """
    # Set scale so that the longest is MAX_SIDE
    min_side = np.min(img.shape[:2])
    max_side = np.max(img.shape[:2])
    scale = min(MIN_SIDE / min_side, 1)
    if max_side * scale >= MAX_SIDE:
        scale = MAX_SIDE / max_side

    samples = classifier.blobs['centroids'].data.shape[1]
    size = classifier.blobs['data'].data.shape[2]
    centroids = np.zeros((1, samples, 2), dtype=np.float32)
    full_shape = grayscale.shape[:2]
    grayscale = resize_by_factor(grayscale, scale)
    raw_shape = grayscale.shape[:2]

    st0 = (size - raw_shape[0])//2
    en0 = st0 + raw_shape[0]
    st1 = (size - raw_shape[1])//2
    en1 = st1 + raw_shape[1]

    grayscale = image.center_crop_reflect(grayscale, (size, size))

    #bgr = raw_img.transpose(2, 0, 1)[np.newaxis]
    #data = bgr.mean(1, keepdims=True)
    data = grayscale[np.newaxis, np.newaxis]

    print('data', data.shape)

    shape = raw_img.shape[:2]

    scaled_size = size // HOLE

    scaled_shape = tuple([shi // HOLE for shi in raw_shape[:2]])

    #scaled_chs = []
    #scaled_a = np.zeros(scaled_shape + (1,))
    #scaled_b = np.zeros(scaled_shape + (1,))
    scaleds = [None] * len(chs)

    img_ii, img_jj = np.meshgrid(range(scaled_shape[0]), range(scaled_shape[1]))

    ii = HOLE//2 + st0 + img_ii * HOLE
    jj = HOLE//2 + st1 + img_jj * HOLE

    indices = np.arange(0, ii.size, samples)
    for chunk in indices:
        chunk_slice = np.s_[chunk:chunk + samples]
        ii_ = ii.ravel()[chunk_slice]
        jj_ = jj.ravel()[chunk_slice]
        centroids[0, :len(ii_), 0] = ii_
        centroids[0, :len(jj_), 1] = jj_

        ret = classifier.forward(data=grayscale, centroids=centroids)

        ii_ = img_ii.ravel()[chunk_slice]
        jj_ = img_jj.ravel()[chunk_slice]

        for i, ch in enumerate(chs):
            if scaleds[i] is None:
                scaleds[i] = np.zeros(scaled_shape + (ret[chs[i]].shape[-1],))
            scaleds[i][ii_, jj_] = ret[ch][:len(ii_)]

    scaled_combined = np.concatenate(scaleds, axis=-1)

    full_combined = resize_by_factor(scaled_combined, HOLE / scale)
    full_combined = image.center_crop(full_combined, img.shape[:2])
    assert full_combined.shape[:2] == img.shape[:2]
    combined = full_combined

    res = []
    cur = 0
    for i, ch in enumerate(chs):
        C = scaleds[i].shape[-1]
        res.append(combined[..., cur:cur + C])
        cur += C

    return tuple(res)


def extract(classifier, grayscale, *chs):
    """
    Extract using dense model (faster).

    classifier: Caffe Classifier object
    grayscale: Input image
    """
    # Set scale so that the longest is MAX_SIDE
    min_side = np.min(grayscale.shape[:2])
    max_side = np.max(grayscale.shape[:2])
    scale = min(MIN_SIDE / min_side, 1)
    if max_side * scale >= MAX_SIDE:
        scale = MAX_SIDE / max_side

    HOLE = 4

    #samples = classifier.blobs['centroids'].data.shape[1]
    size = classifier.blobs['data'].data.shape[2]
    #centroids = np.zeros((1, samples, 2), dtype=np.float32)
    full_shape = grayscale.shape[:2]
    raw_grayscale = grayscale.copy()
    if scale != 1:
        grayscale = resize_by_factor(grayscale, scale)
    raw_shape = grayscale.shape[:2]

    st0 = (size - raw_shape[0])//2
    en0 = st0 + raw_shape[0]
    st1 = (size - raw_shape[1])//2
    en1 = st1 + raw_shape[1]

    #raw_img = image.center_crop_reflect(raw_img, (size, size))
    grayscale = image.center_crop(grayscale, (size, size))

    #bgr = raw_img.transpose(2, 0, 1)[np.newaxis]

    data = grayscale[np.newaxis, np.newaxis]

    #shape = grayscale.shape[:2]

    scaled_size = size // HOLE

    scaled_shape = tuple([shi // HOLE for shi in raw_shape[:2]])

    ret = classifier.forward(data=data)

    data = {key: classifier.blobs[key].data for key in classifier.blobs}

    scaleds = [ret['prediction_h_full'][0].transpose(1, 2, 0), ret['prediction_c_full'][0].transpose(1, 2,0)]

    scaled_combined = np.concatenate(scaleds, axis=-1)

    #full_combined = resize_by_factor(scaled_combined, HOLE / scale)
    full_combined = resize_by_factor(scaled_combined, 1 / scale)
    full_combined = image.center_crop(full_combined, full_shape)
    assert full_combined.shape[:2] == full_shape
    combined = full_combined

    res = []
    cur = 0
    for i, ch in enumerate(chs):
        C = scaleds[i].shape[-1]
        res.append(combined[..., cur:cur + C])
        cur += C

    return tuple(res)

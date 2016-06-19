from __future__ import division, print_function, absolute_import
import numpy as np
from skimage import util


def load(path, dtype=np.float64):
    """
    Loads an image from file.

    Parameters
    ----------
    path : str
        Path to image file.
    dtype : np.dtype
        Defaults to ``np.float64``, which means the image will be returned as a
        float with values between 0 and 1. If ``np.uint8`` is specified, the
        values will be between 0 and 255 and no conversion cost will be
        incurred.
    """
    import skimage.io
    im = skimage.io.imread(path)
    if dtype == np.uint8:
        return im
    elif dtype in {np.float16, np.float32, np.float64}:
        return im.astype(dtype) / 255
    else:
        raise ValueError('Unsupported dtype')


def save(path, im):
    """
    Saves an image to file.

    If the image is type float, it will assume to have values in [0, 1].

    Parameters
    ----------
    path : str
        Path to which the image will be saved.
    im : ndarray (image)
        Image.
    """
    from PIL import Image
    if im.dtype == np.uint8:
        pil_im = Image.fromarray(im)
    else:
        pil_im = Image.fromarray((im*255).astype(np.uint8))
    pil_im.save(path)


def center_crop(img, size, value=0.0):
    """Center crop with padding (using `value`) if necessary"""
    new_img = np.full(size + img.shape[2:], value, dtype=img.dtype)

    dest = [0, 0]
    source = [0, 0]
    ss = [0, 0]
    for i in range(2):
        if img.shape[i] < size[i]:
            diff = size[i] - img.shape[i]
            dest[i] = diff // 2
            source[i] = 0
            ss[i] = img.shape[i]
        else:
            diff = img.shape[i] - size[i]
            source[i] = diff // 2
            ss[i] = size[i]

    new_img[dest[0]:dest[0]+ss[0], dest[1]:dest[1]+ss[1]] = \
            img[source[0]:source[0]+ss[0], source[1]:source[1]+ss[1]]

    return new_img

def center_crop_reflect(img, size):
    """Center crop with mirror padding if necessary"""
    a0 = max(0, size[0] - img.shape[0])
    a1 = max(0, size[1] - img.shape[1])

    v = ((a0//2, a0-a0//2), (a1//2, a1-a1//2))
    if img.ndim == 3:
        v = v + ((0, 0),)

    pimg = util.pad(img, v, mode='reflect')
    return center_crop(pimg, size)

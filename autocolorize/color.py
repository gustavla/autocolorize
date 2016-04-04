from __future__ import division, print_function, absolute_import
import numpy as np


def match_lightness(img, ref_img):
    if ref_img.ndim == 3:
        ref_grayscale = ref_img.mean(-1)
    else:
        ref_grayscale = ref_img

    new_img = img - img.mean(-1, keepdims=True) + ref_grayscale[..., np.newaxis]
    return new_img.clip(0, 1)

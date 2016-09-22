from __future__ import division, print_function, absolute_import
import tensorflow as tf
from tensorflow.python.framework import ops
import os

_module = tf.load_op_library(os.path.expandvars('$TF_USER_OPS/sparse_extractor.so'))
sparse_extractor = _module.sparse_extractor


@ops.RegisterShape('SparseExtractor')
def _SparseExtractorShape(op):
    d = 0
    if op.get_attr('data_format') == b'NHWC':
        d = op.inputs[1].get_shape()[3]
    elif op.get_attr('data_format') == b'NCHW':
        d = op.inputs[1].get_shape()[1]

    # If batch size shape is specified, we'll specify it for the output
    B, L = op.inputs[0].get_shape().as_list()[:2]
    if B is None:
        out_size = None
    else:
        out_size = B * L
    return [tf.TensorShape([out_size, d])]


@ops.RegisterGradient('SparseExtractor')
def _SparseExtractorGrad(op, grad):
    input_grad = _module.sparse_extractor_grad(
        op.inputs[0],
        op.inputs[1],
        op.outputs[0],
        grad,
        op.inputs[2],
        op.inputs[3],
        op.get_attr('data_format'))

    return [None, input_grad, None, None]

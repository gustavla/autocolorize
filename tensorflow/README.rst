Sparse hypercolumn training
===========================

Unlike our Caffe impementation, we chose not to build the final hypercolumn in
one layer, and instead extract sparse locations from one layer only. All layers
can then be composed into a sparse hypercolumn by concatenating each layer.

Place the ``user_ops`` files in ``tensorflow/core/user_ops``. Add the follow to your
``tensorflow/core/user_ops/BUILD`` file (create it, if you do not have one)::

    load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

    tf_custom_op_library(
        name = "sparse_extractor.so",
        srcs = ["sparse_extractor.cc"],
        gpu_srcs = ["sparse_extractor_gpu.cu.cc"],
    )

Now, follow instructions `here
<https://www.tensorflow.org/versions/r0.10/how_tos/adding_an_op/index.html>`__
to build into a shared library.

Place this library (``sparse_extractor.so`` file) in a directory and point the
environment variable ``$TF_USER_OPS`` to it.

Usage in Python
---------------
Use it through Python as::

    from autocolorize.tensorflow.sparse_extractor import sparse_extractor

    # assume batch_size, locations, edge_buffer and input_size are all defined integers
    # assume conv1, conv2, conv3 are activations
    centroids = tf.cast(tf.random_uniform(shape=[batch_size, locations, 2],
                                          minval=edge_buffer,
                                          maxval=input_size - edge_buffer,
                                          name='centroids',
                                          dtype=tf.int32), tf.float32)

    conv1_sparse = sparse_extractor(centroids, conv1, 1.0, [0.0, 0.0])
    conv2_sparse = sparse_extractor(centroids, conv2, 2.0, [0.0, 0.0])
    conv3_sparse = sparse_extractor(centroids, conv3, 4.0, [0.0, 0.0])

    hypercolumn = tf.concat(1, [conv1_sparse, conv2_sparse, conv3_sparse], name='hypercolumn')


The ``[0.0, 0.0]`` are the offsets. I usually don't use any at all in
Tensorflow, since that pairs well with how ``tf.image.resize_bilinear`` works.
That way, you can do dense outputs during inference time, which might look
like this::

    # assume conv1 is the target scale here

    conv2_upscaled = tf.image.resize_bilinear(conv2, [input_size, input_size])
    conv3_upscaled = tf.image.resize_bilinear(conv3, [input_size, input_size])

    dense_hypercolumn = tf.concat(3, [conv1, conv2_upscaled, conv3_upscaled], name='hypercolumn')

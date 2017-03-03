Train colorizer
===============

To train the colorizer from scratch, follow these steps:

* Copy all files from ``caffe/src`` and ``caffe/include`` into your Caffe folder.

* Add ``caffe/add_to_caffe.proto`` to the end of your ``src/caffe/proto/caffe.proto`` file. Place the commented out lines inside the ``message LayerParameter { ... }`` block.

* Add ``LIBRARIES += matio maskApi gason boost_filesystem`` to your
  ``Makefile.config``. If you do not import our extended data loading layer,
  you do not need to do this step. This layer is not essential for
  colorization, but does offer some added features.

* Re-compile Caffe.

* Edit the ``/path/to/train.txt`` in ``train_<model>.prototxt``. This file should
  contain a list of paths to image files. Feel free to change the training time
  in the solver file to fit your needs.

* Run ``caffe train -solver=solver_<model>.prototxt``.

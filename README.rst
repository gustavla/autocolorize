.. image:: https://img.shields.io/pypi/v/autocolorize.svg
    :target: https://pypi.python.org/pypi/autocolorize

autocolorize
============

Automatically colorize images using Machine Learning.

* `Project page <http://people.cs.uchicago.edu/~larsson/colorization/>`__
* ECCV 2016 paper (`arXiv <http://arxiv.org/abs/1603.06668>`__)

Installation
------------
Make sure that you have Caffe (with Python bindings). Then run::

    pip install autocolorize

Run::

    autocolorize grayscale.png -o colorized.png

API
---
You can also colorize from Python (assuming ``grayscale`` is the image that you want to colorize)::

    import autocolorize
    classifier = autocolorize.load_default_classifier()
    rgb = autocolorize.colorize(grayscale, classifier=classifier)

Sparse training
---------------
We provide custom layers for doing sparse hypercolumn training in both Caffe
(see ``caffe/``) and Tensorflow (see ``tensorflow/``). This can be used for other
image-to-image tasks, such as semantic segmentation or edge prediction.

Look inside the ``train`` folder if you want to train from scratch.

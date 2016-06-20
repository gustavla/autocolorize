.. image:: https://img.shields.io/pypi/v/autocolorize.svg
    :target: https://pypi.python.org/pypi/autocolorize

autocolorize
============

Automatically colorize images using Machine Learning.

* `Project page <http://people.cs.uchicago.edu/~larsson/colorization/>`__
* `arXiv paper <http://arxiv.org/abs/1603.06668>`__

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

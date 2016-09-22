Sparse hypercolumn training
===========================

Copy all files into your Caffe folder. Then, add the following to your ``src/caffe/proto/caffe.proto`` file in ``LayerParameter``::

    optional SparseHypercolumnExtractorParameter sparse_hypercolumn_extractor_param = 1234;

Set ``1234`` to whatever you want that is not in conflict with another layer's parameters. Also add the following to the bottom ``caffe.proto``::

    message SparseHypercolumnExtractorParameter {
      repeated float scale = 1;
      repeated float offset_height = 2;
      repeated float offset_width = 3;

      // Sometimes the layers may be scaled differently. This allows adding a multiplier to the activations
      repeated float activation_mult = 4;
    }

Re-compile and you should now have access to the ``SparseHypercolumnExtractor`` layer.

Usage
-----
Here is an example of how to extract a sparse hypercolumn::

    layer {
      type: "SparseHypercolumnExtractor"
      bottom: "centroids"   # First input defines locations, (B, P, 2)
      bottom: "conv1"       # The rest are layers
      bottom: "conv2"
      bottom: "conv3"
      top: "columns"        # Output has shape (B*P, D)
      sparse{
        scale: 1            # scale for input "conv1"
        scale: 2            # scale for input "conv2"
        scale: 4            # scale for input "conv3"
        offset_height: 0    # vertical offset for input "conv1"
        offset_height: 0.5  # vertical offset for input "conv2"
        offset_height: 1.5  # vertical offset for input "conv3"
        offset_width: 0     # horizontal offset for input "conv1"
        offset_width: 0.5   # horizontal offset for input "conv2"
        offset_width: 1.5   # horizontal offset for input "conv3"
      }
    }

Note that ``B`` is the batch-size, ``P`` is the number of sparse locations per
sample, and ``D`` is the total number of channels in all input layers
(``conv1`` - ``conv3``). You can generate sparse samples using a ``DummyData``
layer::

    layer {
      name: "sampler"
      type: "DummyData"
      top: "centroids"
      dummy_data_param {
        data_filler {
          type: "uniform"
          min: 32           # If you want to avoid 32 pixels around the edge
          max: 224          # and your input is 256x256
        }
        shape {
          dim: 8            # B
          dim: 128          # P
          dim: 2            # x/y
        }
      }
    }

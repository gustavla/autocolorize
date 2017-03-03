#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/sparse_histogram_extractor_layer.hpp"

namespace caffe {

template <typename Dtype>
void SparseHistogramExtractorLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Forward_cpu(bottom, top);
}

// Note: The general templated version will fall-back to the CPU version.
// This will happen only for 64-bit floats, since the 32-bit version is defined below
template <typename Dtype>
void SparseHistogramExtractorLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Backward_cpu(top, propagate_down, bottom);
}


INSTANTIATE_LAYER_GPU_FUNCS(SparseHistogramExtractorLayer);

}  // namespace caffe

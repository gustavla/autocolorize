#include <algorithm>
#include <vector>

#include "caffe/layers/uniform_to_histogram_layer.hpp"

namespace caffe {

template <typename Dtype>
void UniformToHistogramLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Forward_cpu(bottom, top);
}

template <typename Dtype>
void UniformToHistogramLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    Backward_cpu(top, propagate_down, bottom);
}

INSTANTIATE_LAYER_GPU_FUNCS(UniformToHistogramLayer);

}  // namespace caffe


#include <algorithm>
#include <vector>

#include "caffe/layers/strip_decimals_layer.hpp"

namespace caffe {

template <typename Dtype>
void StripDecimalsLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    top_data[i] = (Dtype)((int)bottom_data[i]);
  }
}

template <typename Dtype>
void StripDecimalsLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
}


#ifdef CPU_ONLY
STUB_GPU(StripDecimalsLayer);
#endif

INSTANTIATE_CLASS(StripDecimalsLayer);
REGISTER_LAYER_CLASS(StripDecimals);

}  // namespace caffe

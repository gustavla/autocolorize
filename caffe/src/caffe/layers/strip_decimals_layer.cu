#include <algorithm>
#include <vector>

#include "caffe/layers/strip_decimals_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void StripDecimalsForward(const int n, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = (Dtype)((int)in[index]);
  }
}

template <typename Dtype>
void StripDecimalsLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  StripDecimalsForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void StripDecimalsLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
}


INSTANTIATE_LAYER_GPU_FUNCS(StripDecimalsLayer);


}  // namespace caffe

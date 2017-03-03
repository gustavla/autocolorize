#include <algorithm>
#include <vector>

#include "caffe/layers/bgr_to_munsell_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void BgrToMunsellForward(const int num, const int size, const Dtype* bottom_data, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, num * size) {
    int n = index / size;
    int i = index % size;

    Dtype r, g, b, h, s, v, c, delta, out0;
    b = bottom_data[n * 3 * size + 0 * size + i];
    g = bottom_data[n * 3 * size + 1 * size + i];
    r = bottom_data[n * 3 * size + 2 * size + i];

    v = max(r, max(g, b));

    delta = v - min(r, min(g, b));

    if (v == 0) {
      s = 0.0;
    } else {
      s = delta / v;
    }

    if (r == v) {
      out0 = (g - b) / delta;
    } else if (g == v) {
      out0 = 2.0 + (b - r) / delta;
    } else {
      out0 = 4.0 + (r - g) / delta;
    }

    h = fmod((out0 / 6.0) + 10.0, 1.0);
    if (delta == 0) {
      h = 0.0;
    }

    c = s * v;

    top_data[n * 3 * size + 0 * size + i] = h;
    top_data[n * 3 * size + 1 * size + i] = c;
    top_data[n * 3 * size + 2 * size + i] = v;
  }
}

template <typename Dtype>
void BgrToMunsellLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();

  const int num = bottom[0]->num();
  const int size = bottom[0]->height() * bottom[0]->width();

  const int count = num * size;
  // NOLINT_NEXT_LINE(whitespace/operators)
  BgrToMunsellForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      num, size, bottom_data, top_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void BgrToMunsellLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    Backward_cpu(top, propagate_down, bottom);
}

INSTANTIATE_LAYER_GPU_FUNCS(BgrToMunsellLayer);


}  // namespace caffe


#include <algorithm>
#include <vector>

#include "caffe/layers/bgr_to_hsv_layer.hpp"

namespace caffe {

template <typename Dtype>
void BgrToHsvLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void BgrToHsvLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int num = bottom[0]->num();
  const int size = bottom[0]->height() * bottom[0]->width();

  for (int n = 0; n < num; ++n) {
    for (int i = 0; i < size; ++i) {
      Dtype r, g, b, h, s, v, delta, out0;
      b = bottom_data[n * 3 * size + 0 * size + i];
      g = bottom_data[n * 3 * size + 1 * size + i];
      r = bottom_data[n * 3 * size + 2 * size + i];

      v = std::max(r, std::max(g, b));

      delta = v - std::min(r, std::min(g, b));

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

      h = std::fmod((out0 / 6.0) + 10.0, 1.0);
      if (delta == 0) {
        h = 0.0;
      }

      top_data[n * 3 * size + 0 * size + i] = h;
      top_data[n * 3 * size + 1 * size + i] = s;
      top_data[n * 3 * size + 2 * size + i] = v;
    }
  }
}

template <typename Dtype>
void BgrToHsvLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  if (propagate_down[0]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate a gradient.";
  }
}

#ifdef CPU_ONLY
STUB_GPU(BgrToHsvLayer);
#endif

INSTANTIATE_CLASS(BgrToHsvLayer);
REGISTER_LAYER_CLASS(BgrToHsv);

}  // namespace caffe

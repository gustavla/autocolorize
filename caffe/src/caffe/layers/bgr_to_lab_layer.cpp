#include <algorithm>
#include <vector>

#include "caffe/layers/bgr_to_lab_layer.hpp"

namespace caffe {

template <typename Dtype>
void BgrToLabLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void BgrToLabLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int num = bottom[0]->num();
  const int size = bottom[0]->height() * bottom[0]->width();

  for (int n = 0; n < num; ++n) {
    for (int s = 0; s < size; ++s) {
      Dtype r, g, b, r0, g0;
      b = bottom_data[n * 3 * size + 0 * size + s];
      g = bottom_data[n * 3 * size + 1 * size + s];
      r = bottom_data[n * 3 * size + 2 * size + s];

      r = (r > 0.04045) * std::pow((r + 0.055) / 1.055, 2.4) + (r <= 0.04045) * (r / 12.92);
      g = (g > 0.04045) * std::pow((g + 0.055) / 1.055, 2.4) + (g <= 0.04045) * (g / 12.92);
      b = (b > 0.04045) * std::pow((b + 0.055) / 1.055, 2.4) + (b <= 0.04045) * (b / 12.92);

      r0 = r;
      g0 = g;
      r = (0.412453 * r0 + 0.357580 * g0 + 0.180423 * b) / 0.95047;
      g = (0.212671 * r0 + 0.715160 * g0 + 0.072169 * b);
      b = (0.019334 * r0 + 0.119193 * g0 + 0.950227 * b) / 1.08883;

      r = (r > 0.008856) * std::pow(r, 1/3.) + (r <= 0.008856) * (7.787 * r + 16/116.);
      g = (g > 0.008856) * std::pow(g, 1/3.) + (g <= 0.008856) * (7.787 * g + 16/116.);
      b = (b > 0.008856) * std::pow(b, 1/3.) + (b <= 0.008856) * (7.787 * b + 16/116.);

      top_data[n * 3 * size + 0 * size + s] = (116.0 * g) - 16.0;
      top_data[n * 3 * size + 1 * size + s] = 500.0 * (r - g);
      top_data[n * 3 * size + 2 * size + s] = 200.0 * (g - b);
    }
  }
}

template <typename Dtype>
void BgrToLabLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  if (propagate_down[0]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate a gradient.";
  }
}

#ifdef CPU_ONLY
STUB_GPU(BgrToLabLayer);
#endif

INSTANTIATE_CLASS(BgrToLabLayer);
REGISTER_LAYER_CLASS(BgrToLab);

}  // namespace caffe

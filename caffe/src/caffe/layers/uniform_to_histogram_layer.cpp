#include <algorithm>
#include <vector>

#include "caffe/layers/uniform_to_histogram_layer.hpp"

namespace caffe {

template <typename Dtype>
void UniformToHistogramLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  UniformToHistogramParameter param = this->layer_param().uniform_to_histogram_param();
  const int bins = param.bins();
  vector<int> top_shape;
  top_shape.push_back(bottom[0]->num());
  top_shape.push_back(bins);
  top_shape.push_back(bottom[0]->height());
  top_shape.push_back(bottom[0]->width());
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void UniformToHistogramLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  const int top_count = top[0]->count();
  const int size = bottom[0]->height() * bottom[0]->width();

  // Num bins
  const int bins = top[0]->shape(1);

  for (int i = 0; i < top_count; ++i) {
    top_data[i] = 0.0;
  }

  for (int i = 0; i < count; ++i) {
    // CDF value
    Dtype v = bottom_data[i];

    int n = i / size;
    int s = i % size;
    //
    int idx = (int)(v * bins);
    if (idx >= bins) {
        idx = bins - 1;
    }

    top_data[n * bins * size + idx * size + s] = 1.0;
  }
}

template <typename Dtype>
void UniformToHistogramLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  if (propagate_down[0]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate a gradient.";
  }
}

#ifdef CPU_ONLY
STUB_GPU(UniformToHistogramLayer);
#endif

INSTANTIATE_CLASS(UniformToHistogramLayer);
REGISTER_LAYER_CLASS(UniformToHistogram);

}  // namespace caffe

#include <algorithm>
#include <vector>

#include "caffe/layers/uniform_to_joint_histogram_layer.hpp"

namespace caffe {

template <typename Dtype>
void UniformToJointHistogramLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  UniformToHistogramParameter param = this->layer_param().uniform_to_histogram_param();
  const int bins = param.bins();
  vector<int> top_shape;
  top_shape.push_back(bottom[0]->num());
  top_shape.push_back(bins * bins);
  top_shape.push_back(bottom[0]->height());
  top_shape.push_back(bottom[0]->width());
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void UniformToJointHistogramLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data0 = bottom[0]->cpu_data();
  const Dtype* bottom_data1 = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  const int top_count = top[0]->count();
  const int size = bottom[0]->height() * bottom[0]->width();

  // Num bins
  UniformToHistogramParameter param = this->layer_param().uniform_to_histogram_param();
  const int bins = param.bins();
  //const int bins = top[0]->shape(1);

  for (int i = 0; i < top_count; ++i) {
    top_data[i] = 0.0;
  }

  for (int i = 0; i < count; ++i) {
    // CDF value
    Dtype v0 = bottom_data0[i];
    Dtype v1 = bottom_data1[i];

    int n = i / size;
    int s = i % size;
    //
    int idx0 = (int)(v0 * bins);
    if (idx0 >= bins) {
        idx0 = bins - 1;
    }
    int idx1 = (int)(v1 * bins);
    if (idx1 >= bins) {
        idx1 = bins - 1;
    }
    int idx = idx0 * bins + idx1;

    top_data[n * bins * bins * size + idx * size + s] = 1.0;
  }
}

template <typename Dtype>
void UniformToJointHistogramLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  if (propagate_down[0]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate a gradient.";
  }
}

#ifdef CPU_ONLY
STUB_GPU(UniformToJointHistogramLayer);
#endif

INSTANTIATE_CLASS(UniformToJointHistogramLayer);
REGISTER_LAYER_CLASS(UniformToJointHistogram);

}  // namespace caffe

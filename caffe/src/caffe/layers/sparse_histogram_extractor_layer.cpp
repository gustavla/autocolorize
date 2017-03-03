#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/sparse_histogram_extractor_layer.hpp"

#include <algorithm>

namespace caffe {

template <typename Dtype>
void SparseHistogramExtractorLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void SparseHistogramExtractorLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  SparseHistogramExtractorParameter param = this->layer_param().sparse_histogram_extractor_param();

  sizes_.clear();
  std::copy(param.size().begin(),
            param.size().end(),
            std::back_inserter(sizes_));

  CHECK_EQ(top.size(), sizes_.size());

  num_locations_ = bottom[0]->shape(1);
  int num = bottom[0]->shape(0);

  // Locations should be specified with two dimesions (h, w)
  CHECK_EQ(bottom[0]->shape(2), 2);

  vector<int> top_shape;
  top_shape.push_back(num * num_locations_);
  top_shape.push_back(bottom[1]->shape(1));

  for (int i = 0; i < top.size(); ++i) {
    top[i]->Reshape(top_shape);
  }

  integral_histogram_.ReshapeLike(*bottom[1]);
}

template <typename Dtype>
void SparseHistogramExtractorLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype *locations_data = bottom[0]->cpu_data();
  const Dtype *histogram_data = bottom[1]->cpu_data();
  //Dtype *histogram_data = top[0]->mutable_cpu_data();
  Dtype *integral_data = integral_histogram_.mutable_cpu_data();

  const int num = bottom[1]->shape(0);
  const int bins = bottom[1]->shape(1);
  const int height = bottom[1]->shape(2);
  const int width = bottom[1]->shape(3);

  // Create summed area table
  const int stride_n = bins * height * width;
  const int stride_c = height * width;
  const int stride_y = width;

  for (int n = 0, i = 0; n < num; ++n) {
    for (int c = 0; c < bins; ++c) {
      for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
          if (y == 0) {
            integral_data[i] = histogram_data[i];
          } else {
            //integral_data[i] = (integral_data[n * stride_n + c * stride_c + (y - 1) * stride_y + x] +
            integral_data[i] = integral_data[i - stride_y] + histogram_data[i];
          }
          ++i;
        }
      }
    }
  }

  for (int n = 0, i = 0; n < num; ++n) {
    for (int c = 0; c < bins; ++c) {
      for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
          if (x == 0) {
            // Leave unchanged
          } else {
            integral_data[i] += integral_data[i - 1];// + histogram_data[i];
          }
          ++i;
        }
      }
    }
  }

  for (int s = 0; s < sizes_.size(); ++s) {
    Dtype *data = top[s]->mutable_cpu_data();

    for (int n = 0; n < num; ++n) {
      for (int p = 0; p < num_locations_; ++p) {
        int off = (n * num_locations_ + p) * 2;
        int y = locations_data[off + 0];
        int x = locations_data[off + 1];

        int size = sizes_[s];
        Dtype window_size = Dtype(size * size);
        int halfsize = size / 2;
        int y0 = y - halfsize - 1;
        int y1 = y0 + size;
        int x0 = x - halfsize - 1;
        int x1 = x0 + size;


        if (y0 > bottom[1]->shape(2) - 1) {
          y0 = bottom[1]->shape(2) - 1;
        }
        if (y1 > bottom[1]->shape(2) - 1) {
          y1 = bottom[1]->shape(2) - 1;
        }
        if (x0 > bottom[1]->shape(3) - 1) {
          x0 = bottom[1]->shape(3) - 1;
        }
        if (x1 > bottom[1]->shape(3) - 1) {
          x1 = bottom[1]->shape(3) - 1;
        }

        for (int c = 0; c < bins; ++c) {
          Dtype v00, v01, v10, v11;
          if (y0 < 0 || x0 < 0) {
              v00 = 0;
          } else {
              v00 = integral_data[n * stride_n + c * stride_c + y0 * stride_y + x0];
          }

          if (y0 < 0 || x1 < 0) {
              v01 = 0;
          } else {
              v01 = integral_data[n * stride_n + c * stride_c + y0 * stride_y + x1];
          }

          if (y1 < 0 || x0 < 0) {
              v10 = 0;
          } else {
              v10 = integral_data[n * stride_n + c * stride_c + y1 * stride_y + x0];
          }

          if (y1 < 0 || x1 < 0) {
              v11 = 0;
          } else {
              v11 = integral_data[n * stride_n + c * stride_c + y1 * stride_y + x1];
          }

          data[(n * num_locations_ + p) * bins + c] = v11 - v01 - v10 + v00;
        }
      }
    }

    for (int n = 0; n < num * num_locations_; ++n) {
      Dtype sum = 0.0;
      for (int c = 0; c < bins; ++c) {
        sum += data[n * bins + c];
      }
      if (sum > 0) {
        for (int c = 0; c < bins; ++c) {
          data[n * bins + c] /= sum;
        }
      } else {
        // Create a uniform histogram
        for (int c = 0; c < bins; ++c) {
          data[n * bins + c] = 1. / bins;
        }
      }
    }
  }
}

template <typename Dtype>
void SparseHistogramExtractorLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  if (propagate_down[0]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to locations inputs.";
  }

  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to data input.";
  }
}

#ifdef CPU_ONLY
STUB_GPU(SparseHistogramExtractorLayer);
#endif

INSTANTIATE_CLASS(SparseHistogramExtractorLayer);
REGISTER_LAYER_CLASS(SparseHistogramExtractor);

}  // namespace caffe

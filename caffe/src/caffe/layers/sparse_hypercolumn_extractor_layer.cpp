#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/sparse_hypercolumn_extractor_layer.hpp"

#include <algorithm>

namespace caffe {

template <typename Dtype>
void SparseHypercolumnExtractorLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void SparseHypercolumnExtractorLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  SparseHypercolumnExtractorParameter param = this->layer_param().sparse_hypercolumn_extractor_param();

  scales_.clear();
  std::copy(param.scale().begin(),
            param.scale().end(),
            std::back_inserter(scales_));

  offsets_h_.clear();
  std::copy(param.offset_height().begin(),
            param.offset_height().end(),
            std::back_inserter(offsets_h_));

  offsets_w_.clear();
  std::copy(param.offset_width().begin(),
            param.offset_width().end(),
            std::back_inserter(offsets_w_));

  activation_mults_.clear();
  std::copy(param.activation_mult().begin(),
            param.activation_mult().end(),
            std::back_inserter(activation_mults_));

  num_channels_ = 0;
  num_locations_ = bottom[0]->shape(1);
  int num = bottom[0]->shape(0);

  // Locations should be specified with two dimesions (h, w)
  CHECK_EQ(bottom[0]->shape(2), 2);

  num_layers_ = 0;
  for (int i = 1; i < bottom.size(); ++i) {
      CHECK_EQ(bottom[i]->shape(0), num);
      channel_offsets_.push_back(num_channels_);
      num_channels_ += bottom[i]->shape(1);
      num_layers_ += 1;
  }

  while (activation_mults_.size() < num_layers_) {
    activation_mults_.push_back(1.0);
  }

  CHECK_EQ(offsets_w_.size(), num_layers_);
  CHECK_EQ(offsets_h_.size(), num_layers_);
  CHECK_EQ(scales_.size(), num_layers_);
  CHECK_EQ(activation_mults_.size(), num_layers_);

  vector<int> top_shape;
  top_shape.push_back(num * num_locations_);
  top_shape.push_back(num_channels_);
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void SparseHypercolumnExtractorLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype *locations_data = bottom[0]->cpu_data();
  Dtype *hypercol_data = top[0]->mutable_cpu_data();

  for (int n = 0; n < bottom[0]->shape(0); ++n) {
    for (int p = 0; p < num_locations_; ++p) {
      int off = n * num_locations_ * 2 + p * 2;
      Dtype y = locations_data[off + 0];
      Dtype x = locations_data[off + 1];

      int offset0 = n * num_locations_ * num_channels_ + p * num_channels_;

      for (int l = 0; l < num_layers_; ++l) {
        const Dtype *layer_data = bottom[1 + l]->cpu_data();
        int num_layer_channels = bottom[1 + l]->shape(1);

        Dtype offset_h = offsets_h_[l];
        Dtype offset_w = offsets_w_[l];
        int height = bottom[1 + l]->shape(2);
        int width = bottom[1 + l]->shape(3);
        Dtype scale = scales_[l];
        int layer_offset = n * num_layer_channels * height * width;

        // Location in the local layer
        Dtype ly = (y - offset_h) / scale;
        Dtype lx = (x - offset_w) / scale;

        // If the location is outside the integer grid, we'll snap it into
        // a valid location.
        ly = std::max(ly, Dtype(0));
        ly = std::min(ly, Dtype(height) - Dtype(1.0001));

        lx = std::max(lx, Dtype(0));
        lx = std::min(lx, Dtype(width) - Dtype(1.0001));

        // Fetch the four points around this location
        int ylo = static_cast<int>(ly);
        int xlo = static_cast<int>(lx);
        int yhi = ylo + 1;
        int xhi = xlo + 1;

        // Factors (distance to the corners)
        Dtype f_yhi = ly - ylo;
        Dtype f_ylo = yhi - ly;
        Dtype f_xhi = lx - xlo;
        Dtype f_xlo = xhi - lx;

        int cc = channel_offsets_[l];
        for (int c = 0; c < num_layer_channels; ++c) {
          int off1 = layer_offset + c * width * height;
          hypercol_data[offset0 + cc + c] = activation_mults_[l] *
              (f_ylo * (f_xlo * layer_data[off1 + ylo * width + xlo] +
                        f_xhi * layer_data[off1 + ylo * width + xhi]) +
               f_yhi * (f_xlo * layer_data[off1 + yhi * width + xlo] +
                        f_xhi * layer_data[off1 + yhi * width + xhi]));
        }
      }
    }
  }
}

template <typename Dtype>
void SparseHypercolumnExtractorLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  if (propagate_down[0]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to locations inputs.";
  }
  // Not sure if I need to reset these values
  for (int l = 0; l < num_layers_; ++l) {
    Dtype *layer_diff = bottom[1 + l]->mutable_cpu_diff();
    caffe_set(bottom[1 + l]->count(), Dtype(0), layer_diff);
  }

  const Dtype *locations_data = bottom[0]->cpu_data();
  const Dtype *hypercol_diff = top[0]->cpu_diff();

  for (int n = 0; n < bottom[0]->shape(0); ++n) {
    for (int p = 0; p < num_locations_; ++p) {
      int off = n * num_locations_ * 2 + p * 2;
      Dtype y = locations_data[off + 0];
      Dtype x = locations_data[off + 1];

      int offset0 = n * num_locations_ * num_channels_ + p * num_channels_;

      for (int l = 0; l < num_layers_; ++l) {
        Dtype *layer_diff = bottom[1 + l]->mutable_cpu_diff();
        int num_layer_channels = bottom[1 + l]->shape(1);

        Dtype offset_h = offsets_h_[l];
        Dtype offset_w = offsets_w_[l];
        int height = bottom[1 + l]->shape(2);
        int width = bottom[1 + l]->shape(3);
        Dtype scale = scales_[l];
        int layer_offset = n * num_layer_channels * height * width;

        // Location in the local layer
        Dtype ly = (y - offset_h) / scale;
        Dtype lx = (x - offset_w) / scale;

        // If the location is outside the integer grid, we'll snap it into
        // a valid location.
        ly = std::max(ly, Dtype(0));
        ly = std::min(ly, Dtype(height) - Dtype(1.0001));

        lx = std::max(lx, Dtype(0));
        lx = std::min(lx, Dtype(width) - Dtype(1.0001));

        // Fetch the four points around this location
        int ylo = static_cast<int>(ly);
        int xlo = static_cast<int>(lx);
        int yhi = ylo + 1;
        int xhi = xlo + 1;

        // Factors (distance to the corners)
        Dtype f_yhi = ly - ylo;
        Dtype f_ylo = yhi - ly;
        Dtype f_xhi = lx - xlo;
        Dtype f_xlo = xhi - lx;

        int cc = channel_offsets_[l];
        for (int c = 0; c < num_layer_channels; ++c) {
          int off1 = layer_offset + c * width * height;

          Dtype diff = hypercol_diff[offset0 + cc + c] * activation_mults_[l];
          layer_diff[off1 + ylo * width + xlo] += f_ylo * f_xlo * diff;
          layer_diff[off1 + yhi * width + xlo] += f_yhi * f_xlo * diff;
          layer_diff[off1 + ylo * width + xhi] += f_ylo * f_xhi * diff;
          layer_diff[off1 + yhi * width + xhi] += f_yhi * f_xhi * diff;
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(SparseHypercolumnExtractorLayer);
#endif

INSTANTIATE_CLASS(SparseHypercolumnExtractorLayer);
REGISTER_LAYER_CLASS(SparseHypercolumnExtractor);

}  // namespace caffe

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/sparse_hypercolumn_extractor_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void forward_kernel(
    const int nthreads,
    const int num_locations,
    const int num_channels,
    const int num_layer_channels,
    const int height,
    const int width,
    const int channel_offset,
    const Dtype scale,
    const Dtype offset_h,
    const Dtype offset_w,
    const Dtype mult,
    const Dtype *locations_data,
    const Dtype *layer_data,
    Dtype *hypercol_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // n: sample, p: position, c: channel
    int n = index / num_locations / num_layer_channels;
    int p = (index / num_layer_channels) % num_locations;
    int c = index % num_layer_channels;

    int off = (n * num_locations + p) * 2;
    Dtype y = locations_data[off];
    Dtype x = locations_data[off + 1];

    int offset0 = n * num_locations * num_channels + p * num_channels;

    int layer_offset = n * num_layer_channels * height * width;

    // Location in the local layer
    Dtype ly = (y - offset_h) / scale;
    Dtype lx = (x - offset_w) / scale;

    // If the location is outside the integer grid, we'll snap it into
    // a valid location.
    ly = max(ly, Dtype(0));
    ly = min(ly, Dtype(height) - Dtype(1.0001));

    lx = max(lx, Dtype(0));
    lx = min(lx, Dtype(width) - Dtype(1.0001));

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

    int cc = channel_offset;

    int off1 = layer_offset + c * width * height;
    hypercol_data[offset0 + cc + c] = mult *
        (f_ylo * (f_xlo * layer_data[off1 + ylo * width + xlo] +
                  f_xhi * layer_data[off1 + ylo * width + xhi]) +
         f_yhi * (f_xlo * layer_data[off1 + yhi * width + xlo] +
                  f_xhi * layer_data[off1 + yhi * width + xhi]));
  }
}

template <typename Dtype>
void SparseHypercolumnExtractorLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype *locations_data = bottom[0]->gpu_data();
  Dtype *hypercol_data = top[0]->mutable_gpu_data();

  for (int l = 0; l < num_layers_; ++l) {
    int num_layer_channels = bottom[1 + l]->shape(1);
    int total_count = bottom[0]->shape(0) * num_locations_ * num_layer_channels;
    const Dtype* layer_data = bottom[1 + l]->gpu_data();
    int height = bottom[1 + l]->shape(2);
    int width = bottom[1 + l]->shape(3);
    forward_kernel<Dtype><<<CAFFE_GET_BLOCKS(total_count), CAFFE_CUDA_NUM_THREADS>>>(
        total_count,
        num_locations_,
        num_channels_,
        num_layer_channels,
        height,
        width,
        channel_offsets_[l],
        scales_[l],
        offsets_h_[l],
        offsets_w_[l],
        activation_mults_[l],
        locations_data,
        layer_data,
        hypercol_data);
  }
}

template <typename Dtype>
__global__ void backward_kernel(
    const int nthreads,
    const int num_locations,
    const int num_channels,
    const int num_layer_channels,
    const int height,
    const int width,
    const int channel_offset,
    const Dtype scale,
    const Dtype offset_h,
    const Dtype offset_w,
    const Dtype mult,
    const Dtype *locations_data,
    const Dtype *hypercol_diff,
    Dtype *layer_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // n: sample, p: position, c: channel
    int n = index / num_locations / num_layer_channels;
    int p = (index / num_layer_channels) % num_locations;
    int c = index % num_layer_channels;

    int off = (n * num_locations + p) * 2;
    Dtype y = locations_data[off];
    Dtype x = locations_data[off + 1];

    int offset0 = n * num_locations * num_channels + p * num_channels;

    int layer_offset = n * num_layer_channels * height * width;

    // Location in the local layer
    Dtype ly = (y - offset_h) / scale;
    Dtype lx = (x - offset_w) / scale;

    // If the location is outside the integer grid, we'll snap it into
    // a valid location.
    ly = max(ly, Dtype(0));
    ly = min(ly, Dtype(height) - Dtype(1.0001));

    lx = max(lx, Dtype(0));
    lx = min(lx, Dtype(width) - Dtype(1.0001));

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

    int cc = channel_offset;
    Dtype diff = hypercol_diff[offset0 + cc + c] * mult;

    int off1 = layer_offset + c * width * height;

#if 1
    atomicAdd(&layer_diff[off1 + ylo * width + xlo], f_ylo * f_xlo * diff);
    atomicAdd(&layer_diff[off1 + yhi * width + xlo], f_yhi * f_xlo * diff);
    atomicAdd(&layer_diff[off1 + ylo * width + xhi], f_ylo * f_xhi * diff);
    atomicAdd(&layer_diff[off1 + yhi * width + xhi], f_yhi * f_xhi * diff);
#else
    layer_diff[off1 + ylo * width + xlo] += f_ylo * f_xlo * diff;
    layer_diff[off1 + yhi * width + xlo] += f_yhi * f_xlo * diff;
    layer_diff[off1 + ylo * width + xhi] += f_ylo * f_xhi * diff;
    layer_diff[off1 + yhi * width + xhi] += f_yhi * f_xhi * diff;
#endif
  }
}

// Note: The general templated version will fall-back to the CPU version.
// This will happen only for 64-bit floats, since the 32-bit version is defined below
template <typename Dtype>
void SparseHypercolumnExtractorLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Backward_cpu(top, propagate_down, bottom);
}

template <>
void SparseHypercolumnExtractorLayer<float>::Backward_gpu(const vector<Blob<float>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<float>*>& bottom) {
  if (propagate_down[0]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to pairs inputs.";
  }

  // Reset gradients
  for (int l = 0; l < num_layers_; ++l) {
    float *layer_diff = bottom[1 + l]->mutable_gpu_diff();
    caffe_gpu_set(bottom[1 + l]->count(), 0.0f, layer_diff);
  }

  const float *locations_data = bottom[0]->gpu_data();
  const float *hypercolumn_diff = top[0]->gpu_diff();

  for (int l = 0; l < num_layers_; ++l) {
    if (propagate_down[1 + l]) {
      int num_layer_channels = bottom[1 + l]->shape(1);
      int total_count = bottom[0]->shape(0) * num_locations_ * num_layer_channels;
      float* layer_diff = bottom[1 + l]->mutable_gpu_diff();
      int height = bottom[1 + l]->shape(2);
      int width = bottom[1 + l]->shape(3);

      backward_kernel<<<CAFFE_GET_BLOCKS(total_count), CAFFE_CUDA_NUM_THREADS>>>(
          total_count,
          num_locations_,
          num_channels_,
          num_layer_channels,
          height,
          width,
          channel_offsets_[l],
          scales_[l],
          offsets_h_[l],
          offsets_w_[l],
          activation_mults_[l],
          locations_data,
          hypercolumn_diff,
          layer_diff);
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SparseHypercolumnExtractorLayer);

}  // namespace caffe

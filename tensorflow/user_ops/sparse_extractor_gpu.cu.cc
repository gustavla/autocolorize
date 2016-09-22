#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/util/tensor_format.h"

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

using namespace tensorflow;

template <typename Device>
struct SparseExtractorForward {
  bool operator()(const Device& d,
                  typename TTypes<float, 3>::ConstTensor centroids,
                  typename TTypes<float, 4>::ConstTensor input,
                  float scale,
                  float offset_h, float offset_w,
                  TensorFormat data_format,
                  typename TTypes<float, 2>::Tensor output);
};

template <typename Device>
struct SparseExtractorBackward {
  bool operator()(const Device& d,
                  typename TTypes<float, 3>::ConstTensor centroids,
                  typename TTypes<float, 4>::ConstTensor input,
                  typename TTypes<float, 2>::ConstTensor output,
                  typename TTypes<float, 2>::ConstTensor output_gradient,
                  float scale,
                  float offset_h, float offset_w,
                  TensorFormat data_format,
                  typename TTypes<float, 4>::Tensor input_gradient);
};

typedef Eigen::GpuDevice GPUDevice;

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

namespace {

__global__ void forward_kernel_nhwc(
    const int nthreads,
    const int num_locations,
    const int num_channels,
    const int num_layer_channels,
    const int height,
    const int width,
    const int channel_offset,
    const float scale,
    const float offset_h,
    const float offset_w,
    const float mult,
    const float *locations_data,
    const float *layer_data,
    float *hypercol_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // n: sample, p: position, c: channel
    int n = index / num_locations / num_layer_channels;
    int p = (index / num_layer_channels) % num_locations;
    int c = index % num_layer_channels;

    int off = (n * num_locations + p) * 2;
    float y = locations_data[off];
    float x = locations_data[off + 1];

    int offset0 = n * num_locations * num_channels + p * num_channels;

    int layer_offset = n * num_layer_channels * height * width;

    // Location in the local layer
    float ly = (y - offset_h) / scale;
    float lx = (x - offset_w) / scale;

    // If the location is outside the integer grid, we'll snap it into
    // a valid location.
    ly = max(ly, float(0));
    ly = min(ly, float(height) - float(1.0001));

    lx = max(lx, float(0));
    lx = min(lx, float(width) - float(1.0001));

    // Fetch the four points around this location
    int ylo = static_cast<int>(ly);
    int xlo = static_cast<int>(lx);
    int yhi = ylo + 1;
    int xhi = xlo + 1;

    // Factors (distance to the corners)
    float f_yhi = ly - ylo;
    float f_ylo = yhi - ly;
    float f_xhi = lx - xlo;
    float f_xlo = xhi - lx;

    int cc = channel_offset;

    // NCHW
    // n * num_layer_channels * height * width + c * width * height + ylo * width + xlo
    //
    // NHWC
    // n * height * width * num_layer_channels + ylo * width * num_layer_channels + xlo * num_layer_channels + c

    //int off1 = layer_offset + c * width * height;
#define INDEX(y, x) n * height * width * num_layer_channels + y * width * num_layer_channels + x * num_layer_channels + c
    hypercol_data[offset0 + cc + c] = mult *
        (f_ylo * (f_xlo * layer_data[INDEX(ylo, xlo)] +
                  f_xhi * layer_data[INDEX(ylo, xhi)]) +
         f_yhi * (f_xlo * layer_data[INDEX(yhi, xlo)] +
                  f_xhi * layer_data[INDEX(yhi, xhi)]));
#undef INDEX
  }
}

__global__ void forward_kernel_nchw(
    const int nthreads,
    const int num_locations,
    const int num_channels,
    const int num_layer_channels,
    const int height,
    const int width,
    const int channel_offset,
    const float scale,
    const float offset_h,
    const float offset_w,
    const float mult,
    const float *locations_data,
    const float *layer_data,
    float *hypercol_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // n: sample, p: position, c: channel
    int n = index / num_locations / num_layer_channels;
    int p = (index / num_layer_channels) % num_locations;
    int c = index % num_layer_channels;

    int off = (n * num_locations + p) * 2;
    float y = locations_data[off];
    float x = locations_data[off + 1];

    int offset0 = n * num_locations * num_channels + p * num_channels;

    int layer_offset = n * num_layer_channels * height * width;

    // Location in the local layer
    float ly = (y - offset_h) / scale;
    float lx = (x - offset_w) / scale;

    // If the location is outside the integer grid, we'll snap it into
    // a valid location.
    ly = max(ly, float(0));
    ly = min(ly, float(height) - float(1.0001));

    lx = max(lx, float(0));
    lx = min(lx, float(width) - float(1.0001));

    // Fetch the four points around this location
    int ylo = static_cast<int>(ly);
    int xlo = static_cast<int>(lx);
    int yhi = ylo + 1;
    int xhi = xlo + 1;

    // Factors (distance to the corners)
    float f_yhi = ly - ylo;
    float f_ylo = yhi - ly;
    float f_xhi = lx - xlo;
    float f_xlo = xhi - lx;

    int cc = channel_offset;

    // NCHW
    // n * num_layer_channels * height * width + c * width * height + ylo * width + xlo
    //
    // NHWC
    // n * height * width * num_layer_channels + ylo * width * num_layer_channels + xlo * num_layer_channels + c

    //int off1 = layer_offset + c * width * height;
#define INDEX(y, x) n * height * width * num_layer_channels + c * height * width + y * width + x
    hypercol_data[offset0 + cc + c] = mult *
        (f_ylo * (f_xlo * layer_data[INDEX(ylo, xlo)] +
                  f_xhi * layer_data[INDEX(ylo, xhi)]) +
         f_yhi * (f_xlo * layer_data[INDEX(yhi, xlo)] +
                  f_xhi * layer_data[INDEX(yhi, xhi)]));
#undef INDEX
  }
}

__global__ void backward_kernel_nhwc(
    const int nthreads,
    const int num_locations,
    const int num_channels,
    const int num_layer_channels,
    const int height,
    const int width,
    const int channel_offset,
    const float scale,
    const float offset_h,
    const float offset_w,
    const float mult,
    const float *locations_data,
    const float *hypercol_diff,
    float *layer_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // n: sample, p: position, c: channel
    int n = index / num_locations / num_layer_channels;
    int p = (index / num_layer_channels) % num_locations;
    int c = index % num_layer_channels;

    int off = (n * num_locations + p) * 2;
    float y = locations_data[off];
    float x = locations_data[off + 1];

    int offset0 = n * num_locations * num_channels + p * num_channels;

    int layer_offset = n * num_layer_channels * height * width;

    // Location in the local layer
    float ly = (y - offset_h) / scale;
    float lx = (x - offset_w) / scale;

    // If the location is outside the integer grid, we'll snap it into
    // a valid location.
    ly = max(ly, float(0));
    ly = min(ly, float(height) - float(1.0000001));

    lx = max(lx, float(0));
    lx = min(lx, float(width) - float(1.0000001));

    // Fetch the four points around this location
    int ylo = static_cast<int>(ly);
    int xlo = static_cast<int>(lx);
    int yhi = ylo + 1;
    int xhi = xlo + 1;

    // Factors (distance to the corners)
    float f_yhi = ly - ylo;
    float f_ylo = yhi - ly;
    float f_xhi = lx - xlo;
    float f_xlo = xhi - lx;

    int cc = channel_offset;
    float diff = hypercol_diff[offset0 + cc + c] * mult;

    //int off1 = layer_offset + c * width * height;

#if 1
#define INDEX(y, x) n * height * width * num_layer_channels + y * width * num_layer_channels + x * num_layer_channels + c
    atomicAdd(&layer_diff[INDEX(ylo, xlo)], f_ylo * f_xlo * diff);
    atomicAdd(&layer_diff[INDEX(ylo, xhi)], f_ylo * f_xhi * diff);
    atomicAdd(&layer_diff[INDEX(yhi, xlo)], f_yhi * f_xlo * diff);
    atomicAdd(&layer_diff[INDEX(yhi, xhi)], f_yhi * f_xhi * diff);
#undef INDEX
#else
    layer_diff[off1 + ylo * width + xlo] += f_ylo * f_xlo * diff;
    layer_diff[off1 + yhi * width + xlo] += f_yhi * f_xlo * diff;
    layer_diff[off1 + ylo * width + xhi] += f_ylo * f_xhi * diff;
    layer_diff[off1 + yhi * width + xhi] += f_yhi * f_xhi * diff;
#endif
  }
}

__global__ void backward_kernel_nchw(
    const int nthreads,
    const int num_locations,
    const int num_channels,
    const int num_layer_channels,
    const int height,
    const int width,
    const int channel_offset,
    const float scale,
    const float offset_h,
    const float offset_w,
    const float mult,
    const float *locations_data,
    const float *hypercol_diff,
    float *layer_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // n: sample, p: position, c: channel
    int n = index / num_locations / num_layer_channels;
    int p = (index / num_layer_channels) % num_locations;
    int c = index % num_layer_channels;

    int off = (n * num_locations + p) * 2;
    float y = locations_data[off];
    float x = locations_data[off + 1];

    int offset0 = n * num_locations * num_channels + p * num_channels;

    int layer_offset = n * num_layer_channels * height * width;

    // Location in the local layer
    float ly = (y - offset_h) / scale;
    float lx = (x - offset_w) / scale;

    // If the location is outside the integer grid, we'll snap it into
    // a valid location.
    ly = max(ly, float(0));
    ly = min(ly, float(height) - float(1.0000001));

    lx = max(lx, float(0));
    lx = min(lx, float(width) - float(1.0000001));

    // Fetch the four points around this location
    int ylo = static_cast<int>(ly);
    int xlo = static_cast<int>(lx);
    int yhi = ylo + 1;
    int xhi = xlo + 1;

    // Factors (distance to the corners)
    float f_yhi = ly - ylo;
    float f_ylo = yhi - ly;
    float f_xhi = lx - xlo;
    float f_xlo = xhi - lx;

    int cc = channel_offset;
    float diff = hypercol_diff[offset0 + cc + c] * mult;

#if 1
#define INDEX(y, x) n * height * width * num_layer_channels + c * height * width + y * width + x
    atomicAdd(&layer_diff[INDEX(ylo, xlo)], f_ylo * f_xlo * diff);
    atomicAdd(&layer_diff[INDEX(ylo, xhi)], f_ylo * f_xhi * diff);
    atomicAdd(&layer_diff[INDEX(yhi, xlo)], f_yhi * f_xlo * diff);
    atomicAdd(&layer_diff[INDEX(yhi, xhi)], f_yhi * f_xhi * diff);
#undef INDEX
#else
    layer_diff[off1 + ylo * width + xlo] += f_ylo * f_xlo * diff;
    layer_diff[off1 + yhi * width + xlo] += f_yhi * f_xlo * diff;
    layer_diff[off1 + ylo * width + xhi] += f_ylo * f_xhi * diff;
    layer_diff[off1 + yhi * width + xhi] += f_yhi * f_xhi * diff;
#endif
  }
}

} // namespace: anonymous

//namespace functor {

template <>
bool SparseExtractorForward<GPUDevice>::operator()(const GPUDevice& d,
                  typename TTypes<float, 3>::ConstTensor centroids,
                  typename TTypes<float, 4>::ConstTensor input,
                  float scale,
                  float offset_h, float offset_w,
                  TensorFormat data_format,
                  typename TTypes<float, 2>::Tensor output) {
  int dim_batch = input.dimension(0);
  int dim_locations = centroids.dimension(1);
  int dim_channels;
  int dim_height;
  int dim_width;
  if (data_format == FORMAT_NHWC) {
    dim_channels = input.dimension(3);
    dim_height = input.dimension(1);
    dim_width = input.dimension(2);
  } else if (data_format == FORMAT_NCHW) {
    dim_channels = input.dimension(1);
    dim_height = input.dimension(2);
    dim_width = input.dimension(3);
  }

  int total_count = dim_batch * dim_locations * dim_channels;

  CudaLaunchConfig config = GetCudaLaunchConfig(total_count, d);
  if (data_format == FORMAT_NHWC) {
    forward_kernel_nhwc<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
        total_count,
        dim_locations,
        dim_channels,
        dim_channels,
        dim_height,
        dim_width,
        0,
        scale,
        offset_h,
        offset_w,
        1.0,
        centroids.data(),
        input.data(),
        output.data());
  } else if (data_format == FORMAT_NCHW) {
    forward_kernel_nchw<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
        total_count,
        dim_locations,
        dim_channels,
        dim_channels,
        dim_height,
        dim_width,
        0,
        scale,
        offset_h,
        offset_w,
        1.0,
        centroids.data(),
        input.data(),
        output.data());
  }

  return true;
}

template <>
bool SparseExtractorBackward<GPUDevice>::operator()(const GPUDevice& d,
                  typename TTypes<float, 3>::ConstTensor centroids,
                  typename TTypes<float, 4>::ConstTensor input,
                  typename TTypes<float, 2>::ConstTensor output,
                  typename TTypes<float, 2>::ConstTensor output_gradient,
                  float scale,
                  float offset_h, float offset_w,
                  TensorFormat data_format,
                  typename TTypes<float, 4>::Tensor input_gradient) {
  int dim_batch = input.dimension(0);
  int dim_locations = centroids.dimension(1);
  int dim_channels;
  int dim_height;
  int dim_width;
  if (data_format == FORMAT_NHWC) {
    dim_channels = input.dimension(3);
    dim_height = input.dimension(1);
    dim_width = input.dimension(2);
  } else if (data_format == FORMAT_NCHW) {
    dim_channels = input.dimension(1);
    dim_height = input.dimension(2);
    dim_width = input.dimension(3);
  }

  int total_count = dim_batch * dim_locations * dim_channels;

  cudaMemset(input_gradient.data(), 0,
             sizeof(float) * dim_batch * dim_height * dim_width * dim_channels);

  CudaLaunchConfig config = GetCudaLaunchConfig(total_count, d);
  if (data_format == FORMAT_NHWC) {
    backward_kernel_nhwc<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
        total_count,
        dim_locations,
        dim_channels,
        dim_channels,
        dim_height,
        dim_width,
        0,
        scale,
        offset_h,
        offset_w,
        1.0,
        centroids.data(),
        output_gradient.data(),
        input_gradient.data());
  } else if (data_format == FORMAT_NCHW) {
    backward_kernel_nchw<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
        total_count,
        dim_locations,
        dim_channels,
        dim_channels,
        dim_height,
        dim_width,
        0,
        scale,
        offset_h,
        offset_w,
        1.0,
        centroids.data(),
        output_gradient.data(),
        input_gradient.data());
  }

  return true;
}

template struct SparseExtractorForward<GPUDevice>;
template struct SparseExtractorBackward<GPUDevice>;

//} // namespace: functor

#endif  // GOOGLE_CUDA

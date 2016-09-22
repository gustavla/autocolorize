#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/util/tensor_format.h"

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"

#include <iostream>

using namespace tensorflow;
using std::cout;

REGISTER_OP("SparseExtractor")
    .Input("centroids: float32")
    .Input("input: float32")
    .Input("scale: float32")
    .Input("offset: float32")
    .Attr(GetConvnetDataFormatAttrString())
    .Output("columns: float32");

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device>
struct SparseExtractorForward {
  // We assume that the tensor sizes are correct.
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

template <>
bool SparseExtractorForward<CPUDevice>::operator()(const CPUDevice& d,
                  typename TTypes<float, 3>::ConstTensor centroids,
                  typename TTypes<float, 4>::ConstTensor input,
                  float scale,
                  float offset_h, float offset_w,
                  TensorFormat data_format,
                  typename TTypes<float, 2>::Tensor output) {
  int dim_batch = input.dimension(0);
  int dim_locations = centroids.dimension(1);

  if (data_format == FORMAT_NHWC) {
    int dim_channels = input.dimension(3);
    int dim_height = input.dimension(1);
    int dim_width = input.dimension(2);

    for (int n = 0; n < dim_batch; ++n) {
      for (int p = 0; p < dim_locations; ++p) {
        float y = centroids(n, p, 0);
        float x = centroids(n, p, 1);

        float ly = (y - offset_h) / scale;
        float lx = (x - offset_w) / scale;

        ly = std::max(ly, 0.f);
        ly = std::min(ly, (float)dim_height - 1.0000001f);

        lx = std::max(lx, 0.f);
        lx = std::min(lx, (float)dim_width - 1.0000001f);

        int ylo = static_cast<int>(ly);
        int xlo = static_cast<int>(lx);
        int yhi = ylo + 1;
        int xhi = xlo + 1;

        // Factors (distance to the corners)
        float f_yhi = ly - ylo;
        float f_ylo = yhi - ly;
        float f_xhi = lx - xlo;
        float f_xlo = xhi - lx;

        for (int c = 0; c < dim_channels; ++c) {
          output(n * dim_locations + p, c) =
            (f_ylo * (f_xlo * input(n, ylo, xlo, c) +
                      f_xhi * input(n, ylo, xhi, c)) +
             f_yhi * (f_xlo * input(n, yhi, xlo, c) +
                      f_xhi * input(n, yhi, xhi, c)));

        }
      }
    }
  } else if (data_format == FORMAT_NCHW) {
    int dim_channels = input.dimension(1);
    int dim_height = input.dimension(2);
    int dim_width = input.dimension(3);

    for (int n = 0; n < dim_batch; ++n) {
      for (int p = 0; p < dim_locations; ++p) {
        float y = centroids(n, p, 0);
        float x = centroids(n, p, 1);

        float ly = (y - offset_h) / scale;
        float lx = (x - offset_w) / scale;

        ly = std::max(ly, 0.f);
        ly = std::min(ly, (float)dim_height - 1.0000001f);

        lx = std::max(lx, 0.f);
        lx = std::min(lx, (float)dim_width - 1.0000001f);

        int ylo = static_cast<int>(ly);
        int xlo = static_cast<int>(lx);
        int yhi = ylo + 1;
        int xhi = xlo + 1;

        // Factors (distance to the corners)
        float f_yhi = ly - ylo;
        float f_ylo = yhi - ly;
        float f_xhi = lx - xlo;
        float f_xlo = xhi - lx;

        for (int c = 0; c < dim_channels; ++c) {
          output(n * dim_locations + p, c) =
            (f_ylo * (f_xlo * input(n, c, ylo, xlo) +
                      f_xhi * input(n, c, ylo, xhi)) +
             f_yhi * (f_xlo * input(n, c, yhi, xlo) +
                      f_xhi * input(n, c, yhi, xhi)));

        }
      }
    }
  }
  return true;
}

template <>
bool SparseExtractorBackward<CPUDevice>::operator()(const CPUDevice& d,
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

  if (data_format == FORMAT_NHWC) {
    int dim_channels = input.dimension(3);
    int dim_height = input.dimension(1);
    int dim_width = input.dimension(2);

    for (int n = 0; n < dim_batch; ++n) {
      for (int y = 0; y < dim_height; ++y) {
        for (int x = 0; x < dim_height; ++x) {
          for (int c = 0; c < dim_channels; ++c) {
            input_gradient(n, y, x, c) = 0.0;
          }
        }
      }
    }

    for (int n = 0; n < dim_batch; ++n) {
      for (int p = 0; p < dim_locations; ++p) {
        float y = centroids(n, p, 0);
        float x = centroids(n, p, 1);

        float ly = (y - offset_h) / scale;
        float lx = (x - offset_w) / scale;

        ly = std::max(ly, 0.f);
        ly = std::min(ly, (float)dim_height - 1.0001f);

        lx = std::max(lx, 0.f);
        lx = std::min(lx, (float)dim_width - 1.0001f);

        int ylo = static_cast<int>(ly);
        int xlo = static_cast<int>(lx);
        int yhi = ylo + 1;
        int xhi = xlo + 1;

        // Factors (distance to the corners)
        float f_yhi = ly - ylo;
        float f_ylo = yhi - ly;
        float f_xhi = lx - xlo;
        float f_xlo = xhi - lx;

        //int cc = channel_offsets_[l];
        for (int c = 0; c < dim_channels; ++c) {
          float diff = output_gradient(n * dim_locations + p, c);
          input_gradient(n, ylo, xlo, c) += f_ylo * f_xlo * diff;
          input_gradient(n, yhi, xlo, c) += f_yhi * f_xlo * diff;
          input_gradient(n, ylo, xhi, c) += f_ylo * f_xhi * diff;
          input_gradient(n, yhi, xhi, c) += f_yhi * f_xhi * diff;
        }
      }
    }
  } else if (data_format == FORMAT_NCHW) {
    int dim_channels = input.dimension(1);
    int dim_height = input.dimension(2);
    int dim_width = input.dimension(3);

    for (int n = 0; n < dim_batch; ++n) {
      for (int c = 0; c < dim_channels; ++c) {
        for (int y = 0; y < dim_height; ++y) {
          for (int x = 0; x < dim_height; ++x) {
            input_gradient(n, c, y, x) = 0.0;
          }
        }
      }
    }

    for (int n = 0; n < dim_batch; ++n) {
      for (int p = 0; p < dim_locations; ++p) {
        float y = centroids(n, p, 0);
        float x = centroids(n, p, 1);

        float ly = (y - offset_h) / scale;
        float lx = (x - offset_w) / scale;

        ly = std::max(ly, 0.f);
        ly = std::min(ly, (float)dim_height - 1.0001f);

        lx = std::max(lx, 0.f);
        lx = std::min(lx, (float)dim_width - 1.0001f);

        int ylo = static_cast<int>(ly);
        int xlo = static_cast<int>(lx);
        int yhi = ylo + 1;
        int xhi = xlo + 1;

        // Factors (distance to the corners)
        float f_yhi = ly - ylo;
        float f_ylo = yhi - ly;
        float f_xhi = lx - xlo;
        float f_xlo = xhi - lx;

        //int cc = channel_offsets_[l];
        for (int c = 0; c < dim_channels; ++c) {
          float diff = output_gradient(n * dim_locations + p, c);
          input_gradient(n, c, ylo, xlo) += f_ylo * f_xlo * diff;
          input_gradient(n, c, yhi, xlo) += f_yhi * f_xlo * diff;
          input_gradient(n, c, ylo, xhi) += f_ylo * f_xhi * diff;
          input_gradient(n, c, yhi, xhi) += f_yhi * f_xhi * diff;
        }
      }
    }
  }

  return true;
}

//} // functor namespace

template <typename Device>
class SparseExtractorOp : public OpKernel {
 public:
  explicit SparseExtractorOp(OpKernelConstruction* context) : OpKernel(context) {
    string str_data_format;
    auto status = context->GetAttr(StringPiece("data_format"), &str_data_format);
    if (status.ok()) {
      OP_REQUIRES(context, FormatFromString(str_data_format, &this->data_format_),
                  errors::InvalidArgument("Invalid data format"));
    } else {
      this->data_format_ = FORMAT_NHWC;
    }
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& centroids_tensor = context->input(0);
    const Tensor& input_tensor = context->input(1);

    typename TTypes<float, 3>::ConstTensor centroids = centroids_tensor.tensor<float, 3>();
    typename TTypes<float, 4>::ConstTensor input = input_tensor.tensor<float, 4>();

    const Tensor& scale_tensor = context->input(2);
    OP_REQUIRES(
        context, IsLegacyScalar(scale_tensor.shape()),
        errors::InvalidArgument(
            "Scale tensor should be a scalar integer, but got shape ",
            scale_tensor.shape().DebugString()));
    const float scale = scale_tensor.scalar<float>()();

    const Tensor& offset_tensor = context->input(3);
    OP_REQUIRES(context, IsLegacyVector(offset_tensor.shape()),
                errors::InvalidArgument("offset input must be 1-D, not shape ",
                                        offset_tensor.shape().DebugString()));
    OP_REQUIRES(context, offset_tensor.NumElements() == 2,
                errors::InvalidArgument("offset must be of length two, not ",
                                        offset_tensor.NumElements()));
    const int64 num_offsets = offset_tensor.NumElements();
    auto Ovec = offset_tensor.flat<float>();

    const float offset_h = Ovec(0);
    const float offset_w = Ovec(1);

    int dim_locations = centroids_tensor.shape().dim_size(1);
    int dim_batch = input_tensor.shape().dim_size(GetTensorDimIndex(this->data_format_, 'N'));
    int dim_channels = input_tensor.shape().dim_size(GetTensorDimIndex(this->data_format_, 'C'));

    TensorShape output_shape({dim_batch * dim_locations, dim_channels});

    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape,
                                                     &output_tensor));

    typename TTypes<float, 2>::Tensor output = output_tensor->tensor<float, 2>();

    bool status = SparseExtractorForward<Device>()(
        context->eigen_device<Device>(), centroids, input, scale, offset_h, offset_w, data_format_, output);
  }

 private:
  TensorFormat data_format_;
};

REGISTER_KERNEL_BUILDER(Name("SparseExtractor")
                            .Device(DEVICE_CPU)
                            .HostMemory("scale")
                            .HostMemory("offset"),
                        SparseExtractorOp<CPUDevice>);

REGISTER_KERNEL_BUILDER(Name("SparseExtractor")
                            .Device(DEVICE_GPU)
                            .HostMemory("scale")
                            .HostMemory("offset"),
                        SparseExtractorOp<GPUDevice>);

REGISTER_OP("SparseExtractorGrad")
    .Input("centroids: float32")
    .Input("input: float32")
    .Input("output: float32")
    .Input("output_gradient: float32")
    .Input("scale: float32")
    .Input("offset: float32")
    .Attr(GetConvnetDataFormatAttrString())
    .Output("input_gradient: float32");

template <typename Device>
class SparseExtractorGradOp : public OpKernel {
 public:
  explicit SparseExtractorGradOp(OpKernelConstruction* context) : OpKernel(context) {
    string str_data_format;
    auto status = context->GetAttr(StringPiece("data_format"), &str_data_format);
    if (status.ok()) {
      OP_REQUIRES(context, FormatFromString(str_data_format, &this->data_format_),
                  errors::InvalidArgument("Invalid data format"));
    } else {
      this->data_format_ = FORMAT_NHWC;
    }
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& centroids_tensor = context->input(0);
    const Tensor& input_tensor = context->input(1);
    const Tensor& output_tensor = context->input(2);
    const Tensor& output_gradient_tensor = context->input(3);

    typename TTypes<float, 4>::ConstTensor input = input_tensor.tensor<float, 4>();
    typename TTypes<float, 2>::ConstTensor output = output_tensor.tensor<float, 2>();
    typename TTypes<float, 2>::ConstTensor output_gradient = output_gradient_tensor.tensor<float, 2>();
    typename TTypes<float, 3>::ConstTensor centroids = centroids_tensor.tensor<float, 3>();

    const Tensor& scale_tensor = context->input(4);
    OP_REQUIRES(
        context, IsLegacyScalar(scale_tensor.shape()),
        errors::InvalidArgument(
            "Scale tensor should be a scalar integer, but got shape ",
            scale_tensor.shape().DebugString()));

    const float scale = scale_tensor.scalar<float>()();

    const Tensor& offset_tensor = context->input(5);
    OP_REQUIRES(context, IsLegacyVector(offset_tensor.shape()),
                errors::InvalidArgument("offset input must be 1-D, not shape ",
                                        offset_tensor.shape().DebugString()));
    OP_REQUIRES(context, offset_tensor.NumElements() == 2,
                errors::InvalidArgument("offset must be of length two, not ",
                                        offset_tensor.NumElements()));
    const int64 num_offsets = offset_tensor.NumElements();
    auto Ovec = offset_tensor.flat<float>();

    const float offset_h = Ovec(0);
    const float offset_w = Ovec(1);

    TensorShape input_gradient_shape(input_tensor.shape());
    Tensor *input_gradient_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_gradient_shape,
                                                     &input_gradient_tensor));

    typename TTypes<float, 4>::Tensor input_gradient = input_gradient_tensor->tensor<float, 4>();

    bool status = SparseExtractorBackward<Device>()(
        context->eigen_device<Device>(), centroids, input, output, output_gradient, scale, offset_h, offset_w, data_format_, input_gradient);
  }

 private:
  TensorFormat data_format_;
};

REGISTER_KERNEL_BUILDER(Name("SparseExtractorGrad")
                            .Device(DEVICE_CPU)
                            .HostMemory("scale")
                            .HostMemory("offset"),
                        SparseExtractorGradOp<CPUDevice>);

REGISTER_KERNEL_BUILDER(Name("SparseExtractorGrad")
                            .Device(DEVICE_GPU)
                            .HostMemory("scale")
                            .HostMemory("offset"),
                        SparseExtractorGradOp<GPUDevice>);

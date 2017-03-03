#ifndef CAFFE_SPARSE_PATCH_LAYER_HPP
#define CAFFE_SPARSE_PATCH_LAYER_HPP

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"

namespace caffe {

/**
 * Takes locations and extracts a stack of histograms.
 */
template <typename Dtype>
class SparseHistogramExtractorLayer : public Layer<Dtype> {
 public:
  explicit SparseHistogramExtractorLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SparseHistogramExtractor"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int num_layers_;
  int num_channels_;
  int num_locations_;
  std::vector<int> sizes_;

  // Temporary storage for the summed area table
  Blob<Dtype> integral_histogram_;
};

}  // namespace caffe

#endif  // CAFFE_SPARSE_PATCH_LAYER_HPP


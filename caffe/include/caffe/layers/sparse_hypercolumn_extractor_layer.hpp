#ifndef CAFFE_SPARSE_HYPERCOLUMN_LAYER_HPP
#define CAFFE_SPARSE_HYPERCOLUMN_LAYER_HPP

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"

namespace caffe {

/**
 * \ingroup ttic
 * @brief This extracts locations of hypercolumns. It should be faster, and
 * more importantly use less memory, than the original, since you do not need
 * separate upscaling layers. It does the bilinear upscaling by itself.
 *
 * @author Gustav Larsson
 */
template <typename Dtype>
class SparseHypercolumnExtractorLayer : public Layer<Dtype> {
 public:
  explicit SparseHypercolumnExtractorLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SparseHypercolumnExtractor"; }
  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

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
  int num_channels_; // in the entire hypercolumn
  int num_locations_;
  vector<Dtype> offsets_h_;
  vector<Dtype> offsets_w_;
  vector<Dtype> scales_;
  vector<Dtype> activation_mults_;
  vector<int> channel_offsets_;
};

}  // namespace caffe

#endif  // CAFFE_SPARSE_HYPERCOLUMN_LAYER_HPP

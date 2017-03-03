#ifndef CAFFE_SOFTMAX_KLD_LOSSLAYER_HPP_
#define CAFFE_SOFTMAX_KLD_LOSSLAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"

namespace caffe {

/**
 * \ingroup ttic
 *
 * @brief Computes a loss @f$L = -\frac{1}{N} \sum_{n=1}^N m_n \sum_{k}
 *        \big(\log p_{k,n} - \log q_{k,n}\big) q_{k,n}@f$
 *
 * Computes the KL-divergence or cross-entropy between a predicted
 * distribution @f$p_{k,n}@f$, computed as a softmax of the first
 * bottom, and target distribution @f$q_{k,n}@f$ (provided directly as
 * probability values) in the second bottom. An optional third bottom
 * can be provided as a mask. The first two bottoms must be the same
 * shape, and the third, if provided, should be the same shape as the
 * first two, but singleton in the softmax axis dimension. For
 * example, if the first two bottoms (@f$p_{k,n},q_{k,n}@f$) are of
 * size @f$N\times K\times H\times W@f$ each, with softmax_param{
 * axis: 1}, then the mask provided as the third bottom should be of
 * size @f$N \times 1 \times H \times W@f$.
 *
 * By default, the layer only computes the KL-divergence. To
 * output the cross entropy instead (by excluding the @f$-\log
 * q_{k,n}@f$ term above), please set type to CROSS_ENTROPY. This will very
 * slightly speed up the forward computation, but will not affect the backward
 * gradients in any way.
 *
 * Currently, the layer only back-propagates to its first bottom
 * (@f$p_{k,n}@f$). Also, you have to ensure the second bottom
 * constitutes a set of proper probability distributions, i.e.,
 * @f$\sum_k q_{k,n} = 1, \forall n@f$---the layer itself does not
 * enforce or verify this to avoid un-necessary computation.
 *
 * @author Ayan Chakrabarti
 */
template <typename Dtype>
class SoftmaxKLDLossLayer : public LossLayer<Dtype> {
 public:
  /**
   * @param param provides SoftmaxKLDLossParameter softmax_kld_loss_param,
   *    with options:
   *  - type. Choice between KL_DIVERGENCE and CROSS_ENTROPY.
   */
  explicit SoftmaxKLDLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SoftmaxKLDLoss"; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  virtual inline int ExactNumBottomBlobs() const { return -1; }
  virtual inline int MaxBottomBlobs() const { return 3; }
  virtual inline int MinBottomBlobs() const { return 2; }


 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  virtual Dtype get_normalizer(
      LossParameter_NormalizationMode normalization_mode, Dtype valid_count);

  shared_ptr<Layer<Dtype> > softmax_layer_;
  Blob<Dtype> prob_;
  vector<Blob<Dtype>*> softmax_bottom_vec_;
  vector<Blob<Dtype>*> softmax_top_vec_;
  LossParameter_NormalizationMode normalization_;

  int softmax_axis_, outer_num_, inner_num_, sma_num_;
  bool has_mask_, do_kl_;
};

}  // namespace caffe

#endif  // CAFFE_SOFTMAX_KLD_LOSSLAYER_HPP_

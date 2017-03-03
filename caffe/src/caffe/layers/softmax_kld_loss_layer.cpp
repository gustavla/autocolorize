#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/softmax_kld_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxKLDLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

  if (!this->layer_param_.loss_param().has_normalization() &&
      this->layer_param_.loss_param().has_normalize()) {
    normalization_ = this->layer_param_.loss_param().normalize() ?
                     LossParameter_NormalizationMode_VALID :
                     LossParameter_NormalizationMode_BATCH_SIZE;
  } else {
    normalization_ = this->layer_param_.loss_param().normalization();
  }

  do_kl_ = (this->layer_param_.softmax_kld_loss_param().type() ==
            SoftmaxKLDLossParameter_LossType_KL_DIVERGENCE);
}

template <typename Dtype>
void SoftmaxKLDLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  sma_num_ = bottom[0]->shape(softmax_axis_);

  CHECK_EQ(bottom[0]->count(), bottom[1]->count())
    << "First two bottoms must have the same size.";

  if(bottom.size() > 2) {
    has_mask_ = true;
    CHECK_EQ(outer_num_*inner_num_, bottom[2]->count())
      << "Mask is the wrong shape.";
  } else {
    has_mask_ = false;
  }

}

template <typename Dtype>
Dtype SoftmaxKLDLossLayer<Dtype>::get_normalizer(
    LossParameter_NormalizationMode normalization_mode, Dtype valid_count) {
  Dtype normalizer;
  switch (normalization_mode) {
    case LossParameter_NormalizationMode_FULL:
      normalizer = Dtype(outer_num_ * inner_num_);
      break;
    case LossParameter_NormalizationMode_VALID:
      if (valid_count < 0) {
        normalizer = Dtype(outer_num_ * inner_num_);
      } else {
        normalizer = valid_count;
      }
      break;
    case LossParameter_NormalizationMode_BATCH_SIZE:
      normalizer = Dtype(outer_num_);
      break;
    case LossParameter_NormalizationMode_NONE:
      normalizer = Dtype(1);
      break;
    default:
      LOG(FATAL) << "Unknown normalization mode: "
          << LossParameter_NormalizationMode_Name(normalization_mode);
  }
  return std::max(Dtype(1.0), normalizer);
}

template <typename Dtype>
void SMKLD_mskmul_cpu(
    const int outer, const int k, const int inner,
    Dtype* x, const Dtype* msk) {

  for(int i = 0; i < outer; ++i) {
    for(int j = 0; j < k; ++j) {
      caffe_mul(inner, x, msk, x);
      x += inner;
    }
    msk += inner;
  }

}

template <typename Dtype>
void SoftmaxKLDLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);

  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  Dtype * temp = bottom[0]->mutable_cpu_diff();

  int N = prob_.count();
  Dtype loss = 0, count = -1;

  // Prevent underflow in log
  for(int i = 0; i < N; ++i)
    temp[i] = std::max(prob_data[i], Dtype(FLT_MIN));
  caffe_log(N, temp, temp);

  // Subtract entropy of target distribution
  if(do_kl_) {
    for(int i = 0; i < N; ++i)
      temp[i] -= log(std::max(label[i], Dtype(FLT_MIN)));
  }

  if(has_mask_) {
    SMKLD_mskmul_cpu(outer_num_, sma_num_, inner_num_,
                     temp, bottom[2]->cpu_data());
    if(normalization_ == LossParameter_NormalizationMode_VALID)
      count = caffe_cpu_asum(bottom[2]->count(), bottom[2]->cpu_data());
  }

  loss = caffe_cpu_dot(N, temp, label);
  top[0]->mutable_cpu_data()[0] = -loss / get_normalizer(normalization_, count);
}

template <typename Dtype>
void SoftmaxKLDLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to target distribution yet.";
  }
  if(has_mask_) {
    if (propagate_down[2]) {
      LOG(FATAL) << this->type()
                 << " Layer cannot backpropagate to mask.";
    }
  }

  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    const Dtype* label = bottom[1]->cpu_data();
    int N = prob_.count();
    Dtype count = -1;

    caffe_sub(N, prob_data, label, bottom_diff);
    if(has_mask_) {
      SMKLD_mskmul_cpu(outer_num_, sma_num_, inner_num_,
                       bottom_diff, bottom[2]->cpu_data());

      if(normalization_ == LossParameter_NormalizationMode_VALID)
        count = caffe_cpu_asum(bottom[2]->count(), bottom[2]->cpu_data());
    }

    Dtype loss_weight = top[0]->cpu_diff()[0] /
      get_normalizer(normalization_, count);
    caffe_scal(N, loss_weight, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(SoftmaxKLDLossLayer);
#endif

INSTANTIATE_CLASS(SoftmaxKLDLossLayer);
REGISTER_LAYER_CLASS(SoftmaxKLDLoss);

}  // namespace caffe

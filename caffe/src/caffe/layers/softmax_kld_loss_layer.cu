#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/softmax_kld_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SMKLD_fwd_gpu(const int N,
          const Dtype* p, const Dtype* q, const Dtype* msk, Dtype* loss,
          const int outer, const int k, const int inner,
          const bool has_mask_, const bool do_kl_) {
  CUDA_KERNEL_LOOP(index, N) {
    loss[index] = log(max(p[index],Dtype(FLT_MIN)));
    if(do_kl_)
      loss[index] -= log(max(q[index],Dtype(FLT_MIN)));

    if(has_mask_) {
      const int mi = (index/(inner*k))*inner + index % inner;
      loss[index] *= msk[mi];
    }

  }
}

template <typename Dtype>
void SoftmaxKLDLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);

  const Dtype* prob_data = prob_.gpu_data();
  const Dtype* label = bottom[1]->gpu_data();
  const Dtype * msk;
  Dtype* temp = bottom[0]->mutable_gpu_diff();

  int N = prob_.count();
  Dtype loss = 0, count = -1;

  if(has_mask_)
    msk = bottom[2]->gpu_data();

  // NOLINT_NEXT_LINE(whitespace/operators)
  SMKLD_fwd_gpu<Dtype><<<CAFFE_GET_BLOCKS(N),CAFFE_CUDA_NUM_THREADS>>>(
     N,prob_data,label,msk,temp,outer_num_,sma_num_,inner_num_,
     has_mask_,do_kl_);
  caffe_gpu_dot(N,temp,label,&loss);

  if(has_mask_ && normalization_ == LossParameter_NormalizationMode_VALID)
    caffe_gpu_asum(bottom[2]->count(),msk,&count);

  top[0]->mutable_cpu_data()[0] = -loss / get_normalizer(normalization_,count);
}

template <typename Dtype>
__global__ void SMKLD_mskmul_gpu(const int N, Dtype* loss, const Dtype* msk,
    const int outer, const int k, const int inner) {
  CUDA_KERNEL_LOOP(index, N) {
      const int mi = (index/(inner*k))*inner + index % inner;
      loss[index] *= msk[mi];
  }
}

template <typename Dtype>
void SoftmaxKLDLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
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
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* prob_data = prob_.gpu_data();
    const Dtype* label = bottom[1]->gpu_data();
    int N = prob_.count();
    Dtype count = -1;

    caffe_gpu_sub(N,prob_data,label,bottom_diff);

    if(has_mask_) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      SMKLD_mskmul_gpu<Dtype><<<CAFFE_GET_BLOCKS(N),
        CAFFE_CUDA_NUM_THREADS>>>(N,bottom_diff,bottom[2]->gpu_data(),
                                  outer_num_,sma_num_,inner_num_);

      if(normalization_ == LossParameter_NormalizationMode_VALID)
        caffe_gpu_asum(bottom[2]->count(),bottom[2]->gpu_data(),&count);
    }

    Dtype loss_weight = top[0]->cpu_diff()[0] /
      get_normalizer(normalization_, count);

    caffe_gpu_scal(N, loss_weight , bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SoftmaxKLDLossLayer);

}  // namespace caffe

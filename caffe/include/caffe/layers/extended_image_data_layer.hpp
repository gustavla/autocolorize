#ifndef CAFFE_IMAGE_SEG_DATA_LAYER_HPP_
#define CAFFE_IMAGE_SEG_DATA_LAYER_HPP_

#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"

namespace caffe {

template <typename Dtype>
class ExtendedImageDataTransformer : public DataTransformer<Dtype> {
 public:
  void InitRand();
  double RandUniform();
  double RandNormal();
  int cur_epoch_;
  int cur_iter_;

  explicit ExtendedImageDataTransformer(const TransformationParameter& param, Phase phase,
                                   unsigned int seed);
  virtual ~ExtendedImageDataTransformer() {}

  virtual void Transform(const cv::Mat& cv_img, Blob<Dtype>* transformed_data_blob,
                         const Dtype fill_value, const bool ucm2, const bool color_perturbation);

 protected:
  unsigned int seed_;
};

/**
 * \ingroup ttic
 *
 * @author Gustav Larsson
 * @author Chenxi Liu
 * @author DeepLab
 */
template <typename Dtype>
class ExtendedImageDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit ExtendedImageDataLayer(const LayerParameter& param);
  virtual ~ExtendedImageDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char *type() const {
    return "ExtendedImageData";
  }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }
  virtual inline bool AutoTopBlobs() const { return true; }

 protected:
  void InitPrefetchRand();
  virtual void ShuffleImages();
  virtual void load_batch(Batch<Dtype>* batch);

 protected:
  Blob<Dtype> transformed_label_;

  shared_ptr<Caffe::RNG> prefetch_rng_;

  vector<std::pair<std::string, int> > lines_;

  int cur_epoch_;
  int random_epoch_;

  // We're adding a second transformer. This is a hack.
  shared_ptr<ExtendedImageDataTransformer<Dtype> > extended_image_data_transformer_;
};


}  // namespace caffe

#endif  // CAFFE_IMAGE_SEG_DATA_LAYER_HPP_

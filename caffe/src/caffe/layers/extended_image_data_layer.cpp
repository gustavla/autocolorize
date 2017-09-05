#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include <boost/random/uniform_real.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/variate_generator.hpp>

#include <matio.h>
#include <gason.h>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/extended_image_data_layer.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

extern "C" {
#include <maskApi.h>
}

namespace caffe {

// Can probably be made inferable from the files
const static int LABELS = 80;

cv::Mat ReadImageToCVMatPreserveAspectRatio(const string& filename,
    const int shortest_side, const bool is_color) {
  cv::Mat cv_img;
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat cv_img_origin = cv::imread(filename, cv_read_flag);
  if (!cv_img_origin.data) {
    LOG(ERROR) << "Could not open or find file " << filename;
    return cv_img_origin;
  }
  cv::Size s = cv_img_origin.size();
  int height, width;
  if (s.height < s.width) {
    height = shortest_side;
    width = (int)(shortest_side * (double)s.width / s.height);
  } else {
    width = shortest_side;
    height = (int)(shortest_side * (double)s.height / s.width);
  }
  cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
  return cv_img;
}

template <typename Dtype>
cv::Mat _load_mat(const string& filename, const string& key, int image_type,
                  int new_height, int new_width, int shortest_side, bool is_color) {
  if (image_type == ExtendedImageDataParameter_ImageType_PIXEL) {
    cv::Mat cv_img;
    if (shortest_side > 0) {
      cv_img = ReadImageToCVMatPreserveAspectRatio(filename,
            shortest_side, is_color);
    } else {
      cv_img = ReadImageToCVMat(filename,
            new_height, new_width, is_color);
    }
    return cv_img;
  }
  else if (image_type == ExtendedImageDataParameter_ImageType_MAT) {
    CHECK_EQ(new_height, 0) << "new_height not supported if image type is MAT";
    CHECK_EQ(new_width, 0) << "new_width not supported if image type is MAT";
    int img_row, img_col;
    mat_t *matfp;
    matfp = Mat_Open(filename.c_str(), MAT_ACC_RDONLY);
    CHECK(matfp) << "Error opening MAT file " << filename;
    matvar_t *matvar;
    matvar = Mat_VarReadInfo(matfp, key.c_str());
    CHECK(matvar) << "Field '" << key << "' not present in MAT file " << filename;
    CHECK_EQ(matvar->rank, 2) << "Rank must be 2 in MAT file " << filename;
    // Note that intuitively img_row should be index 0, but Matlab is column-major
    img_row = matvar->dims[0];
    img_col = matvar->dims[1];

    // TODO: Should this be CV_64F if Dtype is double?
    cv::Mat cv_img = cv::Mat::zeros(img_row, img_col, CV_32F);
    if (matvar->class_type == MAT_C_DOUBLE) {
      double* data = new double[img_row * img_col];
      int ret = Mat_VarReadDataLinear(matfp, matvar, data, 0, 1, img_row * img_col);
      CHECK(ret == 0) << "Error reading array '" << key << "' from MAT file " << filename;
      Mat_VarFree(matvar);

      for (int i = 0; i < img_col; i++) {
        for (int j = 0; j < img_row; j++) {
          cv_img.at<Dtype>(j, i) = static_cast<Dtype>(data[i * img_row + j]);
        }
      }
      delete[] data;
    } else if (matvar->class_type == MAT_C_SINGLE) {
      Dtype* data = new Dtype[img_row * img_col];
      int ret = Mat_VarReadDataLinear(matfp, matvar, data, 0, 1, img_row * img_col);
      CHECK(ret == 0) << "Error reading array '" << key << "' from MAT file " << filename;
      Mat_VarFree(matvar);

      for (int i = 0; i < img_col; i++) {
        for (int j = 0; j < img_row; j++) {
          cv_img.at<Dtype>(j, i) = data[i * img_row + j];
        }
      }
      delete[] data;
    } else if (matvar->class_type == MAT_C_UINT16) {
      unsigned short* data = new unsigned short[img_row * img_col];
      int ret = Mat_VarReadDataLinear(matfp, matvar, data, 0, 1, img_row * img_col);
      CHECK(ret == 0) << "Error reading array '" << key << "' from MAT file " << filename;
      Mat_VarFree(matvar);

      for (int i = 0; i < img_col; i++) {
        for (int j = 0; j < img_row; j++) {
          cv_img.at<Dtype>(j, i) = static_cast<Dtype>(data[i * img_row + j]);
        }
      }
      delete[] data;
    } else {
      CHECK(0)
        << "Field '" << key << "' is of unsupported type in MAT file " << filename;
    }
    Mat_Close(matfp);
    return cv_img;
  }
  else if (image_type == ExtendedImageDataParameter_ImageType_COCO_JSON) {
    CHECK_EQ(new_height, 0) << "new_height not supported if image type is COCO_JSON";
    CHECK_EQ(new_width, 0) << "new_width not supported if image type is COCO_JSON";
    int img_row, img_col;
    std::ifstream f_json(filename);
    if (!f_json.is_open()) {
      DLOG(INFO) << "Unable to open file " << filename;
    }
    f_json.seekg(0, std::ios::end);
    int f_len = f_json.tellg();
    f_json.seekg(0, std::ios::beg);
    char *source = new char[f_len];
    f_json.read(source, f_len);
    f_json.close();

    // Parse JSON file
    char *endptr;
    JsonValue val;
    JsonAllocator allocator;
    int status = jsonParse(source, &endptr, &val, allocator);
    if (status != JSON_OK) {
      fprintf(stderr, "%s at %zd\n", jsonStrError(status), endptr - source);
      exit(EXIT_FAILURE);
    }

    for (auto i:val) {
      if (strcmp(i->key, "height") == 0) {
        img_row = i->value.toNumber();
      }
      else if (strcmp(i->key, "width") == 0) {
        img_col = i->value.toNumber();
      }
    }

    cv::Mat *cv_imgs = new cv::Mat[LABELS];
    int k = 0;
    byte* M = new byte[img_row*img_col];
    for (auto v:val) {
      if (strcmp(v->key, "height") != 0 && strcmp(v->key, "width") != 0 && strcmp(v->key, "c0") != 0) {
        if (v->value.getTag() == JSON_STRING) {
          char* c = v->value.toString();
          RLE *R = 0;
          rlesInit(&R, 1);
          rleFrString(R, c, img_row, img_col);

          if (R->m > 1) {
            cv_imgs[k] = cv::Mat::zeros(img_col, img_row, CV_8UC1);
            rleDecode(R, cv_imgs[k].data, 1);
            cv_imgs[k] = cv_imgs[k].t();
          } else {
            cv_imgs[k] = cv::Mat::zeros(img_row, img_col, CV_8UC1);
          }
          k++;
          if (k >= LABELS) {
            break;
          }
        }
      }
    }
    delete[] M;
    cv::Mat cv_img;
    cv::merge(cv_imgs, LABELS, cv_img);
    delete[] cv_imgs;
    return cv_img;
  }
  else {
    cv::Mat cv_img(new_height, new_width,
        CV_8UC1, cv::Scalar(0));
    return cv_img;
  }
}

template<typename Dtype>
ExtendedImageDataTransformer<Dtype>::ExtendedImageDataTransformer(const TransformationParameter& param,
    Phase phase, unsigned int seed)
    : DataTransformer<Dtype>(param, phase), seed_(seed) {
}

template <typename Dtype>
void ExtendedImageDataTransformer<Dtype>::InitRand() {
  const bool needs_rand = this->param_.mirror() ||
      (this->phase_ == TRAIN && this->param_.crop_size());
  if (needs_rand) {
    unsigned int rng_seed;
    if (this->seed_ == -1) {
      rng_seed = caffe_rng_rand();
    } else {
      rng_seed = (unsigned int)this->seed_;
    }
    this->rng_.reset(new Caffe::RNG(rng_seed));
  } else {
    this->rng_.reset();
  }
}

template <typename Dtype>
void ExtendedImageDataLayer<Dtype>::InitPrefetchRand() {
  const unsigned int random_seed = this->layer_param_.extended_image_data_param().random_seed();
  unsigned int rng_seed;
  if (random_seed == -1) {
    rng_seed = caffe_rng_rand();
  } else {
    rng_seed = (unsigned int)random_seed + 1231 * this->cur_epoch_;
  }
  this->prefetch_rng_.reset(new Caffe::RNG(rng_seed));
}


template <typename Dtype>
double ExtendedImageDataTransformer<Dtype>::RandUniform() {
  CHECK(this->rng_);
  boost::uniform_real<double> random_distribution(0, 1);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(this->rng_->generator());
  boost::variate_generator<caffe::rng_t*, boost::uniform_real<double> >
      variate_generator(rng, random_distribution);
  return variate_generator();
}


template <typename Dtype>
double ExtendedImageDataTransformer<Dtype>::RandNormal() {
  Dtype u1, u2;
  CHECK(this->rng_);
  boost::normal_distribution<double> random_distribution(0, 1);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(this->rng_->generator());
  boost::variate_generator<caffe::rng_t*, boost::normal_distribution<double> >
      variate_generator(rng, random_distribution);
  return variate_generator();
}


template<typename Dtype>
void ExtendedImageDataTransformer<Dtype>::Transform(const cv::Mat& cv_img,
  Blob<Dtype>* transformed_data_blob, const Dtype fill_value, const bool ucm2,
  const bool color_perturbation) {
  const int img_channels = cv_img.channels();
  // height and width may change due to pad for cropping
  int img_height   = cv_img.rows;
  int img_width    = cv_img.cols;

  const int data_height   = transformed_data_blob->height();
  const int data_width    = transformed_data_blob->width();

  const int raw_crop_size = this->param_.crop_size();
  const Dtype scale = this->param_.scale();
  const bool do_mirror = this->param_.mirror() && this->Rand(2);
  const bool has_mean_file = this->param_.has_mean_file();
  const bool has_mean_values = this->mean_values_.size() > 0;
  int crop_size = raw_crop_size;
  if (ucm2) {
    crop_size = 2 * raw_crop_size + 1;
  }

  CHECK_GT(img_channels, 0);

  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(img_channels, this->data_mean_.channels());
    CHECK_EQ(img_height, this->data_mean_.height());
    CHECK_EQ(img_width, this->data_mean_.width());
    mean = this->data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(this->mean_values_.size() == 1 || this->mean_values_.size() == img_channels) <<
     "Specify either 1 mean_value or as many as channels: " << img_channels;
    if (img_channels > 1 && this->mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < img_channels; ++c) {
        this->mean_values_.push_back(this->mean_values_[0]);
      }
    }
  }

  int h_off = 0;
  int w_off = 0;
  cv::Mat cv_cropped_img = cv_img;

  // transform to double, since we will pad mean pixel values
  cv_cropped_img.convertTo(cv_cropped_img, CV_64F);

  // Check if we need to pad img to fit for crop_size
  // copymakeborder
  int pad_height = std::max(crop_size - img_height, 0);
  int pad_width  = std::max(crop_size - img_width, 0);
  if (pad_height > 0 || pad_width > 0) {
    if (false && ucm2) {
      /*
      int pad_height0 = std::min(1, pad_height);
      int pad_width0 = std::min(1, pad_width);

      int pad_height1 = std::min(1, pad_height0);
      int pad_width1 = std::min(1, pad_width0);

      int pad_height2 = pad_height - pad_height1;
      int pad_width2 = pad_width - pad_width1;

      // If UCM2, we have to pad with a single line of 1, and then outside that 0
      cv::copyMakeBorder(cv_cropped_img, cv_cropped_img, 0, pad_height0,
            0, pad_width0, cv::BORDER_CONSTANT,
            cv::Scalar(0.0));

      if (pad_height1 > 0 || pad_width1 > 0) {
        cv::copyMakeBorder(cv_cropped_img, cv_cropped_img, 0, pad_height1,
              0, pad_width1, cv::BORDER_CONSTANT,
              cv::Scalar(1.0));
      }

      if (pad_height2 > 0 || pad_width2 > 0) {
        cv::copyMakeBorder(cv_cropped_img, cv_cropped_img, 0, pad_height2,
              0, pad_width2, cv::BORDER_CONSTANT,
              cv::Scalar(-0.1));
      }
      */

    } else if (this->mean_values_.size() == 3) {
      cv::copyMakeBorder(cv_cropped_img, cv_cropped_img, 0, pad_height,
            0, pad_width, cv::BORDER_CONSTANT,
            cv::Scalar(this->mean_values_[0], this->mean_values_[1], this->mean_values_[2]));
    } else {
      cv::copyMakeBorder(cv_cropped_img, cv_cropped_img, 0, pad_height,
            0, pad_width, cv::BORDER_CONSTANT,
            cv::Scalar(fill_value));
    }

    // update height/width
    img_height   = cv_cropped_img.rows;
    img_width    = cv_cropped_img.cols;
  }

  // crop img/seg
  if (raw_crop_size) {
    CHECK_EQ(crop_size, data_height);
    CHECK_EQ(crop_size, data_width);
    // We only do random crop when we do training.
    double h_unit = 0.5, w_unit = 0.5;
    if (this->phase_ == caffe::TRAIN) {
      // We draw floats, so that if it is used on two different sources images
      // of different scales, it should still pick the same windows.
      h_unit = this->RandUniform();
      w_unit = this->RandUniform();
    }

    if (ucm2) {
      h_off = 2 * int(h_unit * ((img_height - 1) / 2.0 - raw_crop_size));
      w_off = 2 * int(w_unit * ((img_width - 1) / 2.0 - raw_crop_size));
    } else {
      h_off = int(h_unit * (img_height - crop_size));
      w_off = int(w_unit * (img_width - crop_size));
    }
    cv::Rect roi(w_off, h_off, crop_size, crop_size);
    cv_cropped_img = cv_cropped_img(roi);
  }

  CHECK(cv_cropped_img.data);

  Dtype* transformed_data  = transformed_data_blob->mutable_cpu_data();

  int top_index;
  const double* data_ptr;

  for (int h = 0; h < data_height; ++h) {
    data_ptr = cv_cropped_img.ptr<double>(h);
    int data_index = 0;
    for (int w = 0; w < data_width; ++w) {
      // for image
      for (int c = 0; c < img_channels; ++c) {
        if (do_mirror) {
          top_index = (c * data_height + h) * data_width + (data_width - 1 - w);
        } else {
          top_index = (c * data_height + h) * data_width + w;
        }
        Dtype pixel = static_cast<Dtype>(data_ptr[data_index++]);
        if (has_mean_file) {
          int mean_index = (c * img_height + h_off + h) * img_width + w_off + w;
          transformed_data[top_index] =
            (pixel - mean[mean_index]) * scale;
        } else {
          if (has_mean_values) {
            transformed_data[top_index] =
              (pixel - this->mean_values_[c]) * scale;
          } else {
            transformed_data[top_index] = pixel * scale;
          }
        }
      }
    }
  }

  Dtype alphas[3];
  Dtype delta_color[3];
  if (color_perturbation) {
    alphas[0] = this->RandNormal() * 0.1;
    alphas[1] = this->RandNormal() * 0.1;
    alphas[2] = this->RandNormal() * 0.1;

    delta_color[0] = -0.58845609 * 3.31411285 * alphas[0] + 0.68336212 * 0.26577539 * alphas[1] + 0.43212920 * 0.06736519 * alphas[2];
    delta_color[1] = -0.57755263 * 3.31411285 * alphas[0] + 0.01874956 * 0.26577539 * alphas[1] - 0.81613811 * 0.06736519 * alphas[2];
    delta_color[2] = -0.56582010 * 3.31411285 * alphas[0] - 0.72983880 * 0.26577539 * alphas[1] + 0.38364429 * 0.06736519 * alphas[2];

    for (int c = 0; c < img_channels; ++c) {
      for (int s = 0; s < data_height * data_width; ++s) {
        top_index = c * data_height * data_width + s;
        transformed_data[top_index] += delta_color[c];
      }
    }
  }
}


template <typename Dtype>
ExtendedImageDataLayer<Dtype>::ExtendedImageDataLayer(const LayerParameter& param)
    : BasePrefetchingDataLayer<Dtype>(param), random_epoch_(-1) {
  const unsigned int random_seed = this->layer_param_.extended_image_data_param().random_seed();
  this->extended_image_data_transformer_.reset(
    new ExtendedImageDataTransformer<Dtype>(this->transform_param_, this->phase_, random_seed));
  //this->extended_image_data_transformer_->InitRand(0);
}

template <typename Dtype>
ExtendedImageDataLayer<Dtype>::~ExtendedImageDataLayer<Dtype>() {
  this->StopInternalThread();
}


template <typename Dtype>
void ExtendedImageDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.image_data_param().new_height();
  const int new_width  = this->layer_param_.image_data_param().new_width();
  const bool is_color  = this->layer_param_.image_data_param().is_color();
  const int image_type = this->layer_param_.extended_image_data_param().image_type();
  const bool ucm2 = this->layer_param_.extended_image_data_param().ucm2();
  const bool color_perturbation = this->layer_param_.extended_image_data_param().color_perturbation();
  const string key = this->layer_param_.extended_image_data_param().key();
  //const unsigned int random_seed = this->layer_param_.extended_image_data_param().random_seed();
  string root_folder = this->layer_param_.image_data_param().root_folder();

  const int shortest_side = this->layer_param_.extended_image_data_param().shortest_side();
  const int shortest_side_max = this->layer_param_.extended_image_data_param().shortest_side_max();

  bool with_label = top.size() == 2;

  TransformationParameter transform_param = this->layer_param_.transform_param();
  CHECK(transform_param.has_mean_file() == false) <<
         "ImageSegDataLayer does not support mean file";
  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";

  if (shortest_side > 0) {
    CHECK((new_height == 0 && new_width == 0)) << "If shortest_side is specified, "
        "new_height and new_width must be set to 0";
  }

  // Read the file with filenames and labels
  const string& source = this->layer_param_.image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());

  string linestr;
  int label = 0;
  while (std::getline(infile, linestr)) {
    std::istringstream iss(linestr);
    string imgfn;
    iss >> imgfn;
    if (with_label) {
      iss >> label;
    }

    lines_.push_back(std::make_pair(imgfn, label));
  }

  /*
  if (this->layer_param_.image_data_param().shuffle()) {
    // randomly shuffle data
    unsigned int prefetch_rng_seed;
    if (random_seed == -1) {
      prefetch_rng_seed = caffe_rng_rand();
    } else {
      prefetch_rng_seed = (unsigned int)random_seed;
    }
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  */

  /*
  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.image_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.image_data_param().rand_skip();
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  */

  string filename = root_folder + lines_[0].first;
  cv::Mat cv_img = _load_mat<Dtype>(filename, key, image_type, new_height, new_width, shortest_side, is_color);

  const int raw_crop_size = this->layer_param_.transform_param().crop_size();
  int crop_size = raw_crop_size;
  if (ucm2) {
    crop_size = 2 * raw_crop_size + 1;
  }

  const int channels = cv_img.channels();
  const int height = cv_img.rows;
  const int width = cv_img.cols;

  vector<int> top_shape;
  top_shape.push_back(1);
  top_shape.push_back(channels);

  const int batch_size = this->layer_param_.image_data_param().batch_size();
  vector<int> label_shape(1, batch_size);

  if (crop_size > 0) {
    top_shape.push_back(crop_size);
    top_shape.push_back(crop_size);
  } else {
    top_shape.push_back(height);
    top_shape.push_back(width);
  }

  // Reshape prefetch_data and top[0] according to the batch_size.
  CHECK_GT(batch_size, 0) << "Positive batch size required";

  this->transformed_data_.Reshape(top_shape);
  this->transformed_label_.Reshape(label_shape);

  // data
  top_shape[0] = batch_size;
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
    << top[0]->channels() << "," << top[0]->height() << ","
    << top[0]->width();
  // label
  if (with_label) {
    top[1]->Reshape(label_shape);
  }
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i].label_.Reshape(label_shape);
  }
}

template <typename Dtype>
void ExtendedImageDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(this->prefetch_rng_->generator());
  sort(lines_.begin(), lines_.end());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}


// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void ExtendedImageDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());

  Dtype* top_data = batch->data_.mutable_cpu_data();
  Dtype* top_label = batch->label_.mutable_cpu_data();

  ImageDataParameter image_data_param    = this->layer_param_.image_data_param();
  const int batch_size = image_data_param.batch_size();
  const int new_height = image_data_param.new_height();
  const int new_width  = image_data_param.new_width();
  const int image_type = this->layer_param_.extended_image_data_param().image_type();
  const Dtype fill_value = static_cast<Dtype>(this->layer_param_.extended_image_data_param().fill_value());
  const string key = this->layer_param_.extended_image_data_param().key();
  const bool ucm2 = this->layer_param_.extended_image_data_param().ucm2();
  const bool color_perturbation = this->layer_param_.extended_image_data_param().color_perturbation();
  const bool is_color  = image_data_param.is_color();
  string root_folder   = image_data_param.root_folder();

  const int shortest_side = this->layer_param_.extended_image_data_param().shortest_side();
  const int shortest_side_max = this->layer_param_.extended_image_data_param().shortest_side_max();

  const int lines_size = lines_.size();

  int epoch_size = (lines_size / batch_size);

  if (this->cur_epoch_ != this->random_epoch_) {
    this->random_epoch_ = this->cur_epoch_;
    this->InitPrefetchRand();
    if (this->layer_param_.image_data_param().shuffle()) {
      ShuffleImages();
    }
  }

  this->extended_image_data_transformer_->InitRand();

  int lines_id_ = 0;

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    //CHECK_GT(lines_size, lines_id_);
    //
    int shortest_side_i = shortest_side;

    // Randomly assign scale
    if (shortest_side > 0 && shortest_side_max > 0) {
      double scale_rand = this->extended_image_data_transformer_->RandUniform();
      shortest_side_i = shortest_side + (int)((shortest_side_max - shortest_side) * scale_rand);
    }

    string filename = root_folder + lines_[lines_id_ % lines_size].first;
    cv::Mat cv_img = _load_mat<Dtype>(filename, key, image_type, new_height, new_width, shortest_side_i, is_color);
    CHECK(cv_img.data) << "Could not load " << (root_folder + lines_[lines_id_ % lines_size].first);

    if (!cv_img.data) {
      DLOG(INFO) << "Fail to load img: " << root_folder + lines_[lines_id_ % lines_size].first;
    }

    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the image
    int offset;

    offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(top_data + offset);

    this->extended_image_data_transformer_->Transform(cv_img, &(this->transformed_data_),
        fill_value, ucm2, color_perturbation);
    trans_time += timer.MicroSeconds();

    top_label[item_id] = lines_[lines_id_].second;

    lines_id_++;
    /*
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.image_data_param().shuffle()) {
        ShuffleImages();
      }
    }
    */
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(ExtendedImageDataLayer);
REGISTER_LAYER_CLASS(ExtendedImageData);

}  // namespace caffe

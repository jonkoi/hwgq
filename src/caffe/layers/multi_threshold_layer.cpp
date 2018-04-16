#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/multi_threshold_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include <iostream>

namespace caffe {

template <typename Dtype>
void MultiThresholdLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  m_thres = this->layer_param_.multi_threshold_param().num_thres();
  m_channels = this->layer_param_.multi_threshold_param().num_channels();
  m_useByteOutput = this->layer_param_.multi_threshold_param().use_byte_output();

  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(1);
    // Initialize the threshold parameters blob
    vector<int> thres_shape(2);
    thres_shape[0] = m_thres;
    thres_shape[1] = m_channels;
    this->blobs_[0].reset(new Blob<Dtype>(thres_shape));

  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void MultiThresholdLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* thres_data = this->blobs_[0]->cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  vector<int> dshape = bottom[0]->shape();
  const int batches = dshape[0];
  const int channels = dshape[1];
  const int imgsize = dshape.size() == 2 ? 1 : bottom[0]->count(2);


  if(m_useByteOutput) {
    // treat the top blob as uint8_t
    uint8_t * top_data = (uint8_t*) top[0]->mutable_cpu_data();
    for(size_t b = 0; b < batches; b++) {
      for(size_t c = 0; c < channels; c++) {
        for(size_t i = 0; i < imgsize; i++) {
          uint8_t acc = 0;
          for(size_t t = 0; t < m_thres; t++) {
            size_t thres_ind = t * m_channels + (c % m_channels);
            acc += (bottom_data[i] >= thres_data[thres_ind]) ? 1 : 0;
          }
          top_data[i] = acc;
        }
        bottom_data += imgsize;
        top_data += imgsize;
      }
    }
  } else {
    // treat top blob as Dtype
    Dtype* top_data = top[0]->mutable_cpu_data();
    for(size_t b = 0; b < batches; b++) {
      for(size_t c = 0; c < channels; c++) {
        for(size_t i = 0; i < imgsize; i++) {
          Dtype acc = 0;
          for(size_t t = 0; t < m_thres; t++) {
            size_t thres_ind = t * m_channels + (c % m_channels);
            acc += (bottom_data[i] >= thres_data[thres_ind]) ? 1 : 0;
          }
          top_data[i] = acc;
        }
        bottom_data += imgsize;
        top_data += imgsize;
      }
    }
  }
}

template <typename Dtype>
void MultiThresholdLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
      NOT_IMPLEMENTED;
}

INSTANTIATE_CLASS(MultiThresholdLayer);
REGISTER_LAYER_CLASS(MultiThreshold);

}  // namespace caffe

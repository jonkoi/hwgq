#ifndef CAFFE_MULTI_THRESHOLD_LAYER_HPP_
#define CAFFE_MULTI_THRESHOLD_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

template <typename Dtype>
class MultiThresholdLayer : public NeuronLayer<Dtype> {
 public:
  explicit MultiThresholdLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MultiThreshold"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  unsigned int m_thres;
  unsigned int m_channels;
  bool m_useByteOutput;  // treat output blob as bytes instead of floats

};

}  // namespace caffe

#endif  // CAFFE_MULTI_THRESHOLD_LAYER_HPP_

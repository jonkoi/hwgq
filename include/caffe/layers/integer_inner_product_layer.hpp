#ifndef CAFFE_INT_INNER_PRODUCT_LAYER_HPP_
#define CAFFE_INT_INNER_PRODUCT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "gemm-bitserial.h"

namespace caffe {

template <typename Dtype>
class IntegerInnerProductLayer : public Layer<Dtype> {
 public:
  explicit IntegerInnerProductLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "IntegerInnerProduct"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int m_outputs;
  int m_inputs;
  int m_depth;

  BitSerialMatrix m_weights_;
  BitSerialMatrix m_acts_;
  bool m_weights_ready;

};

}  // namespace caffe

#endif  // CAFFE_INT_INNER_PRODUCT_LAYER_HPP_

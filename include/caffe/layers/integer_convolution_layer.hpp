#ifndef CAFFE_INT_CONVOLUTION_LAYER_HPP_
#define CAFFE_INT_CONVOLUTION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "gemm-bitserial.h"

namespace caffe {

template <typename Dtype>
class IntegerConvolutionLayer : public Layer<Dtype> {
 public:
  explicit IntegerConvolutionLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "IntegerConvolution"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int m_ifm, m_ofm, m_k, m_stride, m_pad, m_indim, m_outdim;
  int m_depth;

  BitSerialMatrix m_weights;
  BitSerialMatrix m_acts;
  bool m_weights_ready;
  Blob<Dtype> col_buffer_;

};

}  // namespace caffe

#endif  // CAFFE_INT_CONVOLUTION_LAYER_HPP_

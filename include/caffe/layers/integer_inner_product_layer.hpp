#ifndef CAFFE_INT_INNER_PRODUCT_LAYER_HPP_
#define CAFFE_INT_INNER_PRODUCT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "gemmbitserial.hpp"
#include "caffe/util/gemmlowp-quantization.hpp"

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

  gemmbitserial::GEMMContext m_gemmctx;
  bool m_weights_ready;

  bool m_useByteInput;  // treat input blob as bytes instead of floats
  bool m_usebitserial;

  std::vector<std::uint8_t> gemmlowp_weights;
  std::vector<std::uint8_t> gemmlowp_acts;
  std::vector<std::int32_t> gemmlowp_res;
  gemmlowp::GemmContext gemm_context;

};

}  // namespace caffe

#endif  // CAFFE_INT_INNER_PRODUCT_LAYER_HPP_

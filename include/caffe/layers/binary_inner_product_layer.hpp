#ifndef CAFFE_BINARY_INNER_PRODUCT_LAYER_HPP_
#define CAFFE_BINARY_INNER_PRODUCT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "gemmlowp.h"

// A structure to hold quantization parameters 'scale' and 'zero_point'
// as discussed in doc/quantization.md. As explained there, the meaning
// of these values is as the constants in the quantization equation
//
//   real_value = scale * (quantized_value - zero_point)
//
// In other words, 'zero_point' is the quantized value that corresponds
// to the real value 0, and 'scale' is the difference of real values
// corresponding to consecutive quantized values.
struct QuantizationParams {
  float scale;
  std::uint8_t zero_point;
};

namespace caffe {

template <typename Dtype>
class BinaryInnerProductLayer : public Layer<Dtype> {
 public:
  explicit BinaryInnerProductLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "BinaryInnerProduct"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int M_;
  int K_;
  int N_;
  bool bias_term_;
  Blob<Dtype> bias_multiplier_;
  bool transpose_;  ///< if true, assume transposed weights
  
  //parameters for binarization
  Blob<Dtype> binary_weights_;
  Blob<Dtype> alphas_;
  Blob<Dtype> filter_means_;
  Blob<Dtype> weight_sum_multiplier_;
  //gemmlowp params
  QuantizationParams lhs_qparams, rhs_qparams, result_qparams;
  int lhs_offset, rhs_offset, result_offset, right_shift;
  float real_multiplier;
  std::int32_t quantized_multiplier;
  gemmlowp::OutputStageQuantizeDownInt32ToUint8ScaleByFixedPoint
      quantize_down_stage;
  gemmlowp::OutputStageSaturatingCastToUint8 saturating_cast_stage;
  std::tuple<gemmlowp::OutputStageQuantizeDownInt32ToUint8ScaleByFixedPoint/*, gemmlowp::OutputStageSaturatingCastToUint8*/> output_pipeline;
  
  std::vector<std::uint8_t> gemmlowp_weights;
  std::vector<std::uint8_t> gemmlowp_acts;
  std::vector<std::int32_t> gemmlowp_res;
  gemmlowp::GemmContext gemm_context;
  bool weights_ready=false;
};

}  // namespace caffe

#endif  // CAFFE_BINARY_INNER_PRODUCT_LAYER_HPP_

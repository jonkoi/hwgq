#ifndef CAFFE_BINARY_CONV_LAYER_HPP_
#define CAFFE_BINARY_CONV_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/base_conv_layer.hpp"
#include "caffe/util/gemmlowp-quantization.hpp"

namespace caffe {

template <typename Dtype>
class BinaryConvolutionLayer : public BaseConvolutionLayer<Dtype> {
 public:

  explicit BinaryConvolutionLayer(const LayerParameter& param);

  virtual inline const char* type() const { return "BinaryConvolution"; }

 protected:
  virtual void forward_cpu_gemm(const Dtype* input, const Dtype* weights,
       Dtype* output, bool skip_im2col = false);
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual inline bool reverse_dimensions() { return false; }
  virtual void compute_output_shape();

  Blob<Dtype> binary_weights_;
  Blob<Dtype> alphas_;
  Blob<Dtype> weight_sum_multiplier_;

  //gemmlowp params
  gemmlowp::QuantizationParams lhs_qparams, rhs_qparams, result_qparams;
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

#endif  // CAFFE_BINARY_CONV_LAYER_HPP_

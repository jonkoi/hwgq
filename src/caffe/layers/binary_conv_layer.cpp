#include <vector>
#include <iostream>
#include "caffe/layers/binary_conv_layer.hpp"

namespace caffe {

template<typename Dtype>
BinaryConvolutionLayer<Dtype>::BinaryConvolutionLayer(const LayerParameter& param)
      : BaseConvolutionLayer<Dtype>(param) {
  weights_ready=false;
  BinaryConvolutionParameter bcp = this->layer_param_.binary_convolution_param();
  // gemmlowp initialization, if desired
  if(bcp.use_gemmlowp()) {
    float wmin = bcp.gemmlowp_wmin(), wmax = bcp.gemmlowp_wmax();
    float imin = bcp.gemmlowp_imin(), imax = bcp.gemmlowp_imax();
    float rmin = bcp.gemmlowp_rmin(), rmax = bcp.gemmlowp_rmax();
    lhs_qparams = gemmlowp::ChooseQuantizationParams(wmin, wmax);
    rhs_qparams = gemmlowp::ChooseQuantizationParams(imin, imax);
    result_qparams = gemmlowp::ChooseQuantizationParams(rmin, rmax);
    lhs_offset = -lhs_qparams.zero_point;
    rhs_offset = -rhs_qparams.zero_point;
    result_offset = result_qparams.zero_point;

    real_multiplier =
    lhs_qparams.scale * rhs_qparams.scale / result_qparams.scale;
    gemmlowp::QuantizeMultiplierSmallerThanOne(real_multiplier, &quantized_multiplier,
                                 &right_shift);

    quantize_down_stage.result_offset_after_shift = result_offset;
    quantize_down_stage.result_fixedpoint_multiplier = quantized_multiplier;
    quantize_down_stage.result_shift = right_shift;

    output_pipeline =
    std::make_tuple(quantize_down_stage);
  }
}

template <typename Dtype>
void BinaryConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void BinaryConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  BinaryConvolutionParameter binary_conv_param = this->layer_param_.binary_convolution_param();
  bool use_alpha = binary_conv_param.use_alpha();
  bool use_binarization = binary_conv_param.use_binarization();
  const Dtype pos_val = binary_conv_param.pos_val();
  const Dtype neg_val = binary_conv_param.neg_val();
  // initialization for binary parameters
  const Dtype* weight = this->blobs_[0]->cpu_data();
  const int weight_dim = this->blobs_[0]->count() / this->blobs_[0]->num();
  weight_sum_multiplier_.Reshape(weight_dim,1,1,1);
  binary_weights_.ReshapeLike(*this->blobs_[0]);
  alphas_.Reshape(this->num_output_,1,1,1);
  if(!weights_ready) {
    weights_ready = true;
    caffe_set(weight_sum_multiplier_.count(),Dtype(1),weight_sum_multiplier_.mutable_cpu_data());
    caffe_set(this->num_output_,Dtype(1),alphas_.mutable_cpu_data());
    caffe_copy(binary_weights_.count(),weight,binary_weights_.mutable_cpu_data());
    // binarize the weights
    if (use_binarization) {
      // compute alpha if needed
      if (use_alpha) {
        caffe_abs(this->num_output_*weight_dim,weight,binary_weights_.mutable_cpu_diff());
        const Dtype* abs_weight = binary_weights_.cpu_diff();
        caffe_cpu_gemv<Dtype>(CblasNoTrans, this->num_output_, weight_dim,
          1. / weight_dim, abs_weight, weight_sum_multiplier_.cpu_data(), 0.,
          alphas_.mutable_cpu_data());
        }
        for (int i = 0; i < this->num_output_; i++) {
          for (int j = 0; j < weight_dim; j++) {
            Dtype binary_code = (weight[i*weight_dim+j]>=0) ? pos_val:neg_val;
            binary_weights_.mutable_cpu_data()[i*weight_dim+j] = binary_code*alphas_.cpu_data()[i];
          }
        }
      }
      // gemmlowp
      const Dtype* binary_weights = binary_weights_.cpu_data();
      if(binary_conv_param.use_gemmlowp()) {
        gemmlowp_weights.resize(this->conv_out_channels_ * this->kernel_dim_);
        gemmlowp_acts.resize(this->kernel_dim_ * this->conv_out_spatial_dim_);
        gemmlowp_res.resize(this->conv_out_channels_ * this->conv_out_spatial_dim_);
        gemmlowp::Quantize(lhs_qparams, binary_weights, &gemmlowp_weights);
      }
  }

  const Dtype* binary_weights = binary_weights_.cpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, binary_weights,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Dtype>
void BinaryConvolutionLayer<Dtype>::forward_cpu_gemm(const Dtype* input, const Dtype* weights,
    Dtype* output, bool skip_im2col) {
  BinaryConvolutionParameter binary_conv_param = this->layer_param_.binary_convolution_param();
  const Dtype* col_buff = input;
  if (!this->is_1x1_) {
    if (!skip_im2col) {
      this->conv_im2col_cpu(input, this->col_buffer_.mutable_cpu_data());
    }
    col_buff = this->col_buffer_.cpu_data();
  }
  /*TODO support groups*/
  if(this->group_ > 1 && binary_conv_param.use_gemmlowp())
    throw "Grouped convs not yet supported with gemmlowp";
  for (int g = 0; g < this->group_; ++g) {
    if(binary_conv_param.use_gemmlowp()) {
      // gemmlowp works best with rm-cm-cm map orders, but im2col gives a
      // row-major rhs matrix. thus, we transpose the im2col result during the
      // quantization and switch the order of the lhs and rhs matrices.
      // TODO a better alternative would be to use im2row instead of im2col
      gemmlowp::QuantizeAndTranspose(rhs_qparams, col_buff, &gemmlowp_acts, this->conv_out_spatial_dim_, this->kernel_dim_);
      const gemmlowp::MatrixMap<const std::uint8_t, gemmlowp::MapOrder::RowMajor> rhs(gemmlowp_acts.data(), this->conv_out_spatial_dim_, this->kernel_dim_);
      const gemmlowp::MatrixMap<const std::uint8_t, gemmlowp::MapOrder::ColMajor> lhs(gemmlowp_weights.data(), this->kernel_dim_, this->conv_out_channels_);
      gemmlowp::MatrixMap<std::int32_t, gemmlowp::MapOrder::ColMajor> resmap(gemmlowp_res.data(), this->conv_out_spatial_dim_, this->conv_out_channels_);

      gemmlowp::GemmWithOutputPipeline<std::uint8_t, std::int32_t,
      gemmlowp::DefaultL8R8BitDepthParams>(
        &gemm_context, rhs, lhs,
        &resmap, rhs_offset, lhs_offset, output_pipeline);

      gemmlowp::Dequantize(result_qparams, gemmlowp_res, output);
    } else {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, this->conv_out_channels_ /
          this->group_, this->conv_out_spatial_dim_, this->kernel_dim_,
          (Dtype)1., weights + this->weight_offset_ * g, col_buff + this->col_offset_ * g,
          (Dtype)0., output + this->output_offset_ * g);
    }
  }
}

template <typename Dtype>
void BinaryConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  weights_ready = false;
  const Dtype* binary_weights = binary_weights_.cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, binary_weights,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(BinaryConvolutionLayer);
#endif

INSTANTIATE_CLASS(BinaryConvolutionLayer);

}  // namespace caffe

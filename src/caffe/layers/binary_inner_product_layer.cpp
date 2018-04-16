#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/binary_inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include <iostream>

namespace caffe {

template <typename Dtype>
void BinaryInnerProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_output = this->layer_param_.inner_product_param().num_output();
  bias_term_ = this->layer_param_.inner_product_param().bias_term();
  transpose_ = false; // no transpose is allowed at this point
  weights_ready=false;
  N_ = num_output;
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  K_ = bottom[0]->count(axis);
  BinaryInnerProductParameter bipp = this->layer_param_.binary_inner_product_param();
  // gemmlowp initialization, if desired
  if(bipp.use_gemmlowp()) {
    float wmin = bipp.gemmlowp_wmin(), wmax = bipp.gemmlowp_wmax();
    float imin = bipp.gemmlowp_imin(), imax = bipp.gemmlowp_imax();
    float rmin = bipp.gemmlowp_rmin(), rmax = bipp.gemmlowp_rmax();
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

  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize the weights
    vector<int> weight_shape(2);
    if (transpose_) {
      weight_shape[0] = K_;
      weight_shape[1] = N_;
    } else {
      weight_shape[0] = N_;
      weight_shape[1] = K_;
    }
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.inner_product_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, intiialize and fill the bias term
    if (bias_term_) {
      vector<int> bias_shape(1, N_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.inner_product_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
  if(bipp.use_gemmlowp()) {
    gemmlowp_weights.resize(N_*K_);
  }
}

template <typename Dtype>
void BinaryInnerProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  const int new_K = bottom[0]->count(axis);
  CHECK_EQ(K_, new_K)
      << "Input size incompatible with inner product parameters.";
  // The first "axis" dimensions are independent inner products; the total
  // number of these is M_, the product over these dimensions.
  M_ = bottom[0]->count(0, axis);
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(axis + 1);
  top_shape[axis] = N_;
  top[0]->Reshape(top_shape);
  // Set up the bias multiplier
  if (bias_term_) {
    vector<int> bias_shape(1, M_);
    bias_multiplier_.Reshape(bias_shape);
    caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
  BinaryInnerProductParameter bipp = this->layer_param_.binary_inner_product_param();
  // gemmlowp initialization, if desired
  if(bipp.use_gemmlowp()) {
    gemmlowp_res.resize(M_*N_);
    gemmlowp_acts.resize(K_*M_);
  }
}

template <typename Dtype>
void BinaryInnerProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int num_output = this->layer_param_.inner_product_param().num_output();
  BinaryInnerProductParameter binary_inner_product_param = this->layer_param_.binary_inner_product_param();
  bool use_alpha = binary_inner_product_param.use_alpha();
  bool use_binarization = binary_inner_product_param.use_binarization();
  const Dtype pos_val = binary_inner_product_param.pos_val();
  const Dtype neg_val = binary_inner_product_param.neg_val();
  // initialization for binary parameters
  const Dtype* weight = this->blobs_[0]->mutable_cpu_data();
  const int weight_dim = this->blobs_[0]->count() / this->blobs_[0]->num();
  weight_sum_multiplier_.Reshape(weight_dim,1,1,1);
  binary_weights_.ReshapeLike(*this->blobs_[0]);
  alphas_.Reshape(num_output,1,1,1);

  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* binary_weights = binary_weights_.cpu_data();

  if(!weights_ready) {
    weights_ready = true;
    caffe_set(weight_sum_multiplier_.count(),Dtype(1),weight_sum_multiplier_.mutable_cpu_data());
    caffe_set(num_output,Dtype(1),alphas_.mutable_cpu_data());
    caffe_copy(binary_weights_.count(),weight,binary_weights_.mutable_cpu_data());

    // binarize the weights
    if (use_binarization) {
      // compute alpha if needed
      if (use_alpha) {
        caffe_abs(binary_weights_.count(),weight,binary_weights_.mutable_cpu_diff());
        const Dtype* abs_weight = binary_weights_.cpu_diff();
        caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output, weight_dim,
            1. / weight_dim, abs_weight, weight_sum_multiplier_.cpu_data(), 0.,
            alphas_.mutable_cpu_data());
      }
      for (int i = 0; i < num_output; i++) {
        for (int j = 0; j < weight_dim; j++) {
          Dtype binary_code = (weight[i*weight_dim+j]>=0) ? pos_val:neg_val;
          binary_weights_.mutable_cpu_data()[i*weight_dim+j] = binary_code*alphas_.cpu_data()[i];
        }
      }
    }

    if(binary_inner_product_param.use_gemmlowp()) {
      gemmlowp::Quantize(lhs_qparams, binary_weights, &gemmlowp_weights);
    }
  }

  if(binary_inner_product_param.use_gemmlowp()) {
    gemmlowp::Quantize(rhs_qparams, bottom_data, &gemmlowp_acts);

    const gemmlowp::MatrixMap<const std::uint8_t, gemmlowp::MapOrder::RowMajor> lhs(gemmlowp_weights.data(), N_, K_);
    const gemmlowp::MatrixMap<const std::uint8_t, gemmlowp::MapOrder::ColMajor> rhs(gemmlowp_acts.data(), K_, M_);
    gemmlowp::MatrixMap<std::int32_t, gemmlowp::MapOrder::ColMajor> resmap(gemmlowp_res.data(), N_, M_);

    gemmlowp::GemmWithOutputPipeline<std::uint8_t, std::int32_t,
                           gemmlowp::DefaultL8R8BitDepthParams>(
    &gemm_context, lhs, rhs,
    &resmap, lhs_offset, rhs_offset, output_pipeline);


    gemmlowp::Dequantize(result_qparams, gemmlowp_res, top_data);
  } else {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, transpose_ ? CblasNoTrans : CblasTrans,
      M_, N_, K_, (Dtype)1.,
      bottom_data, binary_weights, (Dtype)0., top_data);
  }
  if (bias_term_) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
        bias_multiplier_.cpu_data(),
        this->blobs_[1]->cpu_data(), (Dtype)1., top_data);
  }
}

template <typename Dtype>
void BinaryInnerProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* binary_weights = binary_weights_.cpu_data();
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    // Gradient with respect to weight
    if (transpose_) {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          K_, N_, M_,
          (Dtype)1., bottom_data, top_diff,
          (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
    } else {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          N_, K_, M_,
          (Dtype)1., top_diff, bottom_data,
          (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
    }
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bias
    caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.cpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_cpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bottom data
    if (transpose_) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, binary_weights,
          (Dtype)0., bottom[0]->mutable_cpu_diff());
    } else {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, binary_weights,
          (Dtype)0., bottom[0]->mutable_cpu_diff());
    }
  }
  weights_ready=false;
}

#ifdef CPU_ONLY
STUB_GPU(BinaryInnerProductLayer);
#endif

INSTANTIATE_CLASS(BinaryInnerProductLayer);
REGISTER_LAYER_CLASS(BinaryInnerProduct);

}  // namespace caffe

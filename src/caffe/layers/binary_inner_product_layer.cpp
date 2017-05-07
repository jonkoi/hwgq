#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/binary_inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include <iostream>

// Given the min and max values of a float array, return
// reasonable quantization parameters to use for this array.
QuantizationParams ChooseQuantizationParams(float min, float max) {
  // We extend the [min, max] interval to ensure that it contains 0.
  // Otherwise, we would not meet the requirement that 0 be an exactly
  // representable value.
  min = std::min(min, 0.f);
  max = std::max(max, 0.f);

  // the min and max quantized values, as floating-point values
  const float qmin = 0;
  const float qmax = 255;

  // First determine the scale.
  const double scale = (max - min) / (qmax - qmin);

  // Zero-point computation.
  // First the initial floating-point computation. The zero-point can be
  // determined from solving an affine equation for any known pair
  // (real value, corresponding quantized value).
  // We know two such pairs: (rmin, qmin) and (rmax, qmax).
  // Let's use the first one here.
  const double initial_zero_point = qmin - min / scale;

  // Now we need to nudge the zero point to be an integer
  // (our zero points are integer, and this is motivated by the requirement
  // to be able to represent the real value "0" exactly as a quantized value,
  // which is required in multiple places, for example in Im2col with SAME
  // padding).
  std::uint8_t nudged_zero_point = 0;
  if (initial_zero_point < qmin) {
    nudged_zero_point = qmin;
  } else if (initial_zero_point > qmax) {
    nudged_zero_point = qmax;
  } else {
    nudged_zero_point =
        static_cast<std::uint8_t>(std::round(initial_zero_point));
  }

  QuantizationParams result;
  result.scale = scale;
  result.zero_point = nudged_zero_point;
  return result;
}


void Quantize(const QuantizationParams& qparams, const float* src,
              std::vector<std::uint8_t>* dst) {
  for (std::size_t i = 0; i < dst->size(); i++) {
    const float real_val = src[i];
    const float transformed_val = qparams.zero_point + real_val / qparams.scale;
    const float clamped_val = std::max(0.f, std::min(255.f, transformed_val));
    (*dst)[i] = static_cast<std::uint8_t>(std::round(clamped_val));
  }
}

void Quantize(const QuantizationParams& qparams, const double* src,
              std::vector<std::uint8_t>* dst) {
  for (std::size_t i = 0; i < dst->size(); i++) {
    const double real_val = src[i];
    const double transformed_val = qparams.zero_point + real_val / qparams.scale;
    const double clamped_val = std::max(0., std::min(255., transformed_val));
    (*dst)[i] = static_cast<std::uint8_t>(std::round(clamped_val));
  }
}

void Dequantize(const QuantizationParams& qparams,
                const std::vector<std::int32_t>& src, float* dst) {
  for (std::size_t i = 0; i < src.size(); i++) {
    const std::int32_t quantized_val = src[i];
    dst[i] = qparams.scale * (quantized_val - qparams.zero_point);
  }
}

void Dequantize(const QuantizationParams& qparams,
                const std::vector<std::int32_t>& src, double* dst) {
  for (std::size_t i = 0; i < src.size(); i++) {
    const std::int32_t quantized_val = src[i];
    dst[i] = qparams.scale * (quantized_val - qparams.zero_point);
  }
}

// Given a real_multiplier in the interval (0, 1),
// produces a pair (quantized_multiplier, right_shift) where
// quantized_multiplier is an int32 representing a fixed-point value
// in the interval [-1, 1)  (in practice we only produce positive values)
// and right_shift is an amount to shift right by, so that the
// floating-point multiplication of some int32 input value by real_multiplier,
//
//   return static_cast<int32>(int32_value * real_multiplier);
//
// is best approximated by the integer-arithmetic-only code
//
//   return RoundingRightShift(
//       FixedPointMultiplication(int32_value, quantized_multiplier),
//       right_shift);
//
// This is how to obtain the fixed-point multiplier and right shift
// parameters to pass to
// OutputStageQuantizeDownInt32ToUint8ScaleByFixedPoint.
//
// Note: all this code only needs to run offline to generate the quantized
// neural network workload, not at runtime on the
// device on which quantized neural networks need to run. So it's not
// performance-critical at all.
void QuantizeMultiplierSmallerThanOne(float real_multiplier,
                                      std::int32_t* quantized_multiplier,
                                      int* right_shift) {
  assert(real_multiplier > 0.f);
  assert(real_multiplier < 1.f);
  int s = 0;
  // We want to bring the real multiplier into the interval [1/2, 1).
  // We can do so by multiplying it by two, and recording how many times
  // we multiplied by two so that we can compensate that by a right
  // shift by the same amount.
  while (real_multiplier < 0.5f) {
    real_multiplier *= 2.0f;
    s++;
  }
  // Now that the real multiplier is in [1/2, 1), we convert it
  // into a fixed-point number.
  std::int64_t q =
      static_cast<std::int64_t>(std::round(real_multiplier * (1ll << 31)));
  assert(q <= (1ll << 31));
  // Handle the special case when the real multiplier was so close to 1
  // that its fixed-point approximation was undistinguishable from 1.
  // We handle this by dividing it by two, and remembering to decrement
  // the right shift amount.
  if (q == (1ll << 31)) {
    q /= 2;
    s--;
  }
  assert(s >= 0);
  assert(q <= std::numeric_limits<std::int32_t>::max());
  *quantized_multiplier = static_cast<std::int32_t>(q);
  *right_shift = s;
}


// =================================================================================

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
  // gemmlowp initialization
  // TODO the min-max values are hardcoded for binary weights and 2-bit uniform HWGQ
  lhs_qparams = ChooseQuantizationParams(-1, +1);
  rhs_qparams = ChooseQuantizationParams(0, 1.614);
  result_qparams = ChooseQuantizationParams(/*-1.614*(K_/2), 1.614*(K_/2)*/-10,10);
  lhs_offset = -lhs_qparams.zero_point;
  rhs_offset = -rhs_qparams.zero_point;
  result_offset = result_qparams.zero_point;

  real_multiplier =
      lhs_qparams.scale * rhs_qparams.scale / result_qparams.scale;
  QuantizeMultiplierSmallerThanOne(real_multiplier, &quantized_multiplier,
                                   &right_shift);

  quantize_down_stage.result_offset_after_shift = result_offset;
  quantize_down_stage.result_fixedpoint_multiplier = quantized_multiplier;
  quantize_down_stage.result_shift = right_shift;
  
  output_pipeline =
      std::make_tuple(quantize_down_stage/*, saturating_cast_stage*/);
  
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
  gemmlowp_weights.resize(N_*K_);
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
  gemmlowp_res.resize(M_*N_);
  gemmlowp_resf.resize(M_*N_);
  gemmlowp_acts.resize(K_*M_);
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
    
    // gemmlowp
    Quantize(lhs_qparams, binary_weights, &gemmlowp_weights);
  }
  
    // =========================================================
  // gemmlowp

  
  // quantize activations
  Quantize(rhs_qparams, bottom_data, &gemmlowp_acts);
  
  const gemmlowp::MatrixMap<const std::uint8_t, gemmlowp::MapOrder::RowMajor> lhs(gemmlowp_weights.data(), N_, K_);
  const gemmlowp::MatrixMap<const std::uint8_t, gemmlowp::MapOrder::ColMajor> rhs(gemmlowp_acts.data(), K_, M_);
  gemmlowp::MatrixMap<std::int32_t, gemmlowp::MapOrder::ColMajor> resmap(gemmlowp_res.data(), N_, M_);
  
  gemmlowp::GemmWithOutputPipeline<std::uint8_t, std::int32_t,
                                   gemmlowp::DefaultL8R8BitDepthParams>(
      &gemm_context, lhs, rhs,
      &resmap, lhs_offset, rhs_offset, output_pipeline);


  Dequantize(result_qparams, gemmlowp_res, top_data);
  //Dequantize(result_qparams, gemmlowp_res, &gemmlowp_resf[0]);
  // =========================================================
    /*for(int J = 0; J < K_; J++) {
    std::cout << J << " bottom_data: " << bottom_data[J] << " got q " << (int)gemmlowp_acts[J] << std::endl;
  }*/
  
  
  /*caffe_cpu_gemm<Dtype>(CblasNoTrans, transpose_ ? CblasNoTrans : CblasTrans,
      M_, N_, K_, (Dtype)1.,
      bottom_data, binary_weights, (Dtype)0., top_data);*/
  /*if(N_==10)
  for(int J = 0; J < N_ * M_; J++) {
    std::cout << J << " golden: " << top_data[J] << " got q " << (int)gemmlowp_res[J] << " uq " << gemmlowp_resf[J] << std::endl;
  }*/ 
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

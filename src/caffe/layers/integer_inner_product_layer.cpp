#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/integer_inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include <iostream>

namespace caffe {

template <typename Dtype>
void IntegerInnerProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  m_weights_ready=false;
  const int num_output = this->layer_param_.integer_inner_product_param().num_output();
  m_outputs = num_output;
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.integer_inner_product_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  m_inputs = bottom[0]->count(axis);

  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(1);
    // Initialize the weight blob
    vector<int> weight_shape(2);
    weight_shape[0] = m_outputs;
    weight_shape[1] = m_inputs;
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));

  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void IntegerInnerProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.integer_inner_product_param().axis());
  const int new_inputs = bottom[0]->count(axis);
  CHECK_EQ(m_inputs, new_inputs)
      << "Input size incompatible with inner product parameters.";
  // The first "axis" dimensions are independent inner products; the total
  // number of these is M_, the product over these dimensions.
  m_depth = bottom[0]->count(0, axis);
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(axis + 1);
  top_shape[axis] = m_outputs;
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void IntegerInnerProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  IntegerInnerProductParameter iipp = this->layer_param_.inner_product_param();
  const unsigned int wbits = iipp.wbits();
  const unsigned int ibits = iipp.ibits();
  const bool wsigned = iipp.wsigned();
  const bool isigned = iipp.isigned();
  if(!m_weights_ready) {
    // TODO first usage, set up the bit serial matrix
  }

  // TODO turn input into bit serial form
  // TODO matrix matrix product
  // TODO cast back to float -- or templatize accumulator type?

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
void IntegerInnerProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
      NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(IntegerInnerProductLayer);
#endif

INSTANTIATE_CLASS(IntegerInnerProductLayer);
REGISTER_LAYER_CLASS(IntegerInnerProduct);

}  // namespace caffe

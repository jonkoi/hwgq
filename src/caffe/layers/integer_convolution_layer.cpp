#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/integer_convolution_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/im2col.hpp"
#include <iostream>

namespace caffe {

template <typename Dtype>
void IntegerConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  m_weights_ready=false;
  m_ofm = this->layer_param_.integer_convolution_param().num_output();
  // note that we assume equal w/h strides/pad/kernel dims
  m_k = this->layer_param_.integer_convolution_param().kernel_size();
  m_stride = this->layer_param_.integer_convolution_param().stride();
  m_pad = this->layer_param_.integer_convolution_param().pad();
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.integer_convolution_param().axis());
  m_ifm = bottom[0]->shape()[axis];
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(1);
    // Initialize the weight blob
    vector<int> weight_shape(2);
    weight_shape[0] = m_ofm;
    weight_shape[1] = m_ifm * m_k * m_k;
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));

  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void IntegerConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.integer_convolution_param().axis());
  const int new_ifm = bottom[0]->shape()[axis];
  CHECK_EQ(m_ifm, new_ifm)
      << "Input size incompatible with convolution parameters.";
  // The first "axis" dimensions are independent inner products; the total
  // number of these is M_, the product over these dimensions.
  m_depth = bottom[0]->count(0, axis);
  // TODO relax spatial dim assumptions
  m_indim = bottom[0]->shape()[axis+1];
  m_outdim = (m_indim + 2 * m_pad - m_k) / m_stride + 1;
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  vector<int> top_shape = bottom[0]->shape();
  top_shape[axis] = m_ofm;
  top_shape[axis+1] = m_outdim;
  top_shape[axis+2] = m_outdim;
  top[0]->Reshape(top_shape);
  vector<int> col_buffer_shape(2);
  col_buffer_shape[0] = m_outdim*m_outdim;
  col_buffer_shape[1] = m_ifm * m_k * m_k;
  col_buffer_.Reshape(col_buffer_shape);
}

template <typename Dtype>
void IntegerConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  IntegerConvolutionParameter icp = this->layer_param_.integer_convolution_param();
  const unsigned int wbits = icp.wbits();
  const unsigned int ibits = icp.ibits();
  const bool wsigned = icp.wsigned();
  const bool isigned = icp.isigned();
  if(!m_weights_ready) {
    // first usage, set up the bit serial matrix
    const Dtype* weight_buf = this->blobs_[0]->cpu_data();
    m_weights = toBitSerialMatrix(weight_buf, m_ofm, m_ifm * m_k * m_k, wbits);
    m_weights_ready = true;
  }
  for(int d = 0; d < m_depth; d++) {
    // TODO cater specifically for 1x1 case
    /// call im2col
    const Dtype * in_buff = bottom[0]->cpu_data() + m_ifm * m_indim * m_indim * d;
    Dtype * col_buff = col_buffer_.mutable_cpu_data();
    im2col_cpu(in_buff, m_ifm, m_indim, m_indim,
        m_k, m_k, m_pad, m_pad, m_stride, m_stride, 1, 1, col_buff);
    // turn transpose of patch matrix into bit serial form
    m_acts = toBitSerialMatrix_transpose(col_buff, m_outdim*m_outdim, m_ifm * m_k * m_k, ibits);
    // matrix matrix product
    AccumulateMatrix res = bitSerialMatrixMatrix(m_acts, m_weights, isigned, wsigned);
    // cast back to float -- or templatize accumulator type?
    Dtype* top_data = top[0]->mutable_cpu_data() + m_ofm * m_outdim * m_outdim * d;
    for(size_t c = 0; c < m_ofm; c++) {
      for(size_t r = 0; r < m_outdim * m_outdim; r++) {
        top_data[c * m_outdim * m_outdim + r] = (Dtype) res[c][r];
      }
    }
  }
}

template <typename Dtype>
void IntegerConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
      NOT_IMPLEMENTED;
}

INSTANTIATE_CLASS(IntegerConvolutionLayer);
REGISTER_LAYER_CLASS(IntegerConvolution);

}  // namespace caffe

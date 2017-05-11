#include <vector>
#include <chrono>
#include "caffe/filler.hpp"
#include "caffe/layers/integer_inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include <iostream>

//#define ENABLE_TIMERS

#ifndef ENABLE_TIMERS
#define TIMER_START ;
#define TIMER_END   ;
#define TIMER_GET(x) ;
#define TIMER_REPORT(x) ;
#else
#define TIMER_START start = std::chrono::high_resolution_clock::now();
#define TIMER_END   end = std::chrono::high_resolution_clock::now();
#define TIMER_GET(x) x = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
#define TIMER_REPORT(x) x;
#endif

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
  IntegerInnerProductParameter iipp = this->layer_param_.integer_inner_product_param();
  const unsigned int wbits = iipp.wbits();
  const unsigned int ibits = iipp.ibits();
  const bool wsigned = iipp.wsigned();
  const bool isigned = iipp.isigned();
  if(!m_weights_ready) {
    // first usage, set up the bit serial matrix
    const Dtype* weight_buf = this->blobs_[0]->cpu_data();
    m_weights = toBitSerialMatrix(weight_buf, m_outputs, m_inputs, wbits);
    m_weights_ready = true;
  }
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  TIMER_REPORT(
  auto TIMER_START;
  auto TIMER_END;
  double uscount_im2col=0;
  double uscount_quantin;
  double uscount_mm;
  double uscount_quantout;
  )
  // turn input into bit serial form
  // note that this is treated in transposed form
  TIMER_START;
  m_acts = toBitSerialMatrix(bottom_data, m_depth, m_inputs, ibits);
  TIMER_END
  TIMER_GET(uscount_quantin)

  // matrix matrix product
  TIMER_START;
  AccumulateMatrix res = bitSerialMatrixMatrix(m_weights, m_acts, wsigned, isigned);
  TIMER_END
  TIMER_GET(uscount_mm)

  // cast back to float -- or templatize accumulator type?
  // note that result is produced in transposed form
  TIMER_START;
  for(size_t c = 0; c < m_depth; c++) {
    for(size_t r = 0; r < m_outputs; r++) {
      top_data[c * m_outputs + r] = (Dtype) res[c][r];
    }
  }
  TIMER_END
  TIMER_GET(uscount_quantout)
  TIMER_REPORT(
    std::cout << "uscount_im2col uscount_quantin uscount_mm uscount_quantout" << std::endl;
    std::cout << uscount_im2col << " " << uscount_quantin << " " << uscount_mm << " " << uscount_quantout << std::endl;
  )
}

template <typename Dtype>
void IntegerInnerProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
      NOT_IMPLEMENTED;
}

INSTANTIATE_CLASS(IntegerInnerProductLayer);
REGISTER_LAYER_CLASS(IntegerInnerProduct);

}  // namespace caffe

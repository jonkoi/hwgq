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
  if(this->layer_param_.integer_inner_product_param().engine() == "bitserial") {
    m_usebitserial = true;
  } else if(this->layer_param_.integer_inner_product_param().engine() == "gemmlowp") {
    m_usebitserial = false;
  } else {
    // undefined engine
    m_usebitserial = true;
  }
  m_useByteInput = this->layer_param_.integer_inner_product_param().use_byte_input();
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
  const std::int32_t param_wsigned_offset = iipp.wsigned_offset();
  const std::int32_t param_isigned_offset = iipp.isigned_offset();
  if(!m_weights_ready) {
    const Dtype* weight_buf = this->blobs_[0]->cpu_data();
    if(m_usebitserial) {
      // first usage, set up the bit serial matrix
      m_gemmctx = gemmbitserial::allocGEMMContext(
        m_outputs, m_inputs, m_depth, wbits, ibits, wsigned, isigned
      );
      m_gemmctx.lhs.importRegular(weight_buf);
    } else {
      // set up for gemmlowp
      gemmlowp_weights.resize(m_outputs*m_inputs);
      gemmlowp_acts.resize(m_depth*m_inputs);
      gemmlowp_res.resize(m_depth*m_outputs);
      // copy weight matrix, adjusting to stay positive if signed
      uint8_t * gemmlowp_weights_ptr = gemmlowp_weights.data();
      Dtype weight_offs = wsigned ? param_wsigned_offset : 0;
      for(int i = 0; i < m_outputs*m_inputs; i++) {
        gemmlowp_weights_ptr[i] = (uint8_t)(weight_buf[i] + weight_offs);
      }
    }
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
  if(m_usebitserial) {
    if(m_useByteInput) {
      // treat input blob as uint8_t data
      m_gemmctx.rhs.importRegular((uint8_t *) bottom_data);
    } else {
      // TODO gemmbitserial importRegular should support const offsets
      m_gemmctx.rhs.importRegular(bottom_data);
    }
  } else {
    if(m_useByteInput) {
      memcpy(gemmlowp_acts.data(), (uint8_t *) bottom_data, m_depth*m_inputs);
    } else {
      // cast to uint8
      uint8_t * actptr = gemmlowp_acts.data();
      Dtype act_offs = isigned ? param_isigned_offset : 0;
      for(unsigned int i = 0; i < m_depth*m_inputs; i++) {
        actptr[i] = (std::uint8_t) (bottom_data[i] + act_offs);
      }

    }
  }
  TIMER_END
  TIMER_GET(uscount_quantin)

  // matrix matrix product
  TIMER_START;
  if(m_usebitserial) {
    gemmbitserial::gemmBitSerial(m_gemmctx);
  } else {
    // use gemmlowp
    const gemmlowp::MatrixMap<const std::uint8_t, gemmlowp::MapOrder::RowMajor> lhs(gemmlowp_weights.data(), m_outputs, m_inputs);
    const gemmlowp::MatrixMap<const std::uint8_t, gemmlowp::MapOrder::ColMajor> rhs(gemmlowp_acts.data(), m_inputs, m_depth);
    gemmlowp::MatrixMap<std::int32_t, gemmlowp::MapOrder::ColMajor> resmap(gemmlowp_res.data(), m_outputs, m_depth);
    std::tuple<> output_pipeline;
    int lhs_offset = wsigned ? -param_wsigned_offset : 0;
    int rhs_offset = isigned ? -param_isigned_offset : 0;
    gemmlowp::GemmWithOutputPipeline<std::uint8_t, std::int32_t,
    gemmlowp::DefaultL8R8BitDepthParams>(
      &gemm_context, lhs, rhs,
      &resmap, lhs_offset, rhs_offset, output_pipeline);
  }
  TIMER_END
  TIMER_GET(uscount_mm)

  // cast back to float -- or templatize accumulator type?
  // note that result is produced in transposed form
  TIMER_START;
  if(m_usebitserial) {
    for(size_t c = 0; c < m_depth; c++) {
      for(size_t r = 0; r < m_outputs; r++) {
        top_data[c * m_outputs + r] = (Dtype) m_gemmctx.res[c * m_outputs + r];
      }
    }
  } else {
    std::int32_t * gemmlowpres = gemmlowp_res.data();
    for(size_t c = 0; c < m_depth; c++) {
      for(size_t r = 0; r < m_outputs; r++) {
        top_data[c * m_outputs + r] = (Dtype) gemmlowpres[c * m_outputs + r];
      }
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

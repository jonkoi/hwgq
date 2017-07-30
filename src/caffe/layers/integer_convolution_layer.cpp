#include <vector>
#include <chrono>

#include "caffe/filler.hpp"
#include "caffe/layers/integer_convolution_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/im2col.hpp"
#include <iostream>

#define ENABLE_TIMERS

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
void IntegerConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  m_weights_ready=false;
  if(this->layer_param_.integer_convolution_param().engine() == "bitserial") {
    m_usebitserial = true;
  } else if(this->layer_param_.integer_convolution_param().engine() == "gemmlowp") {
    m_usebitserial = false;
  } else {
    // undefined engine
    m_usebitserial = true;
  }
  m_useByteInput = this->layer_param_.integer_convolution_param().use_byte_input();
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

// from DarkNet
template <typename Dtype>
inline Dtype im2row_get_pixel(const Dtype *im, const int height, const int width, const int channels,
                        const int row, const int col, const int channel, const int pad)
{
    const int prow = row - pad;
    const int pcol = col - pad;

    if (prow < 0 || pcol < 0 ||
        prow >= height || pcol >= width) return 0;
    return im[pcol + width*(prow + height*channel)];
}

//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
template <typename Dtype, typename DtypeOut>
void darknet_im2row_cpu(const Dtype* data_im,
     const int channels, const int height, const int width,
     const int ksize, const int stride, const int pad, DtypeOut* data_col)
{
    int c,h,w;
    const int height_col = (height + 2*pad - ksize) / stride + 1;
    const int width_col = (width + 2*pad - ksize) / stride + 1;
    const int k2 = ksize * ksize;

    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        const int w_offset = c % ksize;
        const int h_offset = (c / ksize) % ksize;
        const int c_im = c / k2;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                const int im_row = h_offset + h * stride;
                const int im_col = w_offset + w * stride;
                const int col_index = c + channels_col * (w + h * width_col);
                data_col[col_index] = (DtypeOut) im2row_get_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
            }
        }
    }
}

template <typename Dtype>
void IntegerConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  IntegerConvolutionParameter icp = this->layer_param_.integer_convolution_param();
  const unsigned int wbits = icp.wbits();
  const unsigned int ibits = icp.ibits();
  const bool wsigned = icp.wsigned();
  const bool isigned = icp.isigned();
  TIMER_REPORT(
  auto TIMER_START;
  auto TIMER_END;
  double uscount_im2col;
  double uscount_quantin;
  double uscount_mm;
  double uscount_quantout;
  )
  if(!m_weights_ready) {
    const Dtype* weight_buf = this->blobs_[0]->cpu_data();
    if(m_usebitserial) {
      // first usage, set up the bit serial matrix
      m_gemmctx = gemmbitserial::allocGEMMContext(
        m_outdim*m_outdim, m_ifm * m_k * m_k, m_ofm, ibits, wbits, isigned, wsigned
      );
      m_gemmctx.rhs.importRegular(weight_buf);
    } else {
      // set up for gemmlowp
      gemmlowp_weights.resize(m_ifm * m_k * m_k * m_ofm);
      gemmlowp_acts.resize(m_outdim*m_outdim * m_ifm * m_k * m_k);
      gemmlowp_res.resize(m_outdim*m_outdim*m_ofm);
      // copy weight matrix, adjusting to stay positive if signed
      uint8_t * gemmlowp_weights_ptr = gemmlowp_weights.data();
      Dtype weight_offs = wsigned ? 128 : 0;
      for(int i = 0; i < m_ifm * m_k * m_k * m_ofm; i++) {
        gemmlowp_weights_ptr[i] = (uint8_t)(weight_buf[i] + weight_offs);
      }
      m_weights_ready = true;
    }
  }
  for(int d = 0; d < m_depth; d++) {
    // TODO cater specifically for 1x1 case
    uint8_t * col_buff_u8 = (uint8_t*)(col_buffer_.mutable_cpu_data());
    TIMER_START
    if(m_useByteInput) {
      // the bottom blob actually contains uint8_t values -- interpret as such
      const uint8_t * in_buff_u8 = ((uint8_t*)(bottom[0]->cpu_data())) + m_ifm * m_indim * m_indim * d;
      if(m_usebitserial) {
        darknet_im2row_cpu(
          in_buff_u8, m_ifm, m_indim, m_indim, m_k, m_stride, m_pad, col_buff_u8
        );
      } else {
        // directly lower into gemmlowp activation buffer
        darknet_im2row_cpu(
          in_buff_u8, m_ifm, m_indim, m_indim, m_k, m_stride, m_pad, gemmlowp_acts.data()
        );
      }
    } else {
      // use regular float (or whatever Dtype is) im2col
      const Dtype * in_buff = bottom[0]->cpu_data() + m_ifm * m_indim * m_indim * d;
      Dtype * col_buff = col_buffer_.mutable_cpu_data();
      // darknet_im2row_cpu supports casting internally
      if(m_usebitserial) {
        darknet_im2row_cpu(
          in_buff, m_ifm, m_indim, m_indim, m_k, m_stride, m_pad, col_buff_u8
        );
      } else {
        // directly lower into gemmlowp activation buffer
        darknet_im2row_cpu(
          in_buff, m_ifm, m_indim, m_indim, m_k, m_stride, m_pad, gemmlowp_acts.data()
        );
      }
    }
    TIMER_END
    TIMER_GET(uscount_im2col)

    TIMER_START
    if(m_usebitserial) {
      m_gemmctx.lhs.importRegular(col_buff_u8, false);
    }
    TIMER_END
    TIMER_GET(uscount_quantin);

    // all data for convolution is now ready inside the gemm context
    // matrix matrix product
    TIMER_START
    if(m_usebitserial) {
      gemmbitserial::gemmBitSerial(m_gemmctx);
    } else {
      // use gemmlowp
      const gemmlowp::MatrixMap<const std::uint8_t, gemmlowp::MapOrder::RowMajor> lhs(gemmlowp_acts.data(), m_outdim * m_outdim, m_ifm * m_k * m_k);
      const gemmlowp::MatrixMap<const std::uint8_t, gemmlowp::MapOrder::ColMajor> rhs(gemmlowp_weights.data(), m_ifm * m_k * m_k, m_ofm);
      gemmlowp::MatrixMap<std::int32_t, gemmlowp::MapOrder::ColMajor> resmap(gemmlowp_res.data(), m_outdim * m_outdim, m_ofm);
      std::tuple<> output_pipeline;
      int lhs_offset = isigned ? -128 : 0;
      int rhs_offset = wsigned ? -128 : 0;
      gemmlowp::GemmWithOutputPipeline<std::uint8_t, std::int32_t,
      gemmlowp::DefaultL8R8BitDepthParams>(
        &gemm_context, lhs, rhs,
        &resmap, lhs_offset, rhs_offset, output_pipeline);
    }
    TIMER_END
    TIMER_GET(uscount_mm);
    // cast back to float -- or templatize accumulator type?
    TIMER_START
    Dtype* top_data = top[0]->mutable_cpu_data() + m_ofm * m_outdim * m_outdim * d;
    if(m_usebitserial) {
      for(size_t c = 0; c < m_ofm; c++) {
        for(size_t r = 0; r < m_outdim * m_outdim; r++) {
          top_data[c * m_outdim * m_outdim + r] = (Dtype) m_gemmctx.res[c * m_outdim * m_outdim + r];
        }
      }
    } else {
      std::int32_t * gemmlowpres = gemmlowp_res.data();
      for(size_t c = 0; c < m_ofm; c++) {
        for(size_t r = 0; r < m_outdim * m_outdim; r++) {
          top_data[c * m_outdim * m_outdim + r] = (Dtype) gemmlowpres[c * m_outdim * m_outdim + r];
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

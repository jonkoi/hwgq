#include "caffe/layers/mlbp_offload_layer.hpp"

#ifdef MLBP
// MLBP includes
#include "platform.hpp"
#endif

namespace caffe {

#define ind3D(sizeA, sizeB, sizeC, indA, indB, indC)  (indC + sizeC * (indB + sizeB * indA))

template <typename InType, typename OutType>
void interleaveChannels(const InType *in, OutType *out, unsigned int chans, unsigned int dim) {
  for(unsigned int c = 0; c < chans; c++) {
    for(unsigned int h = 0; h < dim; h++) {
      for(unsigned int w = 0; w < dim; w++) {
        out[ind3D(dim, dim, chans, h, w, c)] = (OutType) in[ind3D(chans, dim, dim, c, h, w)];
      }
    }
  }
}

template <typename InType, typename OutType>
void deinterleaveChannels(const InType *in, OutType *out, unsigned int chans, unsigned int dim) {
  for(unsigned int c = 0; c < chans; c++) {
    for(unsigned int h = 0; h < dim; h++) {
      for(unsigned int w = 0; w < dim; w++) {
        out[ind3D(chans, dim, dim, c, h, w)] = (OutType) in[ind3D(dim, dim, chans, h, w, c)];
      }
    }
  }
}

template <typename Dtype>
void MLBPOffloadLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top)
{
  // copy shapes into vector<int>s of 4D
  m_in_elems = 1;
  for(int i = 0; i < this->layer_param_.mlbp_offload_param().input_shape_size(); i++) {
    m_in_shape.push_back(this->layer_param_.mlbp_offload_param().input_shape(i));
    m_in_elems *= m_in_shape[i];
  }
  // TODO bring back 8-bit and float options when both are tested
  // force 64bit values per now
  //m_bytes_per_in = this->layer_param_.mlbp_offload_param().use_8bit_input() ? sizeof(char) : sizeof(float);
  m_bytes_per_in = sizeof(uint64_t);


  m_out_elems = 1;
  for(int i = 0; i < this->layer_param_.mlbp_offload_param().output_shape_size(); i++) {
    m_out_shape.push_back(this->layer_param_.mlbp_offload_param().output_shape(i));
    m_out_elems *= m_out_shape[i];
  }
  // TODO bring back 8-bit and float options when both are tested
  //m_bytes_per_out = this->layer_param_.mlbp_offload_param().use_8bit_output() ? sizeof(char) : sizeof(float);
  m_bytes_per_out = sizeof(uint64_t);

#ifdef MLBP
  // connect to the MLBP donut driver
  m_driver = initPlatform();
  // execute the FPGA bitfile load command
  m_driver->attach(this->layer_param_.mlbp_offload_param().bitfile_load_cmd().c_str());
  // set up accelerator buffers
  m_accel_in_buf = m_driver->allocAccelBuffer(m_in_elems * m_bytes_per_in);
  m_accel_out_buf = m_driver->allocAccelBuffer(m_out_elems * m_bytes_per_out);
  // disable weight loading mode (assume built-in weights)
  m_driver->writeJamRegAddr(0x28, 0);
  // set number of images to 1
  m_driver->writeJamRegAddr(0x54, 1);
  // set input and output accel buffer addresses
  m_driver->write64BitJamRegAddr(0x10, (AccelDblReg) m_accel_in_buf);
  m_driver->write64BitJamRegAddr(0x1c, (AccelDblReg) m_accel_out_buf);
#endif

  // TODO get rid of these buffers when 8-bit and float support is tested
  m_in_uint64_data = new uint64_t[m_in_elems];
  m_out_uint64_data = new uint64_t[m_out_elems];
}

template <typename Dtype>
MLBPOffloadLayer<Dtype>::~MLBPOffloadLayer() {
#ifdef MLBP
  m_driver->deallocAccelBuffer(m_accel_in_buf);
  m_driver->deallocAccelBuffer(m_accel_out_buf);
  m_driver->detach();
  deinitPlatform(m_driver);
#endif
  // TODO get rid of these buffers when 8-bit and float support is tested
  delete [] m_in_uint64_data;
  delete [] m_out_uint64_data;
}

template <typename Dtype>
void MLBPOffloadLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top)
{
  vector<int> inshape = bottom[0]->shape();
  CHECK_EQ(inshape.size(), m_in_shape.size());
  for(int i  = 0; i < inshape.size(); i++) {
    CHECK_EQ(inshape[i], m_in_shape[i]);
  }
  // reshape top blob to be in the expected shape
  top[0]->Reshape(m_out_shape);
}

// TODO add templated helper function for interleave and deinterleave

template <typename Dtype>
void MLBPOffloadLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
#ifdef MLBP
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  if(this->layer_param_.mlbp_offload_param().interleave_input()) {
    // do input interleaving if desired
    int in_chans = m_in_shape[1];
    int in_dim = m_in_shape[2];
    interleaveChannels(bottom_data, m_in_uint64_data, in_chans, in_dim);
  } else {
    // just cast input buffer from float to uint64_t
    for(unsigned int i = 0; i < m_in_elems; i++) {
      m_in_uint64_data[i] = (uint64_t) bottom_data[i];
    }
  }
  // copy input data into accel-side buffer
  m_driver->copyBufferHostToAccel(m_in_uint64_data, m_accel_in_buf, m_in_elems * m_bytes_per_in);
  // execute and wait for accelerator to complete
  m_driver->writeJamRegAddr(0x00, 1);
  while((m_driver->readJamRegAddr(0x00) & 0x2) == 0) {
    usleep(1);
  }
  // copy results back to host memory
  m_driver->copyBufferAccelToHost(m_accel_out_buf, m_out_uint64_data, m_out_elems * m_bytes_per_out);
  if(this->layer_param_.mlbp_offload_param().deinterleave_output()) {
    // do output deinterleaving if desired
    int out_chans = m_out_shape[1];
    int out_dim = m_out_shape[2];
    deinterleaveChannels(m_out_uint64_data, top_data, out_chans, out_dim);
  } else {
    // cast output buffer from float to uint64_t
    for(unsigned int i = 0; i < m_out_elems; i++) {
      top_data[i] = (Dtype) m_out_uint64_data[i];
    }
  }
#else
  NOT_IMPLEMENTED;
#endif
}

template <typename Dtype>
void MLBPOffloadLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

INSTANTIATE_CLASS(MLBPOffloadLayer);
REGISTER_LAYER_CLASS(MLBPOffload);

} // namespace caffe

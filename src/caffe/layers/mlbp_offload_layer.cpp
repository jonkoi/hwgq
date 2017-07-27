#include "caffe/layers/mlbp_offload_layer.hpp"
// MLBP includes
#include "platform.hpp"

namespace caffe {

template <typename Dtype>
void MLBPOffloadLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top)
{
  // copy shapes into vector<int>s of 4D
  m_in_shape.push_back(1);
  m_in_shape.push_back(this->layer_param_.mlbp_offload_param().input_shape(0));
  m_in_shape.push_back(this->layer_param_.mlbp_offload_param().input_shape(1));
  m_in_shape.push_back(this->layer_param_.mlbp_offload_param().input_shape(1));
  m_in_elems = m_in_shape[1]*m_in_shape[2]*m_in_shape[3];
  // TODO bring back 8-bit and float options when both are tested
  // force 64bit values per now
  //m_bytes_per_in = this->layer_param_.mlbp_offload_param().use_8bit_input() ? sizeof(char) : sizeof(float);
  m_bytes_per_in = sizeof(uint64_t);

  m_out_shape.push_back(1);
  m_out_shape.push_back(this->layer_param_.mlbp_offload_param().output_shape(0));
  m_out_shape.push_back(this->layer_param_.mlbp_offload_param().output_shape(1));
  m_out_shape.push_back(this->layer_param_.mlbp_offload_param().output_shape(1));
  m_out_elems = m_out_shape[1]*m_out_shape[2]*m_out_shape[3];
  // TODO bring back 8-bit and float options when both are tested
  //m_bytes_per_out = this->layer_param_.mlbp_offload_param().use_8bit_output() ? sizeof(char) : sizeof(float);
  m_bytes_per_out = sizeof(uint64_t);

  // connect to the MLBP donut driver
  m_driver = initPlatform();
  // execute the FPGA bitfile load command
  m_driver->attach(this->layer_param_.mlbp_offload_param().bitfile_load_cmd().c_str());
  // set up accelerator buffers
  m_accel_in_buf = m_driver->allocAccelBuffer(m_in_elems * m_bytes_per_in);
  m_accel_out_buf = m_driver->allocAccelBuffer(m_out_elems * m_bytes_per_out);
  // set number of images to 1
  driver->writeJamRegAddr(0x54, 1);
  // set input and output accel buffer addresses
  driver->write64BitJamRegAddr(0x10, (AccelDblReg) m_accel_in_buf);
  driver->write64BitJamRegAddr(0x1c, (AccelDblReg) m_accel_in_buf);

  // TODO get rid of these buffers when 8-bit and float support is tested
  m_in_uint64_data = new uint64_t[total_in_elems];
  m_out_uint64_data = new uint64_t[total_out_elems];
}

virtual MLBPOffloadLayer<Dtype>::~MLBPOffloadLayer() {
  m_driver->deallocAccelBuffer(m_accel_in_buf);
  m_driver->deallocAccelBuffer(m_accel_out_buf);
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
  CHECK_EQ(inshape, m_in_shape);
  // reshape top blob to be in the expected shape
  top[0]->Reshape(m_out_shape);
}

// TODO add templated helper function for interleave and deinterleave

template <typename Dtype>
void MLBPOffloadLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();

  // TODO do input interleaving if desired
  // cast input buffer from float to uint64_t
  for(unsigned int i = 0; i < m_in_elems; i++) {
    m_in_uint64_data[i] = (uint64_t) bottom_data[i];
  }
  // copy input data into accel-side buffer
  driver->copyBufferHostToAccel(m_in_uint64_data, m_accel_in_buf, m_in_elems * m_bytes_per_in);
  // execute and wait for accelerator to complete
  driver->writeJamRegAddr(0x00, 1);
  while((driver->readJamRegAddr(0x00) & 0x2) == 0) {
    usleep(1);
  }
  // copy results back to host memory
  driver->copyBufferAccelToHost(m_accel_out_buf, m_out_uint64_data, m_out_elems * m_bytes_per_out);
  // cast output buffer from float to uint64_t
  for(unsigned int i = 0; i < total_out_elems; i++) {
    top_data[i] = (Dtype) m_out_uint64_data[i];
  }

  // TODO do output deinterleaving if desired
}

template <typename Dtype>
void MLBPOffloadLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

INSTANTIATE_CLASS(MLBPOffloadLayer);
REGISTER_LAYER_CLASS(MLBPOffload);

} // namespace caffe

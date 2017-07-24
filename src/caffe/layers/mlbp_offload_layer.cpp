#include "caffe/layers/mlbp_offload_layer.hpp"

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
  m_bytes_per_in = this->layer_param_.mlbp_offload_param().use_8bit_input() ? sizeof(char) : sizeof(float);

  m_out_shape.push_back(1);
  m_out_shape.push_back(this->layer_param_.mlbp_offload_param().output_shape(0));
  m_out_shape.push_back(this->layer_param_.mlbp_offload_param().output_shape(1));
  m_out_shape.push_back(this->layer_param_.mlbp_offload_param().output_shape(1));
  m_out_elems = m_out_shape[1]*m_out_shape[2]*m_out_shape[3];
  m_bytes_per_out = this->layer_param_.mlbp_offload_param().use_8bit_output() ? sizeof(char) : sizeof(float);

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
  driver->write64BitJamRegAddr(0x10, (AccelDblReg) accelbuf_in);
  driver->write64BitJamRegAddr(0x1c, (AccelDblReg) accelbuf_out);
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
  // TODO do input interleaving if desired
  // TODO copy into accelerator-side buffer
  // execute and wait for accelerator to complete
  driver->writeJamRegAddr(0x00, 1);
  while((driver->readJamRegAddr(0x00) & 0x2) == 0) {
    usleep(1);
  }
  // TODO copy results from accelerator-side buffer
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

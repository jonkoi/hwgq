#include "caffe/layers/mlbp_offload_layer.hpp"

namespace caffe {

template <typename Dtype>
void MLBPOffloadLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top)
{
  // TODO execute the FPGA bitfile load command
  // TODO set up accelerator buffers
}

template <typename Dtype>
void MLBPOffloadLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top)
{
  // TODO assert batch size equals one for now
  // TODO compare shape of bottom blob to expected shape
  /*CHECK_EQ(m_inputs, new_inputs)
      << "Input size incompatible with inner product parameters.";*/
  // TODO reshape top blob to be in the expected shape
  /*
  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(axis + 1);
  top_shape[axis] = m_outputs;
  top[0]->Reshape(top_shape);*/
}

// TODO add templated helper function for interleave and deinterleave

template <typename Dtype>
void MLBPOffloadLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  // TODO do input interleaving if desired
  // TODO copy into accelerator-side buffer
  // TODO execute and wait for accelerator to complete (time execution?)
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

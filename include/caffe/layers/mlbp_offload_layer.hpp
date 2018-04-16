#ifndef CAFFE_MLBP_OFFLOAD_LAYER_HPP_
#define CAFFE_MLBP_OFFLOAD_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#ifdef MLBP
// MLBP includes
#include "platform.hpp"
#endif

namespace caffe {

template <typename Dtype>
class MLBPOffloadLayer : public Layer<Dtype> {
 public:
  explicit MLBPOffloadLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual ~MLBPOffloadLayer();
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MLBPOffload"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  vector<int> m_in_shape, m_out_shape;
  size_t m_bytes_per_in, m_bytes_per_out;
  size_t m_in_elems, m_out_elems;
  void * m_accel_in_buf, * m_accel_out_buf;
  // TODO get rid of these once we have 8bit support:
  uint64_t * m_in_uint64_data, * m_out_uint64_data;

#ifdef MLBP
  DonutDriver * m_driver;
#endif

};

}  // namespace caffe

#endif  // CAFFE_MLBP_OFFLOAD_LAYER_HPP_

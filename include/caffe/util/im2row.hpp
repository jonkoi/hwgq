#ifndef _CAFFE_UTIL_IM2ROW_HPP_
#define _CAFFE_UTIL_IM2ROW_HPP_


namespace caffe {

template <typename Dtype>
void im2row_cpu(
          const Dtype* data,
          size_t depth,
          size_t height,
          size_t width,
          size_t windowHeight,
          size_t windowWidth,
          size_t pad_h,
          size_t pad_w,
          size_t strideY,
          size_t strideX,
          int dilateY,
          int dilateX,
          Dtype* stacked);
}

#endif

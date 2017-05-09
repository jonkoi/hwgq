// Adapted from Andrea Vedaldi's im2row code:
// https://github.com/vlfeat/matconvnet/blob/master/matlab/src/bits/impl/im2row_cpu.cpp


/*
Copyright (C) 2014-16 Andrea Vedaldi.
All rights reserved.
This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/


#include <string.h>

/* ---------------------------------------------------------------- */
/*                                                  Heper functions */
/* ---------------------------------------------------------------- */

static inline int floor_divide(int a, int b) {
  if (a >= 0) return a/b;
  else return (a - b + 1)/b;
}

static inline int ceil_divide(int a, int b) {
  if (a >= 0) return (a + b - 1)/b ;
  else return a/b ;
}

static inline int static_max(int a, int b) {
  return (a>=b) ? a:b ;
}

static inline int static_min(int a, int b) {
  return (a<=b) ? a:b ;
}


#include <vector>

#include "caffe/util/im2row.hpp"
#include "caffe/util/math_functions.hpp"

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
          Dtype* stacked)
  {
    size_t padLeft, padRight, padTop, padBottom;
    padLeft = padRight = pad_w;
    padTop = padBottom = pad_h;

    int windowExtentX = (windowWidth - 1)*dilateX + 1 ;
    int windowExtentY = (windowHeight - 1)*dilateY + 1 ;
    int numPatchesX = (width + (padLeft + padRight) - windowExtentX)/strideX + 1 ;
    int numPatchesY = (height + (padTop + padBottom) - windowExtentY)/strideY + 1 ;
    int numRows = windowWidth * windowHeight * depth ;

    /*
     Fill a row of the patch matrix. Since patches are stored
     along the columns of the matrix, scanning a row menas visiting all
     the patches. Different rows corresponds to a different
     offset within each patch.
     In this manner, as we fill a row
     we tend to access spatially adiacent elements
     in the input image, particulary for small strides.
     */
    for (int row = 0; row < numRows ; ++row) {
      /*
       Get the patch offset corresponding to this row of the stacked
       image.
       */
      int u = row ;
      int v = u / windowWidth ;
      int z = v / windowHeight ;
      u %= windowWidth ;
      v %= windowHeight ;

      /*
       Filling this row requires visiting the pixels in the input tensor
       `data` that appear at the given offset (u,v) in the output patches.
       For the patch at (x,y), the pixel coordinates (x_data,y_data) in the
       `data` tensor are:
       x_data(x) = x * strideX + u * dilateX - padLeft,  0 <= x < numPatchesX,
       y_data(y) = y * strideY + v * dilateY - padTop,   0 <= y < numPatchesY,
       z_data(z) = z.
       Now we visit all patches (x,y) in lexicographical order to fill
       successive output pixels. Patches around the boundary may peek outside
       the `data` tensor, which is padded with zero. We calcualte these
       borders here and fill them with zeros in the output.

       In particular, patch x peeks within the input tensor `data`
       if x is in the range [x0,x1] given by:
       x_data(x) >= 0
       <=> x >= (padLeft - u * dilateX) / stride
       <=> x >= ceil((padLeft - u * dilateX) / stride) = x0

       x_data(x) <= width-1
       <=> x <= (width-1 + padLeft - u * dilateX) / stride
       <=> x <= floor((width-1 + padLeft - u * dilateX) / stride)
       <=> x <  floor((width-1 + padLeft - u * dilateX) / stride) + 1 = x1
       and the same for y. Note that, while usually x0 <= x1, there are
       special cases for which x1 < x0. This is accounted for in the loops
       below.
       */

      int x0 = static_min(numPatchesX, ceil_divide(padLeft - u * dilateX, strideX)) ;
      int y0 = static_min(numPatchesY, ceil_divide(padTop - v * dilateY, strideY)) ;
      int x1 = static_min(numPatchesX, floor_divide(width-1 + padLeft - u * dilateX, strideX) + 1) ;
      int y1 = static_min(numPatchesY, floor_divide(height-1 + padTop - v * dilateY, strideY) + 1) ;
      int x ;
      int y ;

      for (y = 0 ; y < y0 ; ++y) {
        for (x = 0 ; x < numPatchesX ; ++x) {
          *stacked++ = 0 ;
        }
      }
      for ( ; y < y1 ; ++y) {
        for (x = 0 ; x < x0 ; ++x) {
          *stacked++ = 0 ;
        }
        int y_data = y * strideY + v * dilateY - padTop ;
        int x_data = x * strideX + u * dilateX - padLeft ;
        Dtype const * b = data + (z * height + y_data) * width + x_data ;
        for ( ; x < x1 ; ++x) {
          *stacked++ = *b ;
          b += strideX ;
        }
        for ( ; x < numPatchesX ; ++x) {
          *stacked++ = 0 ;
        }
      }
      for ( ; y < numPatchesY ; ++y) {
        for (x = 0 ; x < numPatchesX ; ++x) {
          *stacked++ = 0 ;
        }
      }
    }
  }

// Explicit instantiation
template void im2row_cpu<float>(const float* data, size_t depth, size_t height, size_t width, size_t windowHeight, size_t windowWidth, size_t pad_h, size_t pad_w, size_t strideY, size_t strideX, int dilateY, int dilateX, float* stacked);
template void im2row_cpu<double>(const double* data, size_t depth, size_t height, size_t width, size_t windowHeight, size_t windowWidth, size_t pad_h, size_t pad_w, size_t strideY, size_t strideX, int dilateY, int dilateX, double* stacked);
}

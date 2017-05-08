#pragma once
#include "gemmlowp.h"
#include <vector>

namespace gemmlowp {

// A structure to hold quantization parameters 'scale' and 'zero_point'
// as discussed in doc/quantization.md. As explained there, the meaning
// of these values is as the constants in the quantization equation
//
//   real_value = scale * (quantized_value - zero_point)
//
// In other words, 'zero_point' is the quantized value that corresponds
// to the real value 0, and 'scale' is the difference of real values
// corresponding to consecutive quantized values.
struct QuantizationParams {
  float scale;
  std::uint8_t zero_point;
};

// Given the min and max values of a float array, return
// reasonable quantization parameters to use for this array.
QuantizationParams ChooseQuantizationParams(float min, float max);

template <typename Dtype>
void Quantize(const QuantizationParams& qparams, const Dtype* src,
              std::vector<std::uint8_t>* dst) {
  for (std::size_t i = 0; i < dst->size(); i++) {
    const Dtype real_val = src[i];
    const Dtype transformed_val = qparams.zero_point + real_val / qparams.scale;
    const Dtype clamped_val = std::max((Dtype)0, std::min((Dtype)255, transformed_val));
    (*dst)[i] = static_cast<std::uint8_t>(std::round(clamped_val));
  }
}

template <typename Dtype>
void Dequantize(const QuantizationParams& qparams,
                const std::vector<std::int32_t>& src, Dtype* dst) {
  for (std::size_t i = 0; i < src.size(); i++) {
    const std::int32_t quantized_val = src[i];
    dst[i] = qparams.scale * (quantized_val - qparams.zero_point);
  }
}

void QuantizeMultiplierSmallerThanOne(float real_multiplier,
                                    std::int32_t* quantized_multiplier,
                                    int* right_shift);

} // namespace gemmlowp

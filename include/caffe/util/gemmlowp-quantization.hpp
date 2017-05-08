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

void Quantize(const QuantizationParams& qparams, const float* src,
              std::vector<std::uint8_t>* dst);

void Quantize(const QuantizationParams& qparams, const double* src,
              std::vector<std::uint8_t>* dst);

void Dequantize(const QuantizationParams& qparams,
              const std::vector<std::int32_t>& src, float* dst);


void Dequantize(const QuantizationParams& qparams,
              const std::vector<std::int32_t>& src, double* dst);


void QuantizeMultiplierSmallerThanOne(float real_multiplier,
                                    std::int32_t* quantized_multiplier,
                                    int* right_shift);

} // namespace gemmlowp

// In TensorUtils.h or within Tensor class
#pragma once
#include "Tensor.h"
#include <functional>

template<typename Function>
void apply_type_specific_operation(const Tensor& input, Tensor& output, Function op) {
    size_t total_elems = input.size();

    switch (input.dtype()) {
        case Dtype::Int32: {
            const int32_t* in_ptr = static_cast<const int32_t*>(input.data());
            int32_t* out_ptr = static_cast<int32_t*>(output.data());
          for (size_t i = 0; i < total_elems; i++) {
                out_ptr[i] = op(in_ptr[i]);
            }
            break;
        }
        case Dtype::Float32: {
            const float* in_ptr = static_cast<const float*>(input.data());
            float* out_ptr = static_cast<float*>(output.data());
            for (size_t i = 0; i < total_elems; i++) {
                out_ptr[i] = op(in_ptr[i]);
            }
            break;
        }
        default:
            throw std::runtime_error("Unsupported data type");
    }
}
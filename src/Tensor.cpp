#include <iostream>
#include <cstring>
#include <stdexcept>
#include "Tensor.h"
#include "TensorUtils.h"

// Functions of the class 
// Functions for the creation of Tensors

using namespace std;


Tensor::Tensor(Shape shape, Dtype dtype, Device device, bool requires_grad)
: shape_(shape), dtype_(dtype), device_(device), requires_grad_(requires_grad)
{
    // Calculate strides for row-major order
    stride_.strides.resize(shape.dims.size());
    if (shape.dims.size() > 0) {
        stride_.strides.back() = 1; // Last dimension stride is always 1
        for (int i = shape.dims.size() - 2; i >= 0; --i) {
            stride_.strides[i] = stride_.strides[i + 1] * shape.dims[i + 1];
        }
    
        // Calculate element size based on dtype
        size_t elem_size;
        switch (dtype) {
            case Dtype::Int32: elem_size = sizeof(int32_t); break;
            case Dtype::Float32: elem_size = sizeof(float); break;
            default: throw std::runtime_error("Unsupported data type");
        }
        
        size_t total_elems = 1;
        for (auto dim : shape.dims) total_elems *= dim;
        data_.resize(total_elems * elem_size);
        
        if (requires_grad_) {
            grad_.resize(total_elems * elem_size);
            std::memset(grad_.data(), 0, grad_.size());
        }
    } else {
        throw std::runtime_error("Shape must have at least one dimension");
    } 
}

Tensor Tensor::power(int exp){
    Tensor output(Shape{shape()}, dtype(), device(), requires_grad());
    apply_type_specific_operation(*this, output, [exp](auto x) -> auto{
        using T = decltype(x);

        if (exp == 0) return T(1);
        T result = 1, base = x;
        for (int i=0; i<exp; i++){
            result *= base;
        }
        return result;

    });
    return output;
}


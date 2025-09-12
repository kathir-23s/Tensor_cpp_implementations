#include "UnaryOps.h"
#include "TensorUtils.h"
#include "Tensor.h"
#include <cstdint>

Tensor sqr(const Tensor& input) {
    // Create output with same shape/dtype/device
    Tensor output(Shape{input.shape()}, input.dtype(), input.device(), input.requires_grad());
    apply_type_specific_operation(input, output, [](auto x){
        return x * x;
    });
    return output;
}

Tensor abs(const Tensor &input){
    Tensor output(Shape{input.shape()}, input.dtype(), input.device(), input.requires_grad());
    apply_type_specific_operation(input, output, [](auto x) {return x<0 ? -x : x; });
    return output;
}

Tensor Tensor::operator-() const {  // NO parameters!
    Tensor result(Shape{shape()}, dtype(), device(), requires_grad());
    apply_type_specific_operation(*this, result, [](auto x){ return -x; });
    return result;
}

Tensor pow(const Tensor& input, int exponent){
    Tensor output(Shape{input.shape()}, input.dtype(), input.device(), input.requires_grad());
    apply_type_specific_operation(input, output, [exponent](auto x) -> auto {
        for (int i=0; i<exponent; i++){
            x *= x;
        }
        return x;
    });
    return output;
}


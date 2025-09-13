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

Tensor operator-(const Tensor& input)  {  // NO parameters!
    Tensor result(Shape{input.shape()}, input.dtype(), input.device(), input.requires_grad());
    apply_type_specific_operation(input, result, [](auto x){ return -x; });
    return result;
}

Tensor pow(const Tensor& input, int exponent){
    Tensor output(Shape{input.shape()}, input.dtype(), input.device(), input.requires_grad());
    apply_type_specific_operation(input, output, [exponent](auto x) -> auto {
        using T = decltype(x); // Telling to make T as same type as x

        if (exponent == 0) return T(1);   // any number^0 = 1
        T result = 1;
        T base = x;
        for (int i = 0; i < exponent; i++) {
            result *= base;
        }
        return result;
    });
    return output;

}


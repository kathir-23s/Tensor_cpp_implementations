#pragma once
#include "Tensor.h"
#include <iostream>
#include <type_traits>  // Add this include
#include "TensorUtils.h"

//Declarations
template <typename T>
Tensor operator+(const Tensor& tensor, T scalar);

template <typename T>
Tensor operator+(T scalar, const Tensor& tensor);

template <typename T>
Tensor operator+(const Tensor& tensor, T scalar){
	Tensor result(Shape{tensor.shape()}, tensor.dtype(), tensor.device(), tensor.requires_grad());

	auto add_op = [scalar](auto x){ return x + static_cast<decltype(x)>(scalar);};
	apply_type_specific_operation(tensor, result, add_op);

	return result;
}

template<typename T>
Tensor operator+(T scalar, const Tensor& tensor){
	return tensor + scalar;
}

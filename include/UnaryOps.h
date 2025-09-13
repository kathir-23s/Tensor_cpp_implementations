#pragma once
#include "Tensor.h"


Tensor sqr(const Tensor& input);
Tensor abs(const Tensor& input);
Tensor operator-(const Tensor& input);
Tensor pow(const Tensor& input, int exponent);
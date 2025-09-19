#include <iostream>
#include <cstdint>
#include "Tensor.h"
#include "ScalarOps.h"
#include "TensorUtils.h"
// ScalarOps templates are defined in the header to allow instantiation in
// any translation unit that includes the header. This .cpp can remain empty
// or contain non-template helpers if needed.
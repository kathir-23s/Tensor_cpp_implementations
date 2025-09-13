#include "Tensor.h"
#include "UnaryOps.h"


#include <iostream>
#include <cassert>

void test_sqr_operation() {
    std::cout << "\n=== Testing sqr() operation ===\n";

    Tensor int_tensor(Shape{{2, 3}}, Dtype::Int32, Device::CPU, false);
    int32_t* int_data = static_cast<int32_t*>(int_tensor.data());
    int_data[0] = 2; int_data[1] = 3; int_data[2] = 4;
    int_data[3] = -1; int_data[4] = -5; int_data[5] = 0;

    std::cout << "\nOriginal Int32 tensor:\n";
    int_tensor.display(std::cout);

    Tensor int_result = sqr(int_tensor);
    std::cout << "\nSquared Int32 tensor:\n";
    int_result.display(std::cout);

    const int32_t* int_result_data = static_cast<const int32_t*>(int_result.data());
    assert(int_result_data[0] == 4);
    assert(int_result_data[1] == 9);
    assert(int_result_data[2] == 16);
    assert(int_result_data[3] == 1);
    assert(int_result_data[4] == 25);
    assert(int_result_data[5] == 0);

    Tensor float_tensor(Shape{{2, 2}}, Dtype::Float32, Device::CPU, false);
    float* float_data = static_cast<float*>(float_tensor.data());
    float_data[0] = 2.5f; float_data[1] = -1.5f;
    float_data[2] = 0.0f; float_data[3] = 3.0f;

    std::cout << "\nOriginal Float32 tensor:\n";
    float_tensor.display(std::cout);

    Tensor float_result = sqr(float_tensor);
    std::cout << "\nSquared Float32 tensor:\n";
    float_result.display(std::cout);

    const float* float_result_data = static_cast<const float*>(float_result.data());
    assert(float_result_data[0] == 6.25f);
    assert(float_result_data[1] == 2.25f);
    assert(float_result_data[2] == 0.0f);
    assert(float_result_data[3] == 9.0f);

    std::cout << "\n✓ sqr() test passed!\n\n";
}

void test_abs_operation() {
    std::cout << "\n=== Testing abs() operation ===\n";

    Tensor int_tensor(Shape{{2, 2}}, Dtype::Int32, Device::CPU, false);
    int32_t* int_data = static_cast<int32_t*>(int_tensor.data());
    int_data[0] = -7; int_data[1] = 4;
    int_data[2] = -3; int_data[3] = 0;

    std::cout << "\nOriginal Int32 tensor:\n";
    int_tensor.display(std::cout);

    Tensor int_abs = abs(int_tensor);
    std::cout << "\nAbs Int32 tensor:\n";
    int_abs.display(std::cout);

    const int32_t* int_abs_data = static_cast<const int32_t*>(int_abs.data());
    assert(int_abs_data[0] == 7);
    assert(int_abs_data[1] == 4);
    assert(int_abs_data[2] == 3);
    assert(int_abs_data[3] == 0);

    Tensor float_tensor(Shape{{1, 3}}, Dtype::Float32, Device::CPU, false);
    float* float_data = static_cast<float*>(float_tensor.data());
    float_data[0] = -2.5f; float_data[1] = 0.0f; float_data[2] = 3.7f;

    std::cout << "\nOriginal Float32 tensor:\n";
    float_tensor.display(std::cout);

    Tensor float_abs = abs(float_tensor);
    std::cout << "\nAbs Float32 tensor:\n";
    float_abs.display(std::cout);

    const float* float_abs_data = static_cast<const float*>(float_abs.data());
    assert(float_abs_data[0] == 2.5f);
    assert(float_abs_data[1] == 0.0f);
    assert(float_abs_data[2] == 3.7f);

    std::cout << "\n✓ abs() test passed!\n\n";
}

void test_neg_operation() {
    std::cout << "\n=== Testing unary - operation ===\n";

    Tensor int_tensor(Shape{{2, 2}}, Dtype::Int32, Device::CPU, false);
    int32_t* int_data = static_cast<int32_t*>(int_tensor.data());
    int_data[0] = 5; int_data[1] = -3;
    int_data[2] = 0; int_data[3] = 10;

    std::cout << "\nOriginal Int32 tensor:\n";
    int_tensor.display(std::cout);

    Tensor int_neg = -int_tensor;
    std::cout << "\nNegated Int32 tensor:\n";
    int_neg.display(std::cout);

    const int32_t* int_neg_data = static_cast<const int32_t*>(int_neg.data());
    assert(int_neg_data[0] == -5);
    assert(int_neg_data[1] == 3);
    assert(int_neg_data[2] == 0);
    assert(int_neg_data[3] == -10);

    Tensor float_tensor(Shape{{1, 3}}, Dtype::Float32, Device::CPU, false);
    float* float_data = static_cast<float*>(float_tensor.data());
    float_data[0] = 2.5f; float_data[1] = -1.5f; float_data[2] = 0.0f;

    std::cout << "\nOriginal Float32 tensor:\n";
    float_tensor.display(std::cout);

    Tensor float_neg = -float_tensor;
    std::cout << "\nNegated Float32 tensor:\n";
    float_neg.display(std::cout);

    const float* float_neg_data = static_cast<const float*>(float_neg.data());
    assert(float_neg_data[0] == -2.5f);
    assert(float_neg_data[1] == 1.5f);
    assert(float_neg_data[2] == 0.0f);

    std::cout << "\n✓ unary - test passed!\n\n";
}

void test_pow_operation() {
    std::cout << "\n=== Testing pow() operation ===\n";

    Tensor int_tensor(Shape{{1, 3}}, Dtype::Int32, Device::CPU, false);
    int32_t* int_data = static_cast<int32_t*>(int_tensor.data());
    int_data[0] = 2; int_data[1] = -3; int_data[2] = 4;

    std::cout << "\nOriginal Int32 tensor:\n";
    int_tensor.display(std::cout);

    Tensor int_pow = pow(int_tensor, 3); // cube
    std::cout << "\nInt32 tensor ^3:\n";
    int_pow.display(std::cout);

    const int32_t* int_pow_data = static_cast<const int32_t*>(int_pow.data());
    assert(int_pow_data[0] == 8);
    assert(int_pow_data[1] == -27);
    assert(int_pow_data[2] == 64);

    Tensor float_tensor(Shape{{2, 2}}, Dtype::Float32, Device::CPU, false);
    float* float_data = static_cast<float*>(float_tensor.data());
    float_data[0] = 1.5f; float_data[1] = -2.0f;
    float_data[2] = 0.0f; float_data[3] = 3.0f;

    std::cout << "\nOriginal Float32 tensor:\n";
    float_tensor.display(std::cout);

    Tensor float_pow = pow(float_tensor, 2); // square
    std::cout << "\nFloat32 tensor ^2:\n";
    float_pow.display(std::cout);

    const float* float_pow_data = static_cast<const float*>(float_pow.data());
    assert(float_pow_data[0] == 2.25f);
    assert(float_pow_data[1] == 4.0f);
    assert(float_pow_data[2] == 0.0f);
    assert(float_pow_data[3] == 9.0f);

    std::cout << "\n✓ pow() test passed!\n\n";
}

void test_tensor_metadata() {
    std::cout << "\n=== Testing tensor metadata preservation ===\n";

    Tensor original(Shape{{3, 2}}, Dtype::Float32, Device::CPU, true);

    Tensor squared = sqr(original);
    assert(squared.shape() == original.shape());
    assert(squared.dtype() == original.dtype());
    assert(squared.device() == original.device());
    assert(squared.requires_grad() == original.requires_grad());

    Tensor negated = -original;
    assert(negated.shape() == original.shape());
    assert(negated.dtype() == original.dtype());
    assert(negated.device() == original.device());
    assert(negated.requires_grad() == original.requires_grad());

    std::cout << "\n✓ Metadata preservation test passed!\n\n";
}

int main() {
    std::cout << "\nStarting Tensor Operations Test Suite...\n\n";

    try {
        test_sqr_operation();
        test_abs_operation();
        test_neg_operation();
        test_pow_operation();
        test_tensor_metadata();

        std::cout << "All tests passed! Tensor operations are working correctly.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with error: " << e.what() << "\n";
        return 1;
    }
}

#include "Tensor.h"
#include "UnaryOps.h"
#include <iostream>
#include <cassert>

void test_sqr_operation() {
    std::cout << "Testing sqr() operation...\n";
    
    // Test with Int32
    Tensor int_tensor(Shape{{2, 3}}, Dtype::Int32, Device::CPU, false);
    int32_t* int_data = static_cast<int32_t*>(int_tensor.data());
    int_data[0] = 2; int_data[1] = 3; int_data[2] = 4;
    int_data[3] = -1; int_data[4] = -5; int_data[5] = 0;
    
    Tensor int_result = sqr(int_tensor);
    const int32_t* int_result_data = static_cast<const int32_t*>(int_result.data());
    
    assert(int_result_data[0] == 4);   // 2² = 4
    assert(int_result_data[1] == 9);   // 3² = 9
    assert(int_result_data[2] == 16);  // 4² = 16
    assert(int_result_data[3] == 1);   // (-1)² = 1
    assert(int_result_data[4] == 25);  // (-5)² = 25
    assert(int_result_data[5] == 0);   // 0² = 0
    
    std::cout << "✓ Int32 sqr() test passed!\n";
    
    // Test with Float32
    Tensor float_tensor(Shape{{2, 2}}, Dtype::Float32, Device::CPU, false);
    float* float_data = static_cast<float*>(float_tensor.data());
    float_data[0] = 2.5f; float_data[1] = -1.5f;
    float_data[2] = 0.0f; float_data[3] = 3.0f;
    
    Tensor float_result = sqr(float_tensor);
    const float* float_result_data = static_cast<const float*>(float_result.data());
    
    assert(float_result_data[0] == 6.25f);  // 2.5² = 6.25
    assert(float_result_data[1] == 2.25f);  // (-1.5)² = 2.25
    assert(float_result_data[2] == 0.0f);   // 0² = 0
    assert(float_result_data[3] == 9.0f);   // 3² = 9
    
    std::cout << "✓ Float32 sqr() test passed!\n";
}

void test_neg_operation() {
    std::cout << "Testing unary - operator...\n";
    
    // Test with Int32
    Tensor int_tensor(Shape{{2, 2}}, Dtype::Int32, Device::CPU, false);
    int32_t* int_data = static_cast<int32_t*>(int_tensor.data());
    int_data[0] = 5; int_data[1] = -3;
    int_data[2] = 0; int_data[3] = 10;
    
    Tensor int_neg = -int_tensor;
    const int32_t* int_neg_data = static_cast<const int32_t*>(int_neg.data());
    
    assert(int_neg_data[0] == -5);   // -5
    assert(int_neg_data[1] == 3);    // -(-3) = 3
    assert(int_neg_data[2] == 0);    // -0 = 0
    assert(int_neg_data[3] == -10);  // -10
    
    std::cout << "✓ Int32 unary - test passed!\n";
    
    // Test with Float32
    Tensor float_tensor(Shape{{1, 3}}, Dtype::Float32, Device::CPU, false);
    float* float_data = static_cast<float*>(float_tensor.data());
    float_data[0] = 2.5f; float_data[1] = -1.5f; float_data[2] = 0.0f;
    
    Tensor float_neg = -float_tensor;
    const float* float_neg_data = static_cast<const float*>(float_neg.data());
    
    assert(float_neg_data[0] == -2.5f);  // -2.5
    assert(float_neg_data[1] == 1.5f);   // -(-1.5) = 1.5
    assert(float_neg_data[2] == 0.0f);   // -0 = 0
    
    std::cout << "✓ Float32 unary - test passed!\n";
}

void test_tensor_metadata() {
    std::cout << "Testing tensor metadata preservation...\n";
    
    Tensor original(Shape{{3, 2}}, Dtype::Float32, Device::CPU, true);
    
    // Test sqr preserves metadata
    Tensor squared = sqr(original);
    assert(squared.shape() == original.shape());
    assert(squared.dtype() == original.dtype());
    assert(squared.device() == original.device());
    assert(squared.requires_grad() == original.requires_grad());
    
    // Test neg preserves metadata  
    Tensor negated = -original;
    assert(negated.shape() == original.shape());
    assert(negated.dtype() == original.dtype());
    assert(negated.device() == original.device());
    assert(negated.requires_grad() == original.requires_grad());
    
    std::cout << "✓ Metadata preservation test passed!\n";
}

int main() {
    std::cout << "Starting Tensor Operations Test Suite...\n\n";
    
    try {
        test_sqr_operation();
        std::cout << "\n";
        
        test_neg_operation(); 
        std::cout << "\n";
        
        test_tensor_metadata();
        std::cout << "\n";
        
        std::cout << "🎉 All tests passed! Tensor operations are working correctly.\n";
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with error: " << e.what() << "\n";
        return 1;
    }
}
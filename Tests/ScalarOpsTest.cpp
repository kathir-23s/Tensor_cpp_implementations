#include <iostream>
#include <cstdint>
#include "Tensor.h"
#include "ScalarOps.h"

using namespace std;

void test_tensor_add_int32() {
    cout << "=== Tensor + int32_t ===" << endl;

    Tensor a(Shape{ {2,2} }, Dtype::Int32, Device::CPU, false);
    int32_t* pa = static_cast<int32_t*>(a.data());

    // Fill tensor
    pa[0] = 1; pa[1] = 2; pa[2] = 3; pa[3] = 4;

    // Add int32_t scalar
    Tensor c = a + static_cast<int32_t>(10);
    int32_t* pc = static_cast<int32_t*>(c.data());

    cout << "A: ";
    for (int i = 0; i < 4; i++) cout << pa[i] << " ";
    cout << "\nC = A + 10: ";
    for (int i = 0; i < 4; i++) cout << pc[i] << " ";
    cout << "\n\n";
}

void test_tensor_add_float() {
    cout << "=== Tensor + float ===" << endl;

    Tensor a(Shape{ {3} }, Dtype::Float32, Device::CPU, false);
    float* pa = static_cast<float*>(a.data());

    // Fill tensor
    pa[0] = 1.5f; pa[1] = 2.5f; pa[2] = 3.5f;

    // Add float scalar
    Tensor c = a + 2.5f;
    float* pc = static_cast<float*>(c.data());

    cout << "A: ";
    for (int i = 0; i < 3; i++) cout << pa[i] << " ";
    cout << "\nC = A + 2.5: ";
    for (int i = 0; i < 3; i++) cout << pc[i] << " ";
    cout << "\n\n";
}

int main() {
    test_tensor_add_int32();
    test_tensor_add_float();
    return 0;
}

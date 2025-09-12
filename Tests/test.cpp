#include <iostream>
#include "Tensor.h"

using namespace std;

void test1() {
    // Create a 2x3 tensor of float32 on CPU
    Tensor t(Shape{{3, 2, 3}}, Dtype::Float32, Device::CPU, true);
    
    // Fill the tensor with some values
    float* data = static_cast<float*>(t.data());
    for (int i = 0; i < 12; ++i) {
        data[i] = static_cast<float>(i) + 0.5f;
    }
    
    // Display the tensor
    cout << "Tensor contents:" << endl;
    t.display(cout);
    
}

int main() {
    test1();
    return 0;
}
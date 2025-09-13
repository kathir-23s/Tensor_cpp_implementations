#include "Tensor.h"
#include "UnaryOps.h"
#include <iostream>

void compare_power_ops(){
    std::cout << "Test for comparing assembly of power function\n\n";

    Tensor t(Shape{{1,2}}, Dtype::Int32, Device::CPU, false);
    int32_t* int_data = static_cast<int32_t*>(t.data());
    for (int i=0; i<2; i++){
        int_data[i] = i;
    }

    // Calling Class method
    Tensor class_res = t.power(2);
    const int32_t* class_res_data = static_cast<const int32_t*>(class_res.data());

    // Calling Free Method
    Tensor free_res = pow(t, 3);
    const int32_t* free_res_data = static_cast<const int32_t*>(free_res.data());

    // Outputs
    std::cout << "\n Result from Class method \n";
    class_res.display(std::cout);


    std::cout << "\n Result from Free method \n";
    free_res.display(std::cout);

}

int main() {
    try {
        compare_power_ops();
    std::cout << "Executed successfully\n\n";
return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with error: " << e.what() << "\n";
        return 1;
    }
}
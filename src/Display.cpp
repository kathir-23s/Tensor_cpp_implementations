#include <iostream>
#include <cstring>
#include <iomanip>
#include <stdexcept>
#include <Tensor.h>

using namespace std;

void Tensor::display(ostream& os) const {
    int64_t total_elements = 1;
    for (auto dim : shape_.dims) total_elements *= dim;
    // Print shape header similar to PyTorch: e.g. Tensor(shape=[2, 3], dtype=float32)
    os << "Tensor(shape=[";
    for (size_t i = 0; i < shape_.dims.size(); ++i) {
        os << shape_.dims[i];
        if (i + 1 < shape_.dims.size()) os << ", ";
    }
    os << "], dtype=";
    if (dtype_ == Dtype::Int32) os << "int32";
    else if (dtype_ == Dtype::Float32) os << "float32";
    else os << "unknown";
    os << ")\n";

    // Handle empty tensor
    if (total_elements == 0) {
        os << "[]\n";
        return;
    }

    // Helper to print values with appropriate formatting
    auto print_value = [&](size_t idx) {
        if (dtype_ == Dtype::Int32) {
            const int32_t* base = reinterpret_cast<const int32_t*>(data_.data());
            os << base[idx];
        } else if (dtype_ == Dtype::Float32) {
            const float* base = reinterpret_cast<const float*>(data_.data());
            os << fixed << setprecision(4) << base[idx];
        } else {
            throw std::runtime_error("Unsupported dtype for display");
        }
    };

    // For 1-D tensors, print as a single row
    if (shape_.dims.size() == 1) {
        os << "[";
        for (int64_t i = 0; i < total_elements; ++i) {
            if (i) os << ", ";
            print_value(i);
        }
        os << "]\n";
        return;
    }

    // For 2-D tensors, format rows with brackets like PyTorch
    if (shape_.dims.size() == 2) {
        int64_t rows = shape_.dims[0];
        int64_t cols = shape_.dims[1];
        os << "[";
        for (int64_t r = 0; r < rows; ++r) {
            if (r) os << " \n "; // newline and indent similar to PyTorch
            os << " [";
            for (int64_t c = 0; c < cols; ++c) {
                int64_t idx = r * cols + c;
                if (c) os << ", ";
                print_value(idx);
            }
            os << "]";
            if (r + 1 < rows) os << ",";
        }
        os << " ]\n";
        return;
    }

    // For N-D tensors (N>2), print flat data but keep shape header
    os << "[";
    for (int64_t i = 0; i < total_elements; ++i) {
        if (i) os << ", ";
        print_value(i);
    }
    os << "]\n";
}
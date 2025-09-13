#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <ostream>
#include <stdexcept>


using namespace std;

//Datatypes
enum class Dtype { Int32, Float32 };

// Devices
enum class Device { CPU, CUDA };

//Shape and Stride
struct Shape{
    vector<int64_t> dims; // e.g., {rows, cols}
};

struct Stride {
    vector<int64_t> strides; // e.g., {row_stride, col_stride}
};

class Tensor {
    public: 
        Tensor(Shape shape, Dtype dtype, Device device = Device::CPU, bool requires_grad = false);

        //Functions to Access Metadata
        vector<int64_t> shape() const { return shape_.dims; }
        vector<int64_t> stride() const { return stride_.strides; }
        int64_t ndim() const { return shape_.dims.size(); }
        int64_t size(int dim) const { 
            if (dim < 0 || dim >= ndim()) throw std::runtime_error("Dimension out of range");
            return shape_.dims[dim];
        }
        
        Dtype dtype() const { return dtype_; }
        Device device() const { return device_; }
        bool requires_grad() const { 
            return requires_grad_; 
       }

        void *data() { return data_.data(); }
        const void *data() const { return data_.data(); }
        int64_t nbytes() const { return data_.size() * sizeof(data_[0]); }

        void *grad() { return grad_.data(); }
        const void *grad() const { return grad_.data(); }
        int64_t grad_nbytes() const { return grad_.size() * sizeof(grad_[0]); }

        
        void display(ostream& os) const;

        int64_t size() const { 
        int64_t total = 1;
        for (auto dim : shape_.dims) total *= dim;
        return total;
        }

        Tensor power(int exp);

    private:
        Shape shape_;
        Stride stride_;
        Dtype dtype_;
        Device device_;
        bool requires_grad_;

        std::vector<uint8_t> data_; // Raw data storage
        std::vector<uint8_t> grad_;  // Gradient storage, if required


};
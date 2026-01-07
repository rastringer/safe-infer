#pragma once

#include "safe_infer/tensor_shape.h"
#include <cstddef>
#include <vector>

namespace safe_infer {

class Tensor {

public: 
    // Allocates float buffer sized to shape.num_elements(). 
    explicit Tensor(TensorShape shape);

    // Move-only
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&&) noexcept = default;
    Tensor& operator=(Tensor&&) noexcept = default;

    // Shape() accessor
    const TensorShape& shape() const noexcept;
    std::size_t num_elements() const noexcept;

    // data() accessor 
    float* data() noexcept;
    const float* data() const noexcept;
    float& operator[](std::size_t i) noexcept;
    const float& operator[](std::size_t i) const noexcept;


private: 

    TensorShape shape_;
    std::vector<float> data_;

};
    
} // namespace safe_infer

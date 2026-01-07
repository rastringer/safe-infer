#include "safe_infer/tensor.h"
#include <utility>

namespace safe_infer {

Tensor::Tensor(TensorShape shape) 
    : shape_(std::move(shape))
    , data_(shape_.num_elements())
{}

const TensorShape& Tensor::shape() const noexcept {
    return shape_;
}

std::size_t Tensor::num_elements() const noexcept {
    return data_.size();
}

// data() accessor
float* Tensor::data() noexcept {
    return data_.data();
} 
const float* Tensor::data() const noexcept {
    return data_.data();
}

float& Tensor::operator[](std::size_t i) noexcept {
    return data_[i];
}

const float& Tensor::operator[](std::size_t i) const noexcept {
    return data_[i];
}

} // namespace safe_infer
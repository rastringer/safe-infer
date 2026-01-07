#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace safe_infer {

    
// TensorShape cvlass
// Value type, immutable after construction
// Responsible for safe size computation
// No knowledge of tensors, memory or ops
class TensorShape {

public:
	// Construct from dimensions
    explicit TensorShape(std::vector<std::size_t> dims);

    // Basic accessors
    std::size_t rank() const noexcept;
    const std::vector<std::size_t>& dims() const noexcept;

    // Safe size computation
    std::size_t num_elements() const;

    // Logs, debugging
    std::string to_string() const;

private:
	// Invariant: dims_ is not empay, > 0 
	// and num_elements() doesn't overflow size_t
    std::vector<std::size_t> dims_;
};

} // namespace safe_infer
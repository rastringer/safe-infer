#include "safe_infer/tensor_shape.h"
#include <stdexcept>
#include <limits>
#include <sstream>
#include <utility>


namespace safe_infer {


TensorShape::TensorShape(std::vector<std::size_t> dims) {
		if (dims.empty()) {
		throw std::domain_error("Dims cannot be empty");
		}
	
		for (std::size_t d : dims) {
			if (d == 0) {
				throw std::domain_error("TensorShape dims must be > 0");
			}	 
		}
		dims_ = std::move(dims);	
	}

std::size_t TensorShape::rank() const noexcept {
	return dims_.size();
}

const std::vector<std::size_t>& TensorShape::dims() const noexcept {
	return dims_;
}

std::size_t TensorShape::num_elements() const {
		const std::size_t max_size = std::numeric_limits<std::size_t>::max();
		std::size_t product = 1;

		for (std::size_t d : dims_) {
			if (product > max_size / d) {
				throw std::overflow_error("TensorShape: num_elements overflow");
			}
			product *= d;
		}
		return product;
	}

std::string TensorShape::to_string() const {
		std::ostringstream s;
		bool first = true;
		s << "[";
		for (std::size_t d : dims_) {
				if (!first) {
					s << " x ";
				}
				first = false;
				s << d;
		}
		s << "]";
		return s.str();
}
} // namespace safe_infer


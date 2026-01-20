#pragma once

#include <cstddef>
#include <vector>

namespace safe_infer {

// IDs are just indices into vectors in Graph
using TensorId = std::size_t;
using NodeId = std::size_t;

// Minimal op set for now
enum class OpCode {
    Input, // produces tensor, provided externally
    Const, // produces tensor, stored in the model
    Add, // elementwise add: out = a + b
    MatMul, // matrix multiplication
    Relu // activation function, max(0, x)
};

// Our node consumes input tensors and produces output tensors
struct Node {
    OpCode op{};
    std::vector<TensorId> inputs;
    std::vector<TensorId> outputs;
}; 

} // namespace safe_infer
#pragma once

#include "safe_infer/graph.h"
#include "safe_infer/tensor.h"

#include <vector>

namespace safe_infer {

// Tracks which TensorIds are bound by the caller.
// Minimal -- we track presence, not ownership of buffers.
class InputBindings {
public:
    explicit InputBindings(std::size_t num_tensors);

    // Mark a tensor id as bound/provided by the caller.
    void bind(TensorId id);

    bool is_bound(TensorId id) const noexcept;

private:
    std::vector<bool> bound_;
};


// Executes nodes in 'plan' order, reading and writing tensors in-place.
// Current preconditions:
// tensors.size() == g.tensor_shapes.size()
// all tensors allocated with matching shapes
// only OpCode::Input, OpCode::Relu, OpCode::Add used

// Throws std::domain_error is precondition violated
void execute(const Graph& g, const std::vector<NodeId>& plan, std::vector<Tensor>& tensors, const InputBindings& bindings);

} // namespace safe_infer
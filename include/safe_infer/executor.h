#pragma once

#include "safe_infer/graph.h"
#include "safe_infer/tensor.h"

#include <vector>

namespace safe_infer {

// Executes nodes in 'plan' order, reading and writing tensors in-place.
// Current preconditions:
// tensors.size() == g.tensor_shapes.size()
// all tensors allocated with matching shapes
// only OpCode::Input, OpCode::Relu, OpCode::Add used

// Throws std::domain_error is precondition violated
void execute(const Graph& g, const std::vector<NodeId>& plan, std::vector<Tensor>& tensors);

} // namespace safe_infer
#pragma once

#include "safe_infer/op.h"
#include "safe_infer/tensor_shape.h"

#include <vector> 

namespace safe_infer {

// A minimal computation graph prioritizing learning concepts
// - tensor_shapes[t] describes tensor t
// - nodes[n] describes op node n
// - graph_inputs are tensors provided by the caller
// - graph_outputs are tensors 
struct Graph {
    std::vector<TensorShape> tensor_shapes;
    std::vector<Node> nodes;

    std::vector<TensorId> graph_inputs;
    std::vector<TensorId> graph_outputs;
};

} // namespace infer
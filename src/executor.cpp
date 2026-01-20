#include "safe_infer/executor.h"

#include <algorithm>    
#include <stdexcept>   
#include <string>       

namespace safe_infer {

namespace {

void require(bool cond, const char* msg) {
    if (!cond) {
        throw std::domain_error(msg);
    }
}

std::size_t elem_count(const Tensor& t) noexcept {
    return t.num_elements();
}

void exec_relu(const Node& node, std::vector<Tensor>& tensors) {
    require(node.inputs.size() == 1, "Relu: expected 1 input");
    require(node.outputs.size() == 1, "Relu: expected 1 output");

    const TensorId in_id  = node.inputs[0];
    const TensorId out_id = node.outputs[0];

    Tensor& out = tensors[out_id];
    const Tensor& in = tensors[in_id];

    require(elem_count(in) == elem_count(out), "Relu: input/output element counts must match");

    for (std::size_t i = 0; i < in.num_elements(); ++i) {
        const float x = in[i];
        out[i] = (x > 0.0f) ? x : 0.0f;
    }
}

void exec_add(const Node& node, std::vector<Tensor>& tensors) {
    require(node.inputs.size() == 2, "Add: expected 2 inputs");
    require(node.outputs.size() == 1, "Add: expected 1 output");

    const TensorId a_id = node.inputs[0];
    const TensorId b_id = node.inputs[1];
    const TensorId out_id = node.outputs[0];

    const Tensor& a = tensors[a_id];
    const Tensor& b = tensors[b_id];
    Tensor& out = tensors[out_id];

    require(elem_count(a) == elem_count(b), "Add: input element counts must match");
    require(elem_count(a) == elem_count(out), "Add: output element count must match inputs");

    for (std::size_t i = 0; i < a.num_elements(); ++i) {
        out[i] = a[i] + b[i];
    }
}

} // namespace

void execute(const Graph& g, const std::vector<NodeId>& plan, std::vector<Tensor>& tensors) {
    require(tensors.size() == g.tensor_shapes.size(),
            "execute: tensors.size() must equal g.tensor_shapes.size()");

    // (Optional sanity) ensure plan indexes are valid
    for (NodeId nid : plan) {
        require(nid < g.nodes.size(), "execute: plan contains invalid NodeId");
    }

    for (NodeId nid : plan) {
        const Node& node = g.nodes[nid];

        switch (node.op) {
            case OpCode::Input:
                // No-op for now: the input tensor is expected to be pre-filled by the caller.
                break;

            case OpCode::Relu:
                exec_relu(node, tensors);
                break;

            case OpCode::Add:
                exec_add(node, tensors);
                break;

            default:
                throw std::domain_error("execute: unsupported OpCode in PR#4");
        }
    }
}

} // namespace safe_infer

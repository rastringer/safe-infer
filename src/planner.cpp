#include "safe_infer/planner.h"

#include <unordered_map>
#include <queue>
#include <stdexcept>

namespace safe_infer {

std::vector<NodeId> plan_execution(const Graph& g) {
    // Validation
    const std::size_t num_tensors = g.tensor_shapes.size();
    auto check_tensor_id = [&](TensorId t) {
        if (t >= num_tensors) {
            throw std::domain_error("Graph refernces invalid TensorId");
        }
    };

    // Run check_tensor_id over node and graph inputs and outputs
    for (TensorId t : g.graph_inputs) {
        check_tensor_id(t);
    }
    for (TensorId t : g.graph_outputs) {
        check_tensor_id(t);
    } 
    for (const Node& node : g.nodes) {
        for (TensorId t : node.inputs) {
            check_tensor_id(t);
        }
        for (TensorId t : node.outputs) {
            check_tensor_id(t);
        }
    }

    // Check each tensor has a single producer
    std::vector<bool> has_producer(num_tensors, false);
    std::vector<NodeId> producer(num_tensors, static_cast<NodeId>(-1));

    for (NodeId nid = 0; nid < g.nodes.size(); ++nid) {
        const Node& node = g.nodes[nid];
    
        for (TensorId out : node.outputs) {
            // out's range is validated above
            if (has_producer[out]) {
                throw std::domain_error("Graph invalid: tensor has multiple producers");
            }
            has_producer[out] = true;
            producer[out] = nid; 
        }
    }

    const std::size_t num_nodes = g.nodes.size();
    std::vector<std::vector<NodeId>> adj(num_nodes);
    std::vector<std::size_t> indegree(num_nodes, 0);

    for (NodeId nid = 0; nid < num_nodes; ++nid) {
    const Node& node = g.nodes[nid];

    for (TensorId in : node.inputs) {
        // If this input tensor is produced by some node, we depend on it.
        if (has_producer[in]) {
            NodeId pred = producer[in];
            if (pred == nid) {
                throw std::domain_error("Graph invalid: node depends on its own output");
            }
            adj[pred].push_back(nid);
            indegree[nid] += 1;
        }
    }
}

    // Topological sort
    std::queue<NodeId> q;
    for (NodeId nid = 0; nid < num_nodes; ++nid) {
        if (indegree[nid] == 0) {
            q.push(nid);
        }
    }

    std::vector<NodeId> order;
    order.reserve(num_nodes);

    while (!q.empty()) {
        NodeId cur = q.front();
        q.pop();
        order.push_back(cur);

        for (NodeId nxt : adj[cur]) {
            if (--indegree[nxt] == 0) {
                q.push(nxt);
            }
        }
    }

    if (order.size() != num_nodes) {
        throw std::domain_error("Graph invalid: cycle detected");
    }

    return order;
}

} // namespace safe_infer

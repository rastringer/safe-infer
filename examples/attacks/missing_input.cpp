#include "safe_infer/executor.h"
#include "safe_infer/planner.h"

#include <iostream>
#include <limits>
#include <vector>

int main() {
    using namespace safe_infer;

    // Graph: y = Relu(x)
    Graph g;
    g.tensor_shapes = {
        TensorShape({8}), // tensor 0: x
        TensorShape({8})  // tensor 1: y
    };

    g.nodes.push_back(Node{OpCode::Relu, {0}, {1}});
    g.graph_inputs  = {0};
    g.graph_outputs = {1};

    const auto plan = plan_execution(g);

    std::vector<Tensor> tensors;
    tensors.emplace_back(g.tensor_shapes[0]); // x
    tensors.emplace_back(g.tensor_shapes[1]); // y

    // Intentionally DO NOT bind input 0.
    // This simulates a missing or forgotten input binding.
    InputBindings bindings(g.tensor_shapes.size());

    try {
        execute(g, plan, tensors, bindings);

        std::cerr << "ERROR: execution succeeded despite missing input binding\n";
        return 1;
    } catch (const std::domain_error& e) {
        std::cout << "Expected failure: " << e.what() << "\n";
    }

    return 0;
}

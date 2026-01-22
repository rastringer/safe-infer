#include "safe_infer/executor.h"
#include "safe_infer/planner.h"

#include <iostream>
#include <vector>

int main() {
    using namespace safe_infer;

    // Build a graph: out = a + b
    // But a has 4 elements and b has 3 -> mismatch.
    Graph g;
    g.tensor_shapes = {
        TensorShape(std::vector<std::size_t>{4}), // tensor 0: a
        TensorShape(std::vector<std::size_t>{3}), // tensor 1: b (mismatch!)
        TensorShape(std::vector<std::size_t>{4})  // tensor 2: out
    };

    g.nodes.push_back(Node{OpCode::Add, {0, 1}, {2}});
    g.graph_inputs = {0, 1};
    g.graph_outputs = {2};

    try {
        const auto plan = plan_execution(g);

        std::vector<Tensor> tensors;
        tensors.emplace_back(g.tensor_shapes[0]);
        tensors.emplace_back(g.tensor_shapes[1]);
        tensors.emplace_back(g.tensor_shapes[2]);

        // Fill a and b with something (b is shorter)
        for (std::size_t i = 0; i < tensors[0].num_elements(); ++i) {
            tensors[0][i] = static_cast<float>(i);
        }
        for (std::size_t i = 0; i < tensors[1].num_elements(); ++i) {
            tensors[1][i] = 100.0f + static_cast<float>(i);
        }

        // This should throw (Add requires same element count).
        execute(g, plan, tensors);

        std::cout << "Unexpected success (this should have thrown)\n";
        return 1;
    } catch (const std::exception& e) {
        std::cout << "Expected failure: " << e.what() << "\n";
        return 0;
    }
}

#include "safe_infer/executor.h"
#include "safe_infer/planner.h"

#include <cmath>        // std::isnan
#include <iostream>
#include <limits>       // quiet_NaN
#include <vector>

int main() {
    using namespace safe_infer;

    // Graph: y = Relu(x)
    Graph g;
    g.tensor_shapes = {
        TensorShape(std::vector<std::size_t>{8}), // tensor 0: x
        TensorShape(std::vector<std::size_t>{8})  // tensor 1: y
    };

    g.nodes.push_back(Node{OpCode::Relu, {0}, {1}});
    g.graph_inputs  = {0};
    g.graph_outputs = {1};

    const auto plan = plan_execution(g);

    std::vector<Tensor> tensors;
    tensors.emplace_back(g.tensor_shapes[0]); // x
    tensors.emplace_back(g.tensor_shapes[1]); // y

    // "Missing input" simulation:
    // Instead of binding real input data, we poison x with NaNs.
    // This makes the failure deterministic and visible.
    const float NaN = std::numeric_limits<float>::quiet_NaN();
    for (std::size_t i = 0; i < tensors[0].num_elements(); ++i) {
        tensors[0][i] = NaN;
    }

    execute(g, plan, tensors);

    // Print output and count NaNs
    std::size_t nan_count = 0;
    std::cout << "Output y: ";
    for (std::size_t i = 0; i < tensors[1].num_elements(); ++i) {
        const float v = tensors[1][i];
        if (std::isnan(v)) {
            ++nan_count;
        }
        std::cout << v << " ";
    }
    std::cout << "\n";

    std::cout << "NaN count in output: " << nan_count
              << " / " << tensors[1].num_elements() << "\n";

    std::cout << "\nWhy this matters:\n"
              << "- The graph executed without any error.\n"
              << "- The result is meaningless because the input binding was missing.\n"
              << "- In real systems, this can look like a model bug when it's actually an input/contract bug.\n";

    return 0;
}

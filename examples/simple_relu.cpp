#include "safe_infer/executor.h"
#include "safe_infer/planner.h"

#include <iostream>
#include <vector>

int main() {
    using namespace safe_infer;

    Graph g;
    g.tensor_shapes = {
        TensorShape(std::vector<std::size_t>{4}), // x (tensor 0)
        TensorShape(std::vector<std::size_t>{4})  // y (tensor 1)
    };

    g.nodes.push_back(Node{OpCode::Relu, {0}, {1}});
    g.graph_inputs = {0};
    g.graph_outputs = {1};

    const auto plan = plan_execution(g);

    std::vector<Tensor> tensors;
    tensors.emplace_back(g.tensor_shapes[0]);
    tensors.emplace_back(g.tensor_shapes[1]);

    // Fill input x
    tensors[0][0] = -1.0f;
    tensors[0][1] =  2.0f;
    tensors[0][2] = -3.0f;
    tensors[0][3] =  4.0f;

    InputBindings bindings(g.tensor_shapes.size());
    bindings.bind(0); // x is provided by caller

    execute(g, plan, tensors, bindings);

    // Print output y
    std::cout << "y: ";
    for (std::size_t i = 0; i < tensors[1].num_elements(); ++i) {
        std::cout << tensors[1][i] << " ";
    }
    std::cout << "\n";

    return 0;
}

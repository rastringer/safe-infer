#include "safe_infer/executor.h"
#include "safe_infer/planner.h"

#include <iostream>
#include <vector>

static void run_case(float x1, float x2) {
    using namespace safe_infer;

    Graph g;
    g.tensor_shapes = {
        TensorShape({1,2}), // t0: x
        TensorShape({2,2}), // t1: W1
        TensorShape({1,2}), // t2: z1
        TensorShape({1,2}), // t3: h (relu)
        TensorShape({2,1}), // t4: W2
        TensorShape({1,1})  // t5: y
    };

    // W1 columns are:
    // h1 preact = x1 - x2  => [ 1, -1 ]
    // h2 preact = -x1 + x2 => [ -1, 1 ]
    // row-major [ [1, -1],
    //             [-1, 1] ]
    std::vector<float> W1 = { 1.f, -1.f,
                            -1.f,  1.f };

    // W2 sums h1 + h2: [ [1],
    //                    [1] ]
    std::vector<float> W2 = { 1.f,
                              1.f };

    g.nodes.push_back(Node{OpCode::Const, {}, {1}, W1});
    g.nodes.push_back(Node{OpCode::Const, {}, {4}, W2});
    g.nodes.push_back(Node{OpCode::MatMul, {0, 1}, {2}});
    g.nodes.push_back(Node{OpCode::Relu,   {2},    {3}});
    g.nodes.push_back(Node{OpCode::MatMul, {3, 4}, {5}});

    g.graph_inputs = {0};
    g.graph_outputs = {5};

    const auto plan = plan_execution(g);

    std::vector<Tensor> tensors;
    for (const auto& s : g.tensor_shapes) tensors.emplace_back(s);

    // bind + fill input
    tensors[0][0] = x1;
    tensors[0][1] = x2;

    InputBindings bindings(g.tensor_shapes.size());
    bindings.bind(0);

    execute(g, plan, tensors, bindings);

    std::cout << "x=(" << x1 << "," << x2 << ")  =>  y=" << tensors[5][0] << "\n";
}

int main() {
    run_case(0.f, 0.f); // 0
    run_case(0.f, 1.f); // 1
    run_case(1.f, 0.f); // 1
    run_case(1.f, 1.f); // 0
}

#include "safe_infer/executor.h"
#include "safe_infer/planner.h"

#include <iostream>
#include <string>
#include <vector>

namespace {
int g_failures = 0;
void check(bool cond, const std::string& msg) {
    if (!cond) { std::cerr << "FAIL: " << msg << "\n"; ++g_failures; }
}
}

int main() {
    using namespace safe_infer;

    // A: [1x2]  B: [2x3]  Out: [1x3]
    Graph g;
    g.tensor_shapes = {
        TensorShape({1,2}), // t0 A
        TensorShape({2,3}), // t1 B
        TensorShape({1,3})  // t2 Out
    };

    g.nodes.push_back(Node{OpCode::MatMul, {0,1}, {2}});
    g.graph_inputs = {0,1};
    g.graph_outputs = {2};

    const auto plan = plan_execution(g);

    std::vector<Tensor> tensors;
    tensors.emplace_back(g.tensor_shapes[0]);
    tensors.emplace_back(g.tensor_shapes[1]);
    tensors.emplace_back(g.tensor_shapes[2]);

    // A = [1, 2]
    tensors[0][0] = 1.f; tensors[0][1] = 2.f;

    // B =
    // [ 1  2  3
    //   4  5  6 ]
    float Bv[] = {1,2,3, 4,5,6};
    for (int i=0;i<6;++i) tensors[1][i] = Bv[i];

    InputBindings bindings(g.tensor_shapes.size());
    bindings.bind(0);
    bindings.bind(1);

    execute(g, plan, tensors, bindings);

    // Out = [1,2] x B = [ (1*1+2*4), (1*2+2*5), (1*3+2*6) ] = [9,12,15]
    check(tensors[2][0] == 9.f,  "matmul out[0]==9");
    check(tensors[2][1] == 12.f, "matmul out[1]==12");
    check(tensors[2][2] == 15.f, "matmul out[2]==15");

    if (g_failures == 0) {
        std::cout << "PASS: test_matmul\n";
        return 0;
    }
    std::cerr << "FAILURES: " << g_failures << "\n";
    return 1;
}

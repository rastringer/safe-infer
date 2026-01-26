#include "safe_infer/executor.h"
#include "safe_infer/planner.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
int g_failures = 0;

void check(bool cond, const std::string& msg) {
    if (!cond) { std::cerr << "FAIL: " << msg << "\n"; ++g_failures; }
}

template <typename ExceptionT, typename Func>
void check_throws(Func&& f, const std::string& msg) {
    try {
        f();
        std::cerr << "FAIL: " << msg << " (no exception)\n";
        ++g_failures;
    } catch (const ExceptionT&) {
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << msg << " (wrong exception: " << e.what() << ")\n";
        ++g_failures;
    }
}
}

int main() {
    using namespace safe_infer;

    // Graph: c = Const([1,2,3,4])
    Graph g;
    g.tensor_shapes = { TensorShape({4}) };   // t0
    g.nodes.push_back(Node{OpCode::Const, {}, {0}, {1.f, 2.f, 3.f, 4.f}});
    g.graph_inputs = {};
    g.graph_outputs = {0};

    const auto plan = plan_execution(g);

    std::vector<Tensor> tensors;
    tensors.emplace_back(g.tensor_shapes[0]);

    InputBindings bindings(g.tensor_shapes.size()); // no inputs required

    execute(g, plan, tensors, bindings);

    check(tensors[0][0] == 1.f, "const: t0[0] == 1");
    check(tensors[0][3] == 4.f, "const: t0[3] == 4");

    if (g_failures == 0) {
        std::cout << "PASS: test_const\n";
        return 0;
    }
    std::cerr << "FAILURES: " << g_failures << "\n";
    return 1;
}

#include "safe_infer/executor.h"
#include "safe_infer/planner.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void check(bool cond, const std::string& msg) {
    if (!cond) {
        std::cerr << "FAIL: " << msg << "\n";
        ++g_failures;
    }
}

template <typename ExceptionT, typename Func>
void check_throws(Func&& f, const std::string& msg) {
    try {
        f();
        std::cerr << "FAIL: " << msg << " (no exception thrown)\n";
        ++g_failures;
    } catch (const ExceptionT&) {
        // expected
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << msg << " (wrong exception: " << e.what() << ")\n";
        ++g_failures;
    } catch (...) {
        std::cerr << "FAIL: " << msg << " (non-std exception)\n";
        ++g_failures;
    }
}

} // namespace

int main() {
    using namespace safe_infer;

    // Build a tiny graph: y = Relu(x)
    Graph g;
    g.tensor_shapes = {
        TensorShape({4}), // x (0)
        TensorShape({4})  // y (1)
    };
    g.nodes.push_back(Node{OpCode::Relu, {0}, {1}});
    g.graph_inputs = {0};
    g.graph_outputs = {1};

    const auto plan = plan_execution(g);

    // Allocate runtime tensors
    std::vector<Tensor> tensors;
    tensors.emplace_back(g.tensor_shapes[0]);
    tensors.emplace_back(g.tensor_shapes[1]);

    // ---- Test 1: Missing binding should throw ----
    {
        InputBindings bindings(g.tensor_shapes.size());
        // Note: do NOT bind tensor 0.

        check_throws<std::domain_error>(
            [&] { execute(g, plan, tensors, bindings); },
            "execute should throw if required input is not bound"
        );
    }

    // ---- Test 2: Bound input should execute successfully ----
    {
        InputBindings bindings(g.tensor_shapes.size());
        bindings.bind(0);

        // Fill x
        tensors[0][0] = -1.0f;
        tensors[0][1] =  2.0f;
        tensors[0][2] = -3.0f;
        tensors[0][3] =  4.0f;

        // Run
        try {
            execute(g, plan, tensors, bindings);
        } catch (const std::exception& e) {
            std::cerr << "FAIL: execute should not throw when input is bound ("
                      << e.what() << ")\n";
            ++g_failures;
        }

        // Check output y = relu(x)
        check(tensors[1][0] == 0.0f, "relu: y[0] == 0");
        check(tensors[1][1] == 2.0f, "relu: y[1] == 2");
        check(tensors[1][2] == 0.0f, "relu: y[2] == 0");
        check(tensors[1][3] == 4.0f, "relu: y[3] == 4");
    }

    if (g_failures == 0) {
        std::cout << "PASS: test_executor\n";
        return 0;
    }
    std::cerr << "FAILURES: " << g_failures << "\n";
    return 1;
}

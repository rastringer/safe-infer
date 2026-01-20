#include "safe_infer/planner.h"
#include "safe_infer/graph.h"
#include "safe_infer/tensor_shape.h"

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

bool appears_before(const std::vector<safe_infer::NodeId>& order,
                    safe_infer::NodeId a,
                    safe_infer::NodeId b) {
    std::size_t pa = order.size(), pb = order.size();
    for (std::size_t i = 0; i < order.size(); ++i) {
        if (order[i] == a) pa = i;
        if (order[i] == b) pb = i;
    }
    return pa < pb;
}

} // namespace

int main() {
    using namespace safe_infer;

    // Test 1: Simple chain A -> B -> C
    {
        Graph g;
        g.tensor_shapes = {
            TensorShape({1}), TensorShape({1}), TensorShape({1}), TensorShape({1})
        };

        // Node 0: Input produces tensor 0 (external semantics, but treated like a node output here)
        g.nodes.push_back(Node{OpCode::Input, {}, {0}});
        // Node 1: Relu consumes 0 produces 1
        g.nodes.push_back(Node{OpCode::Relu, {0}, {1}});
        // Node 2: Relu consumes 1 produces 2
        g.nodes.push_back(Node{OpCode::Relu, {1}, {2}});

        g.graph_inputs = {0};
        g.graph_outputs = {2};

        auto order = plan_execution(g);
        check(order.size() == 3, "chain: order should include all 3 nodes");
        check(appears_before(order, 0, 1), "chain: node0 should appear before node1");
        check(appears_before(order, 1, 2), "chain: node1 should appear before node2");
    }

    // Test 2: Diamond A -> {B,C} -> D
    {
        Graph g;
        g.tensor_shapes = {
            TensorShape({1}), TensorShape({1}), TensorShape({1}), TensorShape({1}), TensorShape({1})
        };

        // 0 produces t0
        g.nodes.push_back(Node{OpCode::Input, {}, {0}});
        // 1 consumes t0 produces t1
        g.nodes.push_back(Node{OpCode::Relu, {0}, {1}});
        // 2 consumes t0 produces t2
        g.nodes.push_back(Node{OpCode::Relu, {0}, {2}});
        // 3 consumes t1,t2 produces t3
        g.nodes.push_back(Node{OpCode::Add, {1, 2}, {3}});

        g.graph_inputs = {0};
        g.graph_outputs = {3};

        auto order = plan_execution(g);
        check(order.size() == 4, "diamond: order should include all 4 nodes");
        check(appears_before(order, 0, 1), "diamond: node0 before node1");
        check(appears_before(order, 0, 2), "diamond: node0 before node2");
        check(appears_before(order, 1, 3), "diamond: node1 before node3");
        check(appears_before(order, 2, 3), "diamond: node2 before node3");
    }

    // Test 3: Cycle should throw
    {
        check_throws<std::domain_error>(
            [] {
                Graph g;
                g.tensor_shapes = {TensorShape({1}), TensorShape({1})};

                // node0: consumes t1 produces t0
                g.nodes.push_back(Node{OpCode::Relu, {1}, {0}});
                // node1: consumes t0 produces t1
                g.nodes.push_back(Node{OpCode::Relu, {0}, {1}});

                g.graph_inputs = {};   // none
                g.graph_outputs = {0}; // arbitrary

                (void)plan_execution(g);
            },
            "cycle: should throw std::domain_error"
        );
    }

    // Test 4: Invalid TensorId should throw
    {
        check_throws<std::domain_error>(
            [] {
                Graph g;
                g.tensor_shapes = {TensorShape({1})}; // only tensor 0 exists

                // references tensor 5 (invalid)
                g.nodes.push_back(Node{OpCode::Relu, {5}, {0}});
                g.graph_inputs = {0};
                g.graph_outputs = {0};

                (void)plan_execution(g);
            },
            "invalid tensor id: should throw std::domain_error"
        );
    }

    if (g_failures == 0) {
        std::cout << "PASS: test_planner\n";
        return 0;
    }
    std::cerr << "FAILURES: " << g_failures << "\n";
    return 1;
}

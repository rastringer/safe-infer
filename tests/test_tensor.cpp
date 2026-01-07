#include "safe_infer/tensor.h"
#include "safe_infer/tensor_shape.h"

#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

int g_failures = 0;

void check(bool condition, const std::string& msg) {
    if (!condition) {
        std::cerr << "FAIL: " << msg << "\n";
        ++g_failures;
    }
}

} // namespace

int main() {
    using safe_infer::Tensor;
    using safe_infer::TensorShape;

    // Construct and size check
    {
        TensorShape s(std::vector<std::size_t>{2, 3, 4});
        Tensor t(std::move(s));

        check(t.num_elements() == 24, "Tensor should allocate 24 elements for shape [2,3,4]");
    }

    // Write/read check
    {
        TensorShape s(std::vector<std::size_t>{2, 2});
        Tensor t(std::move(s));

        for (std::size_t i = 0; i < t.num_elements(); ++i) {
            t[i] = static_cast<float>(i) * 1.5f;
        }

        check(t[0] == 0.0f, "t[0] should equal 0.0");
        check(t[1] == 1.5f, "t[1] should equal 1.5");
        check(t[2] == 3.0f, "t[2] should equal 3.0");
        check(t[3] == 4.5f, "t[3] should equal 4.5");
    }

    // Move check
    {
        TensorShape s(std::vector<std::size_t>{3});
        Tensor t(std::move(s));
        t[0] = 42.0f;
        t[1] = 43.0f;
        t[2] = 44.0f;

        Tensor moved = std::move(t);
        check(moved.num_elements() == 3, "Moved tensor should keep element count");
        check(moved[0] == 42.0f, "Moved tensor should preserve data[0]");
        check(moved[2] == 44.0f, "Moved tensor should preserve data[2]");
    }

    if (g_failures == 0) {
        std::cout << "PASS: test_tensor\n";
        return 0;
    }

    std::cerr << "FAILURES: " << g_failures << "\n";
    return 1;
}

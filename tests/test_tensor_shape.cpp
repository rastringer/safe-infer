#include "safe_infer/tensor_shape.h"

#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

int g_failures = 0;

void check(bool condition, const std::string& msg) {
    if (!condition) {
        std::cerr << "FAIL: " << msg << "\n";
        ++g_failures;
    }
}

template <typename ExceptionT, typename Func>
void check_throws(Func&& f, const std::string& msg) {
    try {
        f();
        std::cerr << "FAIL: " << msg << "(no exception thrown)\n";
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
    using safe_infer::TensorShape;

    // Success
    {
        TensorShape s(std::vector<std::size_t>{2,3,4});
        check(s.rank() == 3, "rank should be 3 for shape [2,3,4]");
        check(s.num_elements() == 24, "num_elements should be 24 for shape [2,3,4]");

    }

    // Empty dims should fail
    {
        check_throws<std::domain_error>(
            [] { TensorShape s(std::vector<std::size_t>{}); },
            "empty dims should throw std::domain_error"
        );
    }

    // Zero dimension should fail
    {
        check_throws<std::domain_error>(
            [] { TensorShape s(std::vector<std::size_t>{2, 0, 3}); },
            "zero dimension should throw std::domain_error"
        );
    }

    // Overflow should fail (max * 2 overflows size_t)
    {
        check_throws<std::overflow_error>(
            [] {
                TensorShape s(std::vector<std::size_t>{
                    std::numeric_limits<std::size_t>::max(), 2
                });
                (void)s.num_elements();
            },
            "overflow in num_elements should throw std::overflow_error"
        );
    }

    if (g_failures == 0) {
        std::cout << "PASS: test_tensor_shape\n";
        return 0;
    }

    std::cerr << "FAILURES: " << g_failures << "\n";
    return 1;
}
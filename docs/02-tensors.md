# Lesson 2

Please see the pull request for this stage [here](https://github.com/rastringer/safe-infer/commit/08f9d8e14f0a3f55a68d67052fbc8e3685d48e35).

## TensorShapes, Invariants, and Safe Size Computation

### Goals

Understand how unchecked tensor sizes can lead to security bugs and undefined behaviour; what an invariant is and how to safely compute the number of elements in a tensor.

### TensorShapes

A tensor shape determines how much memory is allocated, how to compute indexing, and which memory locations are read and written to. A malformed shape can cause buffer overflow, memory corruption or crashes. 

Tensor
┌───────────────────────────────────────────────┐
│                                               │
│  TensorShape                                  │
│  ┌───────────────┐                            │
│  │ dims = [2, 3] │  rank = 2                  │
│  │               │  num_elements = 6          │
│  └───────────────┘                            │
│                                               │
│  data (contiguous memory)                     │
│  ┌───────┬───────┬───────┬───────┬───────┬───────┐
│  │ 0.12  │ -1.3  │  2.0  │  0.0  │  4.1  │ -0.7  │
│  └───────┴───────┴───────┴───────┴───────┴───────┘
│                                               │
│  flat storage, shape-aware interpretation     │
│                                               │
└───────────────────────────────────────────────┘



### Enforcing Invariants

An invariant is a property that always must be true for an object to be valid. A key design principle for our small engine is that we enforce these invariants early:

* The shape must not be empty
* All dimensions must be greater than zero
* The total number of elements must fit in `size_t`

Rather than checking and re-checking these conditions as part of our program, we enforce them once, in the contructor. 

### The `TensorShape` Interface

* `rank()` -- the number of dimensions
* `dims()` -- inspect the dimensions
* `num_elements()` -- total number of elements
* `to_string` -- debugging aid

The interface lacks setters or mutable access to dimensions, since after contruction, `TensorShape` cannot be mutated.

### Contstructor Validation

Here is the constructor that enforces our invariants:

```
// tensor_shape.cpp
TensorShape::TensorShape(std::vector<std::size_t> dims) {
if (dims.empty()) {
throw std::domain_error("TensorShape: dims cannot be empty");
}


for (std::size_t d : dims) {
if (d == 0) {
throw std::domain_error("TensorShape: dims must be > 0");
}
}


dims_ = std::move(dims);
}
```

### Safe Size Computation

A naive implementation of `num_elements()` might look like this:

```
std::size_t n = 1;
for (auto d : dims_) {
    n *= d;
}
```

This code is problematic because if the multiplication overflows `size_t`, the behaviour is undefined, and the result wraps around. The wrapped value could then be used to allocate too little memory, index out of bounds, or even to corrupt unrelated memory.

### Overflow-Safe Multiplication

Instead, we check explicitly for overflow:

```
std::size_t TensorShape::num_elements() const {
    const std::size_t max_size = std::numeric_limits<std::size_t>::max();
    std::size_t product = 1;

    for (std::size_t d : dims) {
        if (product > max_size / d) {
            throw std::overflow_error("TensorShape: num_elements overflow");
        }
        product *= d;
    }
    return product;
}
```

By checking `product > max_size / d`, we detect overflow before it occurs.

### Accessors and Immutability

We expose dimensions via:

`const std::vector<std::size_t>& dims() const noexcept;`

This communicates three guarantees:

* the caller cannot modify the dimensions
* calling the function doesn't modify the object
* the function cannot throw

### Tests

We add a test file to verify the program:

* rejects empty shapes and zero dimensions 
* detects overflow

### Next Lesson

We introduce the `Tensor` type, which owns contiguous memory, uses RAII-based lifetime management and move-only semantic to prevent copies.
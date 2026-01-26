# Lesson 3

## Memory Safety with `Tensor`

### Goals

Understand why **memory ownership** is the central safety concern in an inference engine; how RAII makes memory management explicit and reliable; why large data structures should often be **move-only** and how `TensorShape` and `Tensor` work together to enforce safety

This lesson corresponds to PR [#2](https://github.com/rastringer/safe-infer/commit/d8a2019be437a848cb68365e9200d2f3ad6a931f): **Add move-only Tensor (RAII-owned storage)**.


### Tensors Are a Safety Boundary

When talking about ML systems, a tensor represents a contiguous block of memory which indexed by arithmetic derived from shapes. Tensors are typically read and written by multiple operations.

Results of tensor mismanagement may include buffer overruns, use-after-free bugs, and silent data corruption. Once tensor memory is corrupted, no amount of correct math can recover correctness, so tensor ownership a first-class safety concern.

### How Tensor Shapes can Become Memory Accesses

```

                 (USER / MODEL / FILE INPUT)
                         dims = [d0, d1, ...]
                                |
                                v
                 ┌────────────────────────────────┐
                 │  TensorShape constructor        │
                 │  Invariants enforced once:      │
                 │   • dims not empty              │
                 │   • each d > 0                  │
                 │   • num_elements fits in size_t │
                 └───────────────┬────────────────┘
                                 |
                                 v
                    overflow-safe num_elements()
                 product = 1
                 for d in dims:
                    if product > MAX / d:
                        throw overflow_error
                    product *= d
                                 |
                                 v
                      n = num_elements (trusted)
                                 |
                                 v
         ┌─────────────────────────────────────────────────┐
         │                   MEMORY                         │
         │ allocate n * sizeof(T) bytes  →  indices [0..n)  │
         └─────────────────────────────────────────────────┘
                                 |
                                 v
                 indexing uses the same shape math
             linear_index(i, …) computed from dims
                                 |
                                 v
        address = base + linear_index * sizeof(T)
                                 |
                                 v
        SAFE: linear_index < n  ⇒ access stays in bounds
```

### Design Principle: One Clear Owner

The central design choice for `Tensor` is simple:

> **There can be only one owner of a tensor's memory.**

We enforce this by using RAII for lifetime management, deleting copy operations, and allowing ownership to be transferred via move semantics.

This prevents accidental, implicit duplication of large buffers.
```

        ┌──────────────┐        move        ┌──────────────┐
        │   Tensor t1  │  ───────────────▶  │   Tensor t2  │
        │  owns buffer │                    │  owns buffer │
        └──────┬───────┘                    └──────┬───────┘
               │                                   │
               │ moved-from                        │ scope ends
               v                                   v
        valid but inert                    ~Tensor() called
        (do not use)                       vector frees memory

```

### The `Tensor` Interface

The public interface of `Tensor` is intentionally minimal:

* construction from a valid `TensorShape`
* access to shape and element count
* raw data access for computation
* indexed element access

Notably absent are:

* default constructors
* copy constructors
* shared ownership

This keeps ownership explicit and easy to reason about.


### Construction and Allocation

The constructor ties allocation directly to the validated shape:

```cpp
Tensor::Tensor(TensorShape shape)
    : shape_(std::move(shape))
    , data_(shape_.num_elements())
{}
```

Key points:

* The shape is moved into the tensor, transferring ownership
* We compute the number of elements *once*, using overflow-safe logic
* `std::vector<float>` allocates a contiguous buffer of the correct size

Successful construction means the resulting tensor is fully valid.


## RAII and Automatic Cleanup

The tensor does not explicitly free memory, instead relying on RAII:

* `std::vector<float>` frees its buffer in its destructor
* `Tensor` does not need a custom destructor

This guarantees:

* no memory leaks
* no double frees
* correct cleanup even in the presence of exceptions

RAII makes correct behavior the default.


### Why the Tensor Is Move-Only

Copying a tensor would mean copying its entire buffer. Copying significant amounts of data would be expensive, often accidental and may not be what a developer intended. 

To prevent this, we delete copy operations:

```cpp
Tensor(const Tensor&) = delete;
Tensor& operator=(const Tensor&) = delete;
```

And allow moves:

```cpp
Tensor(Tensor&&) noexcept = default;
Tensor& operator=(Tensor&&) noexcept = default;
```

By moving rather than copying a tensor, we transfer ownership of the buffer.


### Moved-From State

After a move:

```cpp
Tensor t2 = std::move(t1);
```

`t2` owns the data.

`t1` is left in a valid but unspecified state:

* it can be destroyed safely
* it should not be used for computation


### Accessors and Contracts

The accessors are designed to communicate who owns the data, what can be mutated, and what operations cannot throw.

```cpp
const TensorShape& shape() const noexcept;
std::size_t num_elements() const noexcept;
float* data() noexcept;
const float* data() const noexcept;
```

Encoding these guarantees in the type system reduces misuse.


### Element Access

Here is the indexing operator, which is intentionally unchecked.

```cpp
float& operator[](std::size_t i) noexcept;
```

Bounds checking should ideally be carried out at higher abstraction layers, and unchecked access is predictable and fast. We improve engine safety through shape validation, correct allocation and controlled usage patterns.


### Tests as Proof of Design

The accompanying tests verify correct allocation size, read/write behavior and correct behavior after `move` operations.


### Current Status

At this point, we have:

* explicit ownership of tensor memory
* automatic cleanup via RAII
* protection against accidental deep copies
* a clear separation between metadata (`TensorShape`) and storage (`Tensor`)

This is the core memory safety foundation of the inference engine.


### Next lesson

With tensors in place, we can begin defining **operations and computation graphs**:

* nodes and edges
* dependency tracking
* execution order

This will introduce graph traversal and scheduling — and new classes of safety concerns.

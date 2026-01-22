# Lesson 5 

## From Graph to Execution

---

### Goals

* How a validated execution plan is turned into **actual computation**
* Why planning and execution are deliberately separated
* Assumptions a minimal executor can safely make
* How even a tiny executor exposes important safety boundaries

This lesson corresponds to PR [#4](https://github.com/rastringer/safe-infer/commit/2cf609ce13808670a6911bf56b52628b653acf91): **Add minimal executor and first runnable inference**.

---

### From Structure to Action

Having built a computation graph and a planner that produces a valid execution order, the system knows:

* *what* operations exist
* *how* they depend on one another
* *in what order* they must run

In this lesson, we specify how the system should perform computations.

---

### Separation of Concerns: Planning vs Execution

A deliberate design choice in this project is to separate:

* **planning** (graph validation and scheduling)
* **execution** (running operations and producing outputs)

This is because planning is about correctness and safety, while execution is about state mutation and computation. Keeping these concerns separate makes both stages easier to reason about and test.

---

### Runtime State: The Tensor Store

Execution operates over a simple runtime state:

```cpp
std::vector<Tensor> tensors;
```

Each entry corresponds to a `TensorId` in the graph.

At execution time:

* tensors are already allocated
* input tensors are pre-filled by the caller
* output tensors are written by operations

The executor does not allocate or resize tensors.

---

### The Executor Interface

As with many functions in this course, the core execution API is intentionally minimal:

```cpp
void execute(const Graph& g,
             const std::vector<NodeId>& plan,
             std::vector<Tensor>& tensors);
```

This function:

* assumes the graph has already been validated
* executes nodes in the given order
* mutates the tensor store in place

Any violation of its assumptions results in an explicit error.

---

### Supported Operations (Initial Set)

For this lesson, the executor supports only a small set of operations to keep the focus on end-to-end inference:

* `Input` — no-op (input tensors are already populated)
* `Relu` — elementwise `max(0, x)`
* `Add` — elementwise addition

---

## Executing a Node

Execution proceeds node by node:

1. Look up the `Node` using its `NodeId`
2. Dispatch based on `OpCode`
3. Read input tensors
4. Write output tensors

Each operation enforces its own local invariants, such as:

* number of inputs and outputs
* matching element counts


---

### Minimal Kernels

Here is our simple ReLU. As a reminder, here is how a *rectified linear unit* functions:

* if the input is positive, the output is the same value
* if the input is zero or negative, the output is zero

```cpp
out[i] = (in[i] > 0.0f) ? in[i] : 0.0f;
```

And Add:

```cpp
out[i] = a[i] + b[i];
```

No broadcasting, no shape inference, no optimizations.

The goal is clarity, not performance.

---

### A First End-to-End Example

With planning and execution in place, we can now run a complete inference:

```
x  →  Relu  →  y
```

Given input:

```
[-1, 2, -3, 4]
```

The output is:

```
[0, 2, 0, 4]
```

This confirms that:

* the graph was planned correctly
* execution followed the correct order
* tensor data flowed as intended

---

### Executor Assumptions

The current executor makes strong assumptions:

* tensors are one-dimensional
* shapes match exactly
* only supported ops appear
* tensor memory is correctly allocated

Such assumptions are *intentional*. They allow us to see where safety checks belong and what happens when they are missing.

---

### Why This Matters for Safety

Even this tiny executor reveals key safety questions:

* What happens if shapes do not match?
* What if an op is unsupported?
* What if inputs are missing or uninitialized?

In real systems, these failures can lead to:

* silent mis-computation
* memory corruption
* undefined behavior

Understanding where assumptions live is the first step to hardening them.

---

### Current Status

At this stage, the system can:

* represent a model as a graph
* validate its structure
* plan execution order
* execute operations
* produce real numerical outputs

This is a complete (if minimal) inference pipeline.

---

### Next Lesson 

Now that inference works, we can explore how it can fail.

The next lesson will intentionally break assumptions:

* mismatched shapes
* missing inputs
* unsupported operations

This is where safety and security concerns become explicit.

---

**Next lesson:** Lesson 6 — Breaking the Executor: Failure Modes and Attacks

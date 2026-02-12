# Lesson 5 

## From Graph to Execution


### Goals

In this lesson, we cover:

* How a validated execution plan is turned into computation
* Why planning and execution are deliberately separated
* Assumptions a minimal executor can safely make
* How even a tiny executor exposes important safety boundaries

This lesson corresponds to PR [#4](https://github.com/rastringer/safe-infer/commit/2cf609ce13808670a6911bf56b52628b653acf91): **Add minimal executor and first runnable inference**.


### From Structure to Action

Having built a computation graph and a planner that produces a valid execution order, the system knows what operations exist, how they depend on one another and in what order they must run. We now need to specify how the system should perform computations.


### Separation of Concerns: Planning vs Execution

A deliberate design choice in this project is to separate:

* planning (graph validation and scheduling)
* execution (running operations and producing outputs)

This is because planning is about correctness and safety, while execution is about state mutation and computation. Keeping these concerns separate makes both stages easier to reason about and test.


### Runtime State: The Tensor Store

Execution operates over a simple runtime state:

```cpp
std::vector<Tensor> tensors;
```

Each entry corresponds to a `TensorId` in the graph.

At execution time, the tensors are already allocated. Input tensors are pre-filled by the caller and output tensors are written by operations. The executor does not allocate or resize tensors.


### The Executor Interface

As with many functions in this course, the core execution API is intentionally minimal:

```cpp
void execute(const Graph& g,
             const std::vector<NodeId>& plan,
             std::vector<Tensor>& tensors);
```

This function assumes the graph has already been validated, executes nodes in the given order and mutates the tensor store in place. Any violation of its assumptions results in an explicit error.


### Supported Operations (Initial Set)

For this lesson, the executor supports only a small set of operations to keep the focus on end-to-end inference:

* `Input` — no-op (input tensors are already populated)
* `Relu` — elementwise `max(0, x)`
* `Add` — elementwise addition


## Executing a Node

Execution proceeds node by node:

1. Look up the `Node` using its `NodeId`
2. Dispatch based on `OpCode`
3. Read input tensors
4. Write output tensors

Each operation enforces its own local invariants, such as the number of inputs and outputs, or matching element counts.


### Minimal Kernels

Here's our simple ReLU. As a reminder, a *rectified linear unit* returns the input if positive, and zero if the input is zero or negative.

```cpp
out[i] = (in[i] > 0.0f) ? in[i] : 0.0f;
```

And Add:

```cpp
out[i] = a[i] + b[i];
```


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

This shows that the graph is planned correctly and tensor data flows as intended.


### Executor Assumptions

The current executor makes strong assumptions:

* tensors are one-dimensional
* shapes match exactly
* only supported ops appear
* tensor memory is correctly allocated

Such assumptions are *intentional*. They allow us to see where safety checks belong and what happens when they are missing.


### Why This Matters for Safety

Even this tiny executor reveals key safety questions. For example, what happens if shapes do not match, or if a caller ask for an op is unsupported? What if inputs are missing or uninitialized? In real systems, these failures can lead to mis-computation that may not be immediately obvious; memory corruption or undefined behaviour.


### Current Status

At this stage, the our minimal pipeline represents a model as a graph and validates its structure. The engine then plans execution order, conducts operation and provides outputs.


### Next Lesson 

Now that inference works, we can explore how it can fail.

The next lesson will intentionally break assumptions:

* mismatched shapes
* missing inputs
* unsupported operations

This is where safety and security concerns become explicit.


**Next lesson:** Lesson 6 — Breaking the Executor: Failure Modes and Attacks

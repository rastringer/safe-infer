# Lesson 6

## Failure Modes and Attacks


### Goals

In this lesson, we explore:

* Why working inference is not the same as safe inference
* How hidden assumptions in executors become concrete failure modes
* The difference between fail-fast errors and silent mis-computation
* How attackers (or accidents) exploit unchecked contracts in inference systems

This lesson builds directly on the previous lesson and PR [#4](https://github.com/rastringer/safe-infer/commit/2cf609ce13808670a6911bf56b52628b653acf91).


### A Dangerous Milestone

At the end of the previous lesson our system validated a graph, planned the execution, ran inference and gave correct-looking outputs. While it's tempting to sit back and enjoy a system that appears to be functioning well, we need to explore how it can go wrong rather than assume its operations are safe.

### Correctness vs Safety

A system can be correct for expected inputs, and unsafe for unexpected or adversarial inputs. Inference engines sit at a boundary between trusted runtime code and untrusted models, graphs, and inputs, so scrutiny anything crossing this boundary is essential.

### The Executor’s Assumptions

Before breaking the system, let’s make its assumptions explicit.

The current executor assumes:

* all input tensors are correctly initialized
* tensor shapes match where required
* only supported operations appear
* execution order is valid

While some of these assumptions are checked, others aren't, and unchecked assumptions define the attack surface.


## Failure Mode 1: Shape Mismatch (Fail-Fast)

**Example:** `examples/attacks/shape_mismatch.cpp`

In this example, the graph attempts to add two tensors with different element counts. The graph plans successfully, begins execution, and the `Add` kernel triggers an error when it detects a mismatch.

```bash
./build/example_shape_mismatch
```

You should see:

```
Expected failure: Add: input element counts must match
```

This is a *good* failure, since execution stops immediately (before producing incorrect outputs), the error is explicit. However, the invalid graph was still allowed to reach execution.


### Failure Mode 2: Missing Input Binding (Silent Failure)

**Example:** `examples/attacks/missing_input.cpp`

Here, the graph expects an input tensor. The caller never binds real input data, yet the executor proceeds regardless.

To better observe this behaviour, we poison the input tensor with **NaNs**.

Run it:

```bash
./build/example_missing_input
```

Since the execution completes successfully, and the output looks plausible (all zeros), the NaNs were *silently masked* by the ReLU operation. The NaNs were silently masked by the ReLU operation. Though our input is trivial in this example, a real attacker could inject:

* extremely large or tiny values to trigger numerical instability
* values crafted to push activations into saturation regimes (killing gradients)
* inputs to introduce downstream bias while appearing valid

Mis-computation can be worse than a system crash since symptoms such as incorrect predictions, bias and false confidence in model behaviour can be more harmful, and difficult to detect.

After Lesson 7 and PR [#6](https://github.com/rastringer/safe-infer/commit/a467f43634655b81f66152b0f1261ed72fd09d2e), this becomes a fail-fast error.

There are other failure modes our system needs to address:

### Failure Mode 3: Unsupported Operations (Fail-Fast)

We are covered here since the executor explicitly rejects unsupported ops.


## Failure Mode 4: Contract Mismatches Between Stages

Our system's planner validates the graph, and the executor trusts the planner. However, what if the graph is modified after the planning is completed? What if runtime tensors don't match graph metadata? We need explicit constraints to avoid undefined behaviour.


## Current Status

We have discussed potential attacks and failures to identify assumptions, observe consequences and will think about remedies.


## Next Lesson

We start hardening the runtime.


**Next lesson:** Lesson 7 — Hardening the Runtime: Contracts and Validation

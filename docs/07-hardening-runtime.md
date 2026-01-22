# Lesson 7 

## Hardening the Runtime: Contracts and Validation

---

### Goals

This lesson explores:

- How to turn implicit assumptions into **explicit contracts**
- Where validation should live (planner vs executor)
- How to eliminate entire classes of failures demonstrated in Lesson 6
- Why “fail-fast” behavior is a security feature

This lesson corresponds to PR [#5](https://github.com/rastringer/safe-infer/commit/4aa729a9af4f34bc65f556b1952a6dd7378b638f), "Planner-side hardening: enforce graph contracts".

---

## The Problem: Assumptions Are an Attack Surface

In Lesson 6 we saw two categories of failures:

- **Fail-fast**: the system throws and stops -- this is intended
- **Silent failure**: the system produces plausible but meaningless output -- this could be abused or cause issues

At a high level, the silent failures happen when a system takes on a contract without enforcing it. 'Hardening' here means we make the contract explicit and enforce it early. 

---

### Contract 1: Every Tensor Must Have a Source

A node input tensor must be either produced by a node in the graph, or provided as a graph input. The graph should be considered invalid if neither is true of a tensor. This prevents graphs with undefined variables and makes execution deterministic.

**Hardening rule:**
> A graph must not reference tensors that have no defined source.

---

### Contract 2: Op Arity Must Be Valid

In Lesson 5, we assumed `Relu` has exactly 1 input and 1 output, and `Add` has exactly 2 inputs and 1 output. Hardening for these functions means we validate arity at graph-validation time, and, if failure is necessary, we fail before reaching execution. This prevents out-of-range indexing into `node.inputs`, and accidental interpretation of malformed nodes.

---

### Contract 3: Inputs Must Be Explicitly Bound

The missing-input example showed that though inference ran successfully and the output (zeroes) looked plausible, the input wasn't correctly provided. 

To prevent this, the runtime introduces an explicit concept of **input binding**. This stipulates that a caller must bind each `graph_input`, and the execution checks bindings before running. This should turn silent failures into obvious errors which are easier to debug.


Hardening rule:
> If a required input is not bound, execution must fail fast.


---

### Where to Validate: Planner vs Executor

We take the approach that by validating the **structure** in the planner, we can satisfy that tensor IDs are in range; that we have a single producer; the graph is acyclic; and every input has a source.

By validating **runtime** state in the executor, our required inputs are bound; output buffers are allocated and have appropriate shape compatibility.

This separation keeps each layer small and trustworthy.

---

## Revisiting Lesson 6

### Shape mismatch

Previously, our graph planned and the executor made rejections at runtime. Now, the graph validator can make rejections earlier, while the executor remains a last line of defence.

### Missing input binding

In the last lesson, the graph executed anyway, and NaNs were masked into zeroes. Now, missing bindings are detected before execution begins, and the system fails fast with a clear message.

After PR [#6](https://github.com/rastringer/safe-infer/commit/a467f43634655b81f66152b0f1261ed72fd09d2e), this becomes a fail-fast error.

---

## Current Status 

We added clear contracts and enforced them at suitable stages in the engine's operations to:
- reject malformed graphs early
- detect missing inputs deterministically
- make failures loud rather than silent

---

## Next Lesson

As with many security considerations in software, the tradeoff is that more checks make an application safer and easier to debug while increasing the computational overhead.

In the next lesson, we’ll discuss how production systems balance
performance vs safety, and how we can aim for both optimized checking and speed.

---

**Next lesson:** Lesson 8 — Performance vs Safety Tradeoffs

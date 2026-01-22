# Lesson 8 

## Performance vs Safety: Engineering Tradeoffs

---

### Goals

A discussion of how production systems balance performance, correctness and security; how to reason about tradeoffs as an engineer.

This lesson builds on Lessons 6 and 7 and PRs [#5](https://github.com/rastringer/safe-infer/commit/2cf609ce13808670a6911bf56b52628b653acf91) and [#6](https://github.com/rastringer/safe-infer/commit/a467f43634655b81f66152b0f1261ed72fd09d2e).

---

## Costs

Every safety check has a cost, whether CPU time, memory overhead or code complexity. They can add up at scale, and in machine learning or other systems that perform billions and trillions of operations. Engineers reason about which checks are worth paying for, and where to conduct them.

---

### Where to Validate

In our inference engine, validation now exists in the planner, which checks the structures (eg tensor shapes); the executor, which checks runtime contracts, and kernels, which check for local correctness. Each layer has different tradeoffs.

---

### Planner-Side Validation

Planner validation happens once per graph, and before any execution. Examples from the course include:

* op arity validation (PR #5)
* every tensor must have a source (PR #5)
* cycle detection

### Pros

* cheap relative to execution
* catches entire classes of bugs early
* simplifies downstream code

### Cons

* cannot catch data-dependent errors
* assumes the graph will not change after planning

The design decision:

> Structural invariants belong in the planner.

---

## Executor-Side Validation

Executor validation happens once per execution, and before or during runntime.

Examples from this course include:

* input binding checks (PR #6)
* tensor shape consistency
* plan index validation

### Pros

* prevents silent mis-computation
* protects against misuse of the API
* enforces contracts between stages

### Cons

* adds overhead to every run
* must be carefully scoped

Design decision:

> Cross-stage contracts belong in the executor.

---

### Kernel-Level Checks

Kernel checks happen inside tight loops.

Examples:

* element count matching in `Add`
* bounds assumptions

### Pros

* catches the last line of defense errors
* protects against undefined behavior

### Cons

* most expensive place to check
* can meaningfully impact performance

Design decision:

> Kernel checks should be minimal and defensive.

---

### Debug vs Release Builds

Real systems often distinguish between debug builds, which predictably have generous amounts of checks enabled, and release builds, which may thin out all but those checks considered essential.

Common strategies:

* assertions compiled out in release
* optional validation flags
* trusted vs untrusted execution paths

This is a *policy decision*, not a purely technical one.

---

### Trust Boundaries 

Checks are most valuable at so-called trust boundaries, such as the space between user input and runtime, serialized models and executors, and other components.

Our biggest trust boundary was the caller -> executor, which was hardened in PR [#6](https://github.com/rastringer/safe-infer/commit/a467f43634655b81f66152b0f1261ed72fd09d2e).

---

### Fail Fast vs Fail Safe

*Fail fast* typically means the system crashes early with a clear error. *Fail safe* means execution continues, albeit in a degraded mode. When creating inference systems, silent failure is generally worse than a crash, and incorrect outputs can be exponential trouble makers far downstream (failures, confusing, offensive or dangerous outputs etc).

---

### Why Real Systems Sometimes Relax Checks

Production systems may trust model loaders, assume validated graphs
and skip redundant or less crucial checks. Machine learning-related workflows can quickly become extremely expensive, especially at the scale of large companies or research organizations. Being economical with checks is often vital to make sustained progress.

While operating amid such constraints, we can still be careful to make assumptions explicit, document contracts, choose which validation costs are essential, and reason about which checks may be removed from runtime.

---

## What Now? 

As we come to the end of the core course, we have:

* built a minimal inference engine
* identified real failure modes
* hardened the system with explicit contracts

Further additions to the course (with enough interest in the core material thus far) will feature an optional capstone. In this project, we will add a small real world model and introduce constants and matrix multiplication.

---

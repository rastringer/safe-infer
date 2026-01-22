# Lesson 4 

## Operations, Graphs, and Execution Order

---

### Goals

This lesson covers:

* Why AI inference is fundamentally a **graph execution problem**
* How operations depend on one another via inputs and outputs
* Why **execution order must be explicit and validated**
* How graph validation prevents entire classes of runtime failures

This lesson corresponds to PR [#3](https://github.com/rastringer/safe-infer/commit/af2a5ed8caa5a8ed6172f16c9e25b90bba2530b6): **Graph IR and execution planner**.

---

### From Tensors to Programs

In the previous lessons, we focused on *individual building blocks*:

* `TensorShape` ensured safe metadata and sizes
* `Tensor` ensured correct memory ownership

Once we have multiple tensors and multiple operations, we need to consider the order in which operations should run. The answer is determined by data dependencies.

---

### Inference as a Directed Acyclic Graph (DAG)

An inference model can be viewed as a **directed graph** in which:

* nodes represent operations
* edges represent data flowing between operations

The direction of an edge encodes a requirement:

> *The producer of a tensor must run before any consumer of that tensor.*

If these requirements form a cycle, the program is invalid.

---

## The Graph Intermediate Representation

To make these ideas explicit, we introduce a small graph IR:

```cpp
struct Node {
    OpCode op;
    std::vector<TensorId> inputs;
    std::vector<TensorId> outputs;
};

struct Graph {
    std::vector<TensorShape> tensor_shapes;
    std::vector<Node> nodes;
    std::vector<TensorId> graph_inputs;
    std::vector<TensorId> graph_outputs;
};
```

This representation is intentionally minimal, contains no tensor data and encodes only *structure* and *dependencies*.

---

## Tensor IDs as Symbols

`TensorId` values are indices into `tensor_shapes`.

This means that every tensor referenced by the graph must exist, and invalid tensor IDs represent undefined variables. Before any execution logic runs, the graph must be validated.

---

### Validation as a Safety Boundary

The planner enforces several key invariants:

* all tensor IDs are in range
* each tensor has at most one producer
* nodes cannot depend on their own outputs

Rejecting invalid graphs early prevents:

* out-of-bounds memory access
* ambiguous writes
* unpredictable execution order

Execution code assumes the graph is valid.

---

### Producer–Consumer Relationships

To determine dependencies, the planner builds a producer map:

> *Which node produces each tensor?*

This allows us to derive edges:

* if node B consumes a tensor produced by node A
* then A must run before B

---

## Building the Dependency Graph

Once producers are known, dependencies are represented as:

* an adjacency list (edges between nodes)
* an indegree count (how many unmet dependencies each node has)

These structures make execution order explicit.

---

### Topological Sorting

To compute a valid execution order, we perform a [topological sort](https://en.wikipedia.org/wiki/Topological_sorting). This means we undertake a graph traversal, visiting each node only after dependant nodes are visited:

1. Start with all nodes that have no dependencies
2. Repeatedly schedule nodes whose dependencies are satisfied
3. Remove their outgoing edges

If all nodes can be scheduled, the graph is valid. Otherwise, the graph contains a cycle.

---

### Cycle Detection

Cycles represent impossible execution requirements:

* A depends on B
* B depends on A

The planner detects cycles by checking whether all nodes were scheduled. If not, it throws an error and rejects the graph. This prevents infinite loops and undefined behavior at runtime.

---

### Diagram

```
Legend:
  [OpN: OpCode] = node id N and operation
  (tK)          = TensorId K
  t0,t1 are graph_inputs (no producer required)
  arrows show "produces tensor consumed by"

DATAFLOW (what the model *means*)

   graph_inputs
   (t0)      (t1)
     |         |
     v         v
 [0:Input]  [1:Input]
    |          |
   (t2)       (t3)
      \       /
       \     /
        v   v
       [2:Add]
          |
         (t4)
          |
          v
       [3:Relu]
          |
         (t5)    graph_output
```

### Tests

The planner is validated with tests that cover:

* a simple linear chain of operations
* branching and merging dependencies
* cyclic graphs
* invalid tensor references

---

### Current Status

At this stage, the system can represent models as explicit programs, validate structural correctness, and compute a deterministic execution order.

---

### Next Lesson

We now know:

* *what* operations exist
* *how* they depend on one another
* *in what order* they should run

In the next lesson, we will actually **execute** the graph:

* allocate runtime tensors
* implement a minimal executor
* produce real numerical outputs

---

**Next lesson:** Lesson 5 — From Graph to Execution

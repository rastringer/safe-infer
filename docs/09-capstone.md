# Lesson 9

### Goals

Run a small model through our inference engine!

In this lesson, you will add to the code base to add:

- `Const` tensors for learned parameters
- a minimal `MatMul` op for matrix multiplication
- build a tiny neural network as a computation graph
- execute it using the planner and executor pipeline
- observe real outputs

Please bear in mind this is a simple engine to illustrate safe capabilities to run a simple model and we leave out the various performance optimizations and other capabilities we would see in a more fully-developed inference engine.

### XOR

We will implement a model that computes **XOR**.

XOR (“exclusive or”) behaves like this:

| x1 | x2 | XOR |
|----|----|-----|
| 0  | 0  |  0  |
| 0  | 1  |  1  |
| 1  | 0  |  1  |
| 1  | 1  |  0  |

Intuitively, output is 1 if the inputs are *different*, and output is 0 if they are the *same*

We use XOR because it's the smallest example that actually behaves like a neural network: it can't be computed with a single linear layer, forcnig us to use matrix multiplication, nonlinearity (via ReLU) and multiple layers. 


### The XOR model via ReLU

We'll compute XOR for inputs in {0, 1}  using a tiny network. 

We construct two hidden features: 
- h1 = relu(x1-x2)
- h2 = relu(x2 - x1)

We can consider these values as asking by how much is x1 bigger than x2, and x2 by x1?

Then we add them:

-y = h1 + h2

for binary inputs, this equals XOR.

### Graph structure

X -> MatMul(W1) -> Relu -> MatMul(W2) ->

Where `x` is a graph input tensor of shape [1,2], W1 and W2, the model's *weights*, are Const tensors.

### `Const`

`Const` matters since weights are part of the model, rather than runtime inputs. We make them const to ensure they are written once and then only read from that point. This mitigates either accidental or intentional misue - the weights cannot be treated as caller-provided inputs by the executor.

`Const` turns the model parameters into explicity graph nodes and tensor slots.

### MatMul

Matrix multiplication is the core building block of dense neural net layers. We implement only rank-2 matrices and perform shape-checked multiplication. We leave out more sophisticated operations such as broadcasting, batching, transposes or optimizations.

## Exercise

Here's a look at our entire system:

                 (model as data: Graph)
        ┌───────────────────────────────────┐
        │ Graph                             │
        │  - tensor_shapes (TensorId→Shape) │
        │  - nodes (NodeId→Op+in/out)       │
        │  - graph_inputs / graph_outputs   │
        └───────────────────┬───────────────┘
                            │
                            v
                    plan_execution(g)
      (validate + topo sort → NodeId execution order)
                            │
                            v
        ┌───────────────────────────────────┐
        │ Runtime                           │
        │  tensors[TensorId] : Tensor       │
        │  bindings : InputBindings         │
        └───────────────────┬───────────────┘
                            │
                            v
              execute(g, plan, tensors, bindings)
     (run kernels in plan order, read/write tensors)


Try to implement the following:

### 1: Add `Const`

Implement an operator that:

- takes 0 inputs
- produces 1 output
- copies stored values into its output tensor

There are three files which will need additions: `op.h`, `executor.cpp` and `planner.cpp`.

### 2: Add `MatMul`

Implement matrix multiplication for rank-2 tensors:

[N, D] x [D, M] -> [N, M]

- validate ranks, inner dimensions, output shape, and fail fast on mismatch.

Look at the same three files as in task **1**.

### 3: Build the XOR Graph

Create a graph with:

- input tensor `x`
- constant tensors `W1`, `W2`
- MatMul -> Relu -> MatMul pipeline
- proper graph_inputs / graph_outputs

Here's what we are aiming for:

TensorIds:
  t0 = x    [1x2]      (bound at runtime)
  t1 = W1   [2x2]      (Const)
  t2 = z1   [1x2]
  t3 = h    [1x2]
  t4 = W2   [2x1]      (Const)
  t5 = y    [1x1]

Nodes (dataflow):
      t0      t1
      │       │
      └──▶ MatMul ───▶ t2 ───▶ Relu ───▶ t3
                         t3      t4
                         │       │
                         └──▶ MatMul ───▶ t5


### 4: Run!

Run inference for:

(0, 1), (0, 1), (1, 0), (1, 1)

Output should look like this:

x=(0,0)  =>  y=0
x=(0,1)  =>  y=1
x=(1,0)  =>  y=1
x=(1,1)  =>  y=0

Please see [this PR](https://github.com/rastringer/safe-infer/commit/6c97b7ec21fd2a1e1e2bfe1b92a20f4d9f5f15ae) for the solution (ignore the `docs/` updates and focus on `op.h`, `executor.cpp` and `planner.cpp`).


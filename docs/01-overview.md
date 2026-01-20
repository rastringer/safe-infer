# Lesson 1

## AI Inference as a Systems Problem

### Goals

* Understand why AI safety and security can and should be considered from a *systems* approach, rather than exclusively as a model problem
* Learn about the basics of inference engines  
* Understand the scope of the course and an introduction to the codebase

### Why a course on inference engines?

Most AI safety and security materials focus on research topics such as model architectures, training techniques, accuracy metrics and various approaches to interpreting and ensuring behaviour.

When considering AI safety and security, however, we need to appraise the entire system supporting a call to a hosted model. Many failures and vulnerabilities can come from:

* memory safety bugs
* unchecked sizes and overflows
* invalid execution graphs
* nondeterministic behaviour
* unsafe assumptions at system boundaries

In this course, we will learn about how such issues arise and can be mitigated by building a small inference engine in C++. If you enjoy C++, hands on code examples and are curious about how the various AI tools and frameworks carry out their magic under the hood, this course is for you.

### The engine

We will build a small, CPU-only inference engine with the following properties:

* Dependency-light (STL only)
* Single-threaded initially
* Supports a small set of operations, such as matrix multiplication, add and ReLU. A real-world engine would of course have to allow for many more
* Uses a simple, custom model format


### Structure

* The course follows artifacts from GitHub pull requests which show our project progressing. Please follow along on your own machine by implementing the code as we go (recommended), or read along and consider what vulnerabilities or improvements may be relevant in the next lesson.

The course mimics the evolution of many systems:
* Build something simple
* See how it fails
* Reason about why
* Improve the design

### Phases

In Phase 1, we will focus on making basic execution of models work in modern, clean C++. The engine will function however will be unsafe.

In Phase 2, We will trigger crashes or corruptions and add validation, invariants and other mitigations to improve system safety and security.

### Next Lesson

We will build the foundation of the engine by defining tensor shapes.
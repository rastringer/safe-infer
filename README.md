# Safe Infer

**Safe Infer** is a small, educational C++ inference engine built to explore **AI safety and reliability as systems problems**.

Rather than focusing on model architecture or training, this project examines how real-world AI failures and security vulnerabilities arise from low-level issues such as memory ownership, unchecked sizes, invalid execution graphs, and undefined behavior.

The codebase is intentionally minimal, CPU-only, and dependency-light, with an emphasis on clarity, correctness, and explicit invariants.

---

## Project Goals

* Demonstrate how AI inference can be treated as **program execution**
* Show how small design choices lead to large safety & security consequences
* Teach modern C++ systems techniques (RAII, move semantics, validation)
* Provide a concrete artifact for learning and discussion

This is **not** a performance-focused engine and **not** a replacement for existing frameworks.

---

## Structure

```
safe-infer/
├── include/        # Public headers
├── src/            # Implementation
├── tests/          # Minimal executable tests
├── docs/           # Course lessons (Markdown)
└── CMakeLists.txt
```

Each lesson in `docs/` corresponds to a digestible addition to the codebase, usually via a PR.

---

## Course Format

The project is accompanied by a short online course structured around:

1. **Building a naïve inference engine**
2. **Identifying failure modes and attacks**
3. **Hardening the system through validation and design**

Lessons are written in Markdown and reference real code diffs rather than isolated snippets.

---

## Build Instructions

Requirements:

* C++20-compatible compiler
* CMake ≥ 3.20

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

---

## Current Status

* Tensor shapes with explicit invariants and overflow-safe size computation
* Move-only tensor type with RAII-owned storage

Further features (graphs, ops, execution, validation) are added incrementally.

---

## License

This work is licensed under CC BY-SA 4.0. To view a copy of this license, visit https://creativecommons.org/licenses/by-sa/4.0/
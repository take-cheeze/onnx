<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

# ONNX C++ micro-benchmarks

## Arena allocator for parsed `ModelProto` (`arena_parse_benchmark`)

### Is there a place to use a protobuf arena allocator in the C++ code?

Yes. protobuf supports [arena allocation](https://protobuf.dev/reference/cpp/arenas/)
for both proto2 and proto3 messages, and ONNX's generated messages support it out
of the box. An arena is worthwhile wherever the code builds a *transient* tree of
many small sub-messages and then throws the whole thing away, because:

- every `add_*()` / `mutable_*()` sub-message is carved from one contiguous block
  instead of a separate `operator new`, and
- the entire tree is reclaimed with a single bulk free instead of running one
  destructor + `free()` per object.

The clearest such site is the **ONNX text parser** (`OnnxParser::Parse`). It builds
the whole `ModelProto` tree by calling `mutable_*`/`add_*` on the caller-supplied
root message, and protobuf propagates the root's arena to every descendant created
that way. So creating the root with
`google::protobuf::Arena::CreateMessage<ModelProto>(&arena)` makes the entire node
/ attribute / tensor / value-info tree arena-allocated with no change to the parser
itself.

This is not hypothetical: the Python-exposed `onnx.parser.parse_model` /
`parse_graph` entry points (`onnx/cpp2py_export.cc`) do exactly *parse text → build
a `ModelProto` → serialize to bytes → discard the `ModelProto`* — the ideal
build-then-discard arena lifecycle.

Other candidate sites (not benchmarked here) with the same build-and-discard shape:
the function **inliner** (`onnx/inliner/inliner.cc`), which synthesizes new nodes and
graphs, and **shape inference**, which materializes temporary `TypeProto`s.

### Benchmark

`arena_parse_benchmark.cc` generates a large ONNX text model, then parses it into a
heap-allocated (default) `ModelProto` and into an arena-allocated one, timing the
*parse* (construction) and *destruction* phases separately. It depends only on the
public parser API and `std::chrono`, so it adds no new third-party dependency.

Build and run:

```bash
cmake -S . -B .setuptools-cmake-build -DONNX_BUILD_BENCHMARKS=ON
cmake --build .setuptools-cmake-build --target onnx_arena_parse_benchmark
# args: <num_nodes> <num_initializers> <initializer_len> <iterations>
.setuptools-cmake-build/onnx_arena_parse_benchmark 20000 500 64 20
```

### Results

Measured on this environment (Linux x86-64, GCC, `-O3` Release, system protobuf
3.21.12). Numbers are per-parse averages; "faster" is the arena's improvement over
heap.

| Model (nodes / inits) | phase | heap (ms) | arena (ms) | faster |
|---|---|---:|---:|---:|
| 2,000 / 100 | parse | 5.31 | 4.16 | 21.7% |
| | destroy | 2.04 | 0.18 | 91.4% |
| | **total** | **7.35** | **4.34** | **41.0%** |
| 20,000 / 500 | parse | 70.46 | 57.95 | 17.8% |
| | destroy | 38.90 | 7.06 | 81.9% |
| | **total** | **109.37** | **65.01** | **40.6%** |
| 50,000 / 1,000 | parse | 169.33 | 141.15 | 16.6% |
| | destroy | 88.88 | 15.86 | 82.2% |
| | **total** | **258.21** | **157.01** | **39.2%** |

### Takeaway

For building and tearing down a `ModelProto` tree, an arena is roughly **17–22%
faster to construct** and **~82–91% faster to destroy**, for an end-to-end speedup
of **~40%** that holds steady as the model grows. The construction win comes from
cheaper sub-message allocation; the (much larger) destruction win comes from
replacing thousands of individual destructor + `free()` calls with one arena
teardown.

The trade-off: arena memory is only reclaimed when the arena is destroyed, so an
arena fits transient, build-then-serialize/discard uses (like the parser bindings)
rather than long-lived `ModelProto`s whose sub-messages are individually released
over time. Adopting it in the parser bindings would be a small, contained,
backward-compatible change (the returned serialized bytes are identical); this
benchmark is provided as the evidence to justify that follow-up.

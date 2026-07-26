<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

# ONNX C++ micro-benchmarks

## Arena allocator for a transient `ModelProto` (`arena_parse_benchmark`)

### Is there a place to use a protobuf arena allocator in the C++ code?

Yes. protobuf supports [arena allocation](https://protobuf.dev/reference/cpp/arenas/)
for both proto2 and proto3 messages, and ONNX's generated messages support it out
of the box. An arena is worthwhile wherever the code builds (or deserializes) a
*transient* tree of many small sub-messages and then throws the whole thing away,
because:

- every `add_*()` / `mutable_*()` sub-message is carved from one contiguous block
  instead of a separate `operator new`, and
- the entire tree is reclaimed with a single bulk free instead of running one
  destructor + `free()` per object.

ONNX ingests a model in C++ two ways, and both fill a **caller-supplied root**
message, so making just the root arena-allocated (via
`google::protobuf::Arena::CreateMessage<ModelProto>(&arena)`) arenas the whole
node / attribute / tensor / value-info tree with no change to the ingest code:

- **Text**: `OnnxParser::Parse` (`onnx/defs/parser.h`) builds the tree via
  `mutable_*`/`add_*`.
- **Binary**: `ParseProtoFromBytes` (`onnx/proto_utils.h`) calls
  `ParseFromCodedStream` on the root. This is the path every
  `onnx/cpp2py_export.cc` binding takes when it receives serialized bytes from
  Python (parse bytes → operate → reserialize → discard the `ModelProto`) — the
  ideal build-then-discard arena lifecycle.

Other candidate sites (not benchmarked here) with the same build-and-discard
shape: the inliner's per-call-site temporary model
(`ConvertVersion`, `onnx/inliner/inliner.cc`), the path-based
`InferShapes`/`check_model` entry points and the `full_check` model copy
(`onnx/checker.cc`), and the version converter's `ModelProto → Graph → ModelProto`
round-trip (`onnx/common/ir_pb_converter.cc`).

### Benchmark

`arena_parse_benchmark.cc` generates a large ONNX model, then builds it into a
heap-allocated (default) `ModelProto` and into an arena-allocated one — once by
**text parsing** and once by **binary deserialize** — timing the *build* and
*destroy* phases separately. It depends only on public headers and `std::chrono`,
so it adds no new third-party dependency.

Build and run:

```bash
cmake -S . -B .setuptools-cmake-build -DONNX_BUILD_BENCHMARKS=ON
cmake --build .setuptools-cmake-build --target onnx_arena_parse_benchmark
# args: <num_nodes> <num_initializers> <initializer_len> <iterations>
.setuptools-cmake-build/onnx_arena_parse_benchmark 20000 500 64 20
```

### Results

Measured on this environment (Linux x86-64, GCC, `-O3` Release, system protobuf
3.21.12), 20k-node / 500-initializer model (~1.8 MiB text, ~1.9 MiB binary).
Per-op averages; "faster" is the arena's improvement over heap (negative = arena
is slower).

**Text parse — `OnnxParser::Parse`** (unambiguous win):

| phase | heap (ms) | arena (ms) | faster |
|---|---:|---:|---:|
| build | 46.9 | 31.0 | 33.8% |
| destroy | 14.0 | 1.1 | 92.1% |
| **total** | **60.9** | **32.1** | **47.2%** |

**Binary deserialize — `ParseProtoFromBytes`** (the cpp2py boundary path):

| phase | heap (ms) | arena (ms) | faster |
|---|---:|---:|---:|
| build | 14.1 | 18.1 | **-28.7%** |
| destroy | 11.8 | 1.9 | 83.9% |
| **total** | **25.9** | **20.0** | **22.6%** |

But the binary result is **strongly size-dependent** — the arena only pays off
above a crossover. Net end-to-end (build+destroy) speedup of the binary path vs
model size (200 initializers, len 32):

| nodes | heap total (ms) | arena total (ms) | faster |
|---:|---:|---:|---:|
| 2,000 | 2.75 | 3.44 | **-25.1%** |
| 5,000 | 6.56 | 8.38 | **-27.8%** |
| 10,000 | 13.98 | 9.19 | 34.3% |
| 20,000 | 27.15 | 21.39 | 21.2% |
| 40,000 | 53.00 | 47.23 | 10.9% |
| 80,000 | 157.37 | 105.60 | 32.9% |

### Takeaway

- **Text parse: a robust win at every size tested** (~40–48% end-to-end). The
  tokenizer is expensive and allocation-bound, so the arena helps *both* build
  (~30%) and destroy (~92%).
- **Binary deserialize: size-dependent, and a net loss for small/medium models.**
  protobuf's binary parser is already allocation-light and highly tuned, so on an
  arena the build phase is consistently *slower* (each call allocates a fresh
  arena and pays first-touch page faults the heap allocator avoids by recycling
  freed pages). The destruction win only outweighs that above a crossover around
  **~10k nodes** here; below it, the arena loses on every phase. Numbers are also
  noisier on this path.
- **The destruction win is real but not unconditional** — it scales with the
  number of sub-messages, so it dominates for large trees and is negligible for
  small ones.

So the strongest case for an arena is where construction is *also* expensive: the
text parser (`OnnxParser::Parse`) and the Tier-2 internal temporaries that are
built field-by-field. The pure binary-deserialize boundary is **not a blanket
win** — worthwhile only for large models, so any adoption there should be gated on
size (or simply skipped) rather than applied unconditionally.

### Adopting it at the cpp2py boundary (prototype pattern)

`ParseProtoFromBytes` needs **no change** — it already fills whatever root it is
given, arena-allocated or not. Only the call site changes. A small RAII helper
keeps the arena and its message together:

```cpp
// Owns an arena and a root message allocated on it; the whole tree is freed in
// one bulk operation when the holder goes out of scope.
template <typename Proto>
class ArenaProto {
 public:
  ArenaProto() : arena_(), msg_(google::protobuf::Arena::CreateMessage<Proto>(&arena_)) {}
  Proto* get() { return msg_; }
  Proto& operator*() { return *msg_; }
  Proto* operator->() { return msg_; }
 private:
  google::protobuf::Arena arena_;
  Proto* msg_;
};
```

A binding such as `infer_shapes` then becomes a one-line change — same input, byte-identical output:

```cpp
// before:
ModelProto proto{};
ParseProtoFromBytes(&proto, buffer, length);
shape_inference::InferShapes(proto);
return ProtoToBytes(proto);

// after:
ArenaProto<ModelProto> proto;
ParseProtoFromBytes(proto.get(), buffer, length);
shape_inference::InferShapes(*proto);
return ProtoToBytes(*proto);
```

The `RunBinary*` functions in `arena_parse_benchmark.cc` exercise exactly this
`ParseProtoFromBytes`-into-an-arena path, so the numbers above are the concrete
evidence for that follow-up.

Given the size-dependence, the clear first candidate is the **text**
`parse_model` / `parse_graph` binding (`Parse<>` in `onnx/cpp2py_export.cc`),
which wins at every size — the same helper applies with `OnnxParser::Parse` in
place of `ParseProtoFromBytes`. The binary bindings (`infer_shapes`,
`convert_version`, `inline_local_functions`) only benefit for large models and
should be gated on model size (or left on the heap) rather than switched
unconditionally. The trade-off is unchanged either way: arena memory is only
reclaimed when the arena dies, so this fits transient
parse-then-serialize/discard bindings, not long-lived `ModelProto`s whose
sub-messages are released individually over time.

## Internal transient protos — Tier 2 & 3 (`arena_internal_benchmark`)

Beyond the ingest boundary, the code builds protos *while operating on a model*
and discards them. `arena_internal_benchmark.cc` models three such sites, each
mapped to a real one. Unlike the binary-deserialize path, these are **allocation-
heavy on both build and destroy**, so the arena is an unambiguous win at every
size tested (Linux x86-64, GCC `-O3`, protobuf 3.21.12; 20k-node model, medians
of repeat runs):

| Pattern | real site | build | destroy | **total** |
|---|---|---:|---:|---:|
| **Tier 2** — whole-model deep clone (`CopyFrom`) | `ModelProto copy = model;` in `check_model` full_check (`onnx/checker.cc:1231`); version-converter round-trip (`onnx/common/ir_pb_converter.cc`) | ~38% | ~92% | **~64%** |
| **Tier 3a** — pass-lifetime `TypeProto` pool | `InferredTypes` `vector<unique_ptr<TypeProto>>` (`onnx/shape_inference/implementation.cc:300`) | ~50% | ~98% | **~69%** |
| **Tier 3b** — per-node `NodeProto` copy churn | throwaway `NodeProto copy_n(n)` per node (`implementation.cc:604`) | — | — | **~28–36%** |

Notes:

- **Tier 2** is the strongest self-contained opportunity: a deep clone allocates
  the entire sub-message tree and then frees it, so both phases benefit. It needs
  no API change — only the local `ModelProto copy` becomes arena-allocated. The
  `full_check` copy is especially attractive because it exists *purely* to be
  inferred-on and thrown away.
- **Tier 3a** is the biggest relative win (~69%) because it is pure small-message
  allocation with exactly arena lifetime (the `InferredTypes` vector already lives
  for the whole pass, so the arena changes nothing about lifetime — only the
  allocator). Capturing it means threading an arena into the inference pass.
- **Tier 3b** wins on speed (~28–36%) but **raises peak memory**: the real code
  frees each `copy_n` immediately (peak = one node), whereas an arena defers every
  free to end-of-pass (peak = all nodes). This is the speed *ceiling*; whether it
  is worth the memory depends on the workload. The `total_only` column reflects
  that build and destroy are not separable here (heap frees interleave into the
  loop).

### Priority, combining both benchmarks

1. **Text `parse_model` binding** (ingest) and **Tier 2 whole-model clones** —
   robust ~40–65% wins, contained local changes, no API/behavior change.
2. **Tier 3a inference type pool** — largest relative win, but needs an arena
   threaded through the shape-inference pass; do it if inference shows up in
   profiles.
3. **Binary ingest bindings** and **Tier 3b node churn** — conditional: the former
   only helps large models, the latter trades memory for speed. Gate on size /
   measure first.

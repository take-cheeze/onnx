// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Micro-benchmark for the "internal transient proto" arena candidates -- the
// Tier-2 and Tier-3 sites that build protos while operating on a model rather
// than at the ingest boundary. Each measurement models a specific real site:
//
//   * Tier 2  whole-model deep clone (CopyFrom)
//       maps to `ModelProto copy = model;` in checker::check_model full_check
//       (onnx/checker.cc) and the version converter's PrepareOutput/Export
//       round-trip (onnx/common/ir_pb_converter.cc).
//
//   * Tier 3a  pass-lifetime pool of small messages
//       maps to InferredTypes, which holds a std::vector<unique_ptr<TypeProto>>
//       for the duration of a FunctionProto inference pass
//       (onnx/shape_inference/implementation.cc:300). Same lifetime as an
//       arena, so it is a clean apples-to-apples candidate.
//
//   * Tier 3b  per-iteration temporary churn
//       maps to the throwaway `NodeProto copy_n(n)` created for every node
//       during attribute binding
//       (onnx/shape_inference/implementation.cc:604). NOTE the trade-off: the
//       real code frees each copy immediately (peak = one node), whereas an
//       arena defers all frees to end-of-pass (peak = all nodes). This measures
//       the speed ceiling, at the cost of higher peak memory.
//
// Depends only on public headers and std::chrono -- no new dependency.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "google/protobuf/arena.h"
#include "onnx/defs/parser.h"
#include "onnx/onnx_pb.h"

namespace {

using ONNX_NAMESPACE::ModelProto;
using ONNX_NAMESPACE::NodeProto;
using ONNX_NAMESPACE::OnnxParser;
using ONNX_NAMESPACE::TypeProto;
using Clock = std::chrono::steady_clock;

double MillisSince(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

[[noreturn]] void Fail(const char* what) {
  std::fprintf(stderr, "%s\n", what);
  std::exit(1);
}

std::string GenerateModelText(int num_nodes, int num_initializers, int initializer_len) {
  std::string s;
  s.reserve(static_cast<size_t>(num_nodes) * 128 + static_cast<size_t>(num_initializers) * initializer_len * 8);
  s += "<ir_version: 7, opset_import: [ \"\" : 17 ]>\n";
  s += "agraph (float[N] X) => (float[N] Y)\n<\n";
  for (int i = 0; i < num_initializers; ++i) {
    s += "  float[";
    s += std::to_string(initializer_len);
    s += "] w";
    s += std::to_string(i);
    s += " = {";
    for (int j = 0; j < initializer_len; ++j) {
      if (j)
        s += ", ";
      s += std::to_string((i + j) % 7);
      s += ".5";
    }
    s += "}";
    if (i + 1 < num_initializers)
      s += ",";
    s += "\n";
  }
  s += ">\n{\n";
  std::string prev = "X";
  for (int i = 0; i < num_nodes; ++i) {
    std::string out = "t" + std::to_string(i);
    s += "  ";
    s += out;
    s += " = SomeOp(";
    s += prev;
    s += ") <alpha = 1.5, beta = 2, mode = \"linear\", axes = [0, 1, 2]>\n";
    prev = out;
  }
  s += "  Y = Identity(";
  s += prev;
  s += ")\n}\n";
  return s;
}

// A representative inferred type: a float tensor with a rank-4 symbolic shape,
// like the TypeProtos shape inference materializes.
TypeProto MakeSampleType() {
  TypeProto type;
  auto* tensor = type.mutable_tensor_type();
  tensor->set_elem_type(1 /* FLOAT */);
  auto* shape = tensor->mutable_shape();
  for (int i = 0; i < 4; ++i)
    shape->add_dim()->set_dim_value(1 + i);
  return type;
}

struct PhaseTimes {
  double build_ms = 0;
  double destroy_ms = 0;
};

struct Stats {
  double build_ms = 0;
  double destroy_ms = 0;
};

template <typename Fn, typename Input>
Stats Average(Fn&& fn, const Input& input, int iters) {
  Stats total;
  for (int i = 0; i < iters; ++i) {
    PhaseTimes t = fn(input);
    total.build_ms += t.build_ms;
    total.destroy_ms += t.destroy_ms;
  }
  total.build_ms /= iters;
  total.destroy_ms /= iters;
  return total;
}

void PrintTable(const char* title, const Stats& heap, const Stats& arena, bool total_only = false) {
  auto pct = [](double base, double other) { return (base - other) / base * 100.0; };
  std::printf("%s\n", title);
  if (total_only) {
    std::printf("%-10s %12s\n", "", "total (ms)");
    std::printf("%-10s %12.3f\n", "heap", heap.build_ms + heap.destroy_ms);
    std::printf("%-10s %12.3f\n", "arena", arena.build_ms + arena.destroy_ms);
    std::printf(
        "%-10s %11.1f%%\n\n", "faster", pct(heap.build_ms + heap.destroy_ms, arena.build_ms + arena.destroy_ms));
    return;
  }
  std::printf("%-10s %12s %12s %12s\n", "", "build (ms)", "destroy (ms)", "total (ms)");
  std::printf("%-10s %12.3f %12.3f %12.3f\n", "heap", heap.build_ms, heap.destroy_ms, heap.build_ms + heap.destroy_ms);
  std::printf(
      "%-10s %12.3f %12.3f %12.3f\n", "arena", arena.build_ms, arena.destroy_ms, arena.build_ms + arena.destroy_ms);
  std::printf(
      "%-10s %11.1f%% %11.1f%% %11.1f%%\n\n",
      "faster",
      pct(heap.build_ms, arena.build_ms),
      pct(heap.destroy_ms, arena.destroy_ms),
      pct(heap.build_ms + heap.destroy_ms, arena.build_ms + arena.destroy_ms));
}

// ---- Tier 2: whole-model deep clone (CopyFrom) ----------------------------

PhaseTimes CloneHeap(const ModelProto& src) {
  PhaseTimes t;
  auto start = Clock::now();
  auto* copy = new ModelProto(src); // deep copy
  t.build_ms = MillisSince(start);
  start = Clock::now();
  delete copy;
  t.destroy_ms = MillisSince(start);
  return t;
}

PhaseTimes CloneArena(const ModelProto& src) {
  PhaseTimes t;
  auto arena = std::make_unique<google::protobuf::Arena>();
  auto start = Clock::now();
  auto* copy = google::protobuf::Arena::CreateMessage<ModelProto>(arena.get());
  copy->CopyFrom(src);
  t.build_ms = MillisSince(start);
  start = Clock::now();
  arena.reset();
  t.destroy_ms = MillisSince(start);
  return t;
}

// ---- Tier 3a: pass-lifetime pool of small messages (InferredTypes) --------

PhaseTimes TypePoolHeap(int count) {
  const TypeProto sample = MakeSampleType();
  PhaseTimes t;
  auto start = Clock::now();
  std::vector<std::unique_ptr<TypeProto>> pool;
  pool.reserve(count);
  for (int i = 0; i < count; ++i)
    pool.emplace_back(std::make_unique<TypeProto>(sample));
  t.build_ms = MillisSince(start);
  start = Clock::now();
  pool.clear(); // frees all N TypeProtos individually
  t.destroy_ms = MillisSince(start);
  return t;
}

PhaseTimes TypePoolArena(int count) {
  const TypeProto sample = MakeSampleType();
  PhaseTimes t;
  auto arena = std::make_unique<google::protobuf::Arena>();
  auto start = Clock::now();
  std::vector<TypeProto*> pool;
  pool.reserve(count);
  for (int i = 0; i < count; ++i) {
    auto* p = google::protobuf::Arena::CreateMessage<TypeProto>(arena.get());
    p->CopyFrom(sample);
    pool.push_back(p);
  }
  t.build_ms = MillisSince(start);
  start = Clock::now();
  arena.reset(); // one bulk free
  t.destroy_ms = MillisSince(start);
  return t;
}

// ---- Tier 3b: per-iteration temporary churn (NodeProto copy_n) ------------

PhaseTimes NodeChurnHeap(const google::protobuf::RepeatedPtrField<NodeProto>& nodes) {
  PhaseTimes t;
  auto start = Clock::now();
  // Each copy is constructed and destroyed within its own scope, exactly like
  // `NodeProto copy_n(n);` per node -- peak memory is one node.
  for (const auto& n : nodes) {
    NodeProto copy_n(n);
    // Touch a field so the copy is not optimized away.
    if (copy_n.op_type().empty())
      Fail("unexpected empty op_type");
  }
  t.build_ms = MillisSince(start);
  t.destroy_ms = 0; // destruction is interleaved into the loop above
  return t;
}

PhaseTimes NodeChurnArena(const google::protobuf::RepeatedPtrField<NodeProto>& nodes) {
  PhaseTimes t;
  auto arena = std::make_unique<google::protobuf::Arena>();
  auto start = Clock::now();
  // All copies live on one pass-lifetime arena -- peak memory is all nodes.
  for (const auto& n : nodes) {
    auto* copy_n = google::protobuf::Arena::CreateMessage<NodeProto>(arena.get());
    copy_n->CopyFrom(n);
    if (copy_n->op_type().empty())
      Fail("unexpected empty op_type");
  }
  t.build_ms = MillisSince(start);
  start = Clock::now();
  arena.reset();
  t.destroy_ms = MillisSince(start);
  return t;
}

} // namespace

int main(int argc, char** argv) {
  int num_nodes = argc > 1 ? std::atoi(argv[1]) : 20000;
  int num_initializers = argc > 2 ? std::atoi(argv[2]) : 500;
  int initializer_len = argc > 3 ? std::atoi(argv[3]) : 64;
  int iters = argc > 4 ? std::atoi(argv[4]) : 20;

  ModelProto model;
  if (!OnnxParser::Parse(model, GenerateModelText(num_nodes, num_initializers, initializer_len)).IsOK())
    Fail("failed to parse model");
  const int node_count = model.graph().node_size();

  std::printf(
      "Model: %d graph nodes, %d initializers (len %d)\n"
      "Iterations: %d (plus 3 warmup)\n\n",
      node_count,
      num_initializers,
      initializer_len,
      iters);

  for (int i = 0; i < 3; ++i) {
    CloneHeap(model);
    CloneArena(model);
    TypePoolHeap(node_count);
    TypePoolArena(node_count);
    NodeChurnHeap(model.graph().node());
    NodeChurnArena(model.graph().node());
  }

  PrintTable(
      "[Tier 2] whole-model deep clone (checker full_check `ModelProto copy = model`)",
      Average(CloneHeap, model, iters),
      Average(CloneArena, model, iters));

  PrintTable(
      "[Tier 3a] pass-lifetime TypeProto pool (shape-inference InferredTypes)",
      Average(TypePoolHeap, node_count, iters),
      Average(TypePoolArena, node_count, iters));

  PrintTable(
      "[Tier 3b] per-node NodeProto copy churn (attribute binding; arena raises peak memory)",
      Average(NodeChurnHeap, model.graph().node(), iters),
      Average(NodeChurnArena, model.graph().node(), iters),
      /*total_only=*/true);

  return 0;
}

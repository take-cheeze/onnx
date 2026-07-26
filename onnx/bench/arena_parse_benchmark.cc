// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Micro-benchmark: building a ModelProto on a google::protobuf::Arena versus on
// the heap (the default), for the two ways ONNX ingests a model in C++:
//
//   * text parsing      -- OnnxParser::Parse (onnx/defs/parser.h)
//   * binary deserialize -- ParseProtoFromBytes (onnx/proto_utils.h), the path
//                           every onnx/cpp2py_export.cc binding takes when it
//                           receives serialized bytes from Python.
//
// Both entry points fill a caller-supplied root message via mutable_*/add_*
// (or ParseFromCodedStream), and protobuf propagates the root's arena to every
// sub-message. So creating the root with Arena::CreateMessage<ModelProto> makes
// the whole node/attribute/tensor/value-info tree arena-allocated, with no
// change to the parser or the deserializer. For each ingest path the benchmark
// times the two phases that dominate a transient model's lifetime:
//
//   * build   -- constructing/deserializing the tree (many small allocations)
//   * destroy -- tearing it down (per-object dtors + free vs. one bulk free)
//
// It depends only on public headers and std::chrono, so it pulls in no new
// third-party dependency.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>

#include "google/protobuf/arena.h"
#include "onnx/defs/parser.h"
#include "onnx/onnx_pb.h"
#include "onnx/proto_utils.h"

namespace {

using ONNX_NAMESPACE::ModelProto;
using ONNX_NAMESPACE::OnnxParser;
using ONNX_NAMESPACE::ParseProtoFromBytes;
using Clock = std::chrono::steady_clock;

double MillisSince(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

[[noreturn]] void Fail(const char* what) {
  std::fprintf(stderr, "%s\n", what);
  std::exit(1);
}

// Build an ONNX text model with `num_nodes` nodes. Each node carries a handful
// of attributes and each graph carries `num_initializers` tensor initializers,
// so the resulting ModelProto is a deep tree of many small sub-messages -- the
// regime where the allocator choice actually matters.
std::string GenerateModelText(int num_nodes, int num_initializers, int initializer_len) {
  std::string s;
  s.reserve(static_cast<size_t>(num_nodes) * 128 + static_cast<size_t>(num_initializers) * initializer_len * 8);
  s += "<ir_version: 7, opset_import: [ \"\" : 17 ]>\n";
  s += "agraph (float[N] X) => (float[N] Y)\n<\n";
  // Initializers: each is a TensorProto with a repeated float_data field.
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
  // Nodes: chained so outputs feed inputs; each has float/int/string attributes.
  std::string prev = "X";
  for (int i = 0; i < num_nodes; ++i) {
    std::string out = "t" + std::to_string(i);
    s += "  ";
    s += out;
    s += " = SomeOp";
    s += "(";
    s += prev;
    s += ") <alpha = 1.5, beta = 2, mode = \"linear\", axes = [0, 1, 2]>\n";
    prev = out;
  }
  s += "  Y = Identity(";
  s += prev;
  s += ")\n}\n";
  return s;
}

struct PhaseTimes {
  double build_ms = 0;
  double destroy_ms = 0;
};

// ---- Text parsing (OnnxParser::Parse) -------------------------------------

PhaseTimes RunTextHeap(const std::string& text) {
  PhaseTimes t;
  auto* model = new ModelProto();
  auto start = Clock::now();
  OnnxParser parser(text);
  auto status = parser.Parse(*model);
  t.build_ms = MillisSince(start);
  if (!status.IsOK())
    Fail(status.ErrorMessage().c_str());
  start = Clock::now();
  delete model;
  t.destroy_ms = MillisSince(start);
  return t;
}

PhaseTimes RunTextArena(const std::string& text) {
  PhaseTimes t;
  auto arena = std::make_unique<google::protobuf::Arena>();
  auto* model = google::protobuf::Arena::CreateMessage<ModelProto>(arena.get());
  auto start = Clock::now();
  OnnxParser parser(text);
  auto status = parser.Parse(*model);
  t.build_ms = MillisSince(start);
  if (!status.IsOK())
    Fail(status.ErrorMessage().c_str());
  // Freeing the arena reclaims the whole ModelProto tree in one shot.
  start = Clock::now();
  arena.reset();
  t.destroy_ms = MillisSince(start);
  return t;
}

// ---- Binary deserialize (ParseProtoFromBytes) -----------------------------
// This is the path every cpp2py_export.cc binding takes on serialized bytes.
// ParseProtoFromBytes already works on an arena-allocated root unchanged; only
// the allocation of the root differs between the two functions below.

PhaseTimes RunBinaryHeap(const std::string& bytes) {
  PhaseTimes t;
  auto* model = new ModelProto();
  auto start = Clock::now();
  if (!ParseProtoFromBytes(model, bytes.data(), bytes.size()))
    Fail("binary deserialize failed");
  t.build_ms = MillisSince(start);
  start = Clock::now();
  delete model;
  t.destroy_ms = MillisSince(start);
  return t;
}

PhaseTimes RunBinaryArena(const std::string& bytes) {
  PhaseTimes t;
  auto arena = std::make_unique<google::protobuf::Arena>();
  auto* model = google::protobuf::Arena::CreateMessage<ModelProto>(arena.get());
  auto start = Clock::now();
  if (!ParseProtoFromBytes(model, bytes.data(), bytes.size()))
    Fail("binary deserialize failed");
  t.build_ms = MillisSince(start);
  start = Clock::now();
  arena.reset();
  t.destroy_ms = MillisSince(start);
  return t;
}

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

void PrintTable(const char* title, const Stats& heap, const Stats& arena) {
  auto pct = [](double base, double other) { return (base - other) / base * 100.0; };
  std::printf("%s\n", title);
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

} // namespace

int main(int argc, char** argv) {
  int num_nodes = argc > 1 ? std::atoi(argv[1]) : 20000;
  int num_initializers = argc > 2 ? std::atoi(argv[2]) : 500;
  int initializer_len = argc > 3 ? std::atoi(argv[3]) : 64;
  int iters = argc > 4 ? std::atoi(argv[4]) : 20;

  std::string text = GenerateModelText(num_nodes, num_initializers, initializer_len);

  // Produce the binary form once, from the same model, so the two ingest paths
  // build an identical ModelProto tree.
  std::string bytes;
  {
    ModelProto reference;
    if (!OnnxParser::Parse(reference, text).IsOK())
      Fail("failed to parse reference model");
    if (!reference.SerializeToString(&bytes))
      Fail("failed to serialize reference model");
  }

  std::printf(
      "Model: %d nodes, %d initializers (len %d)\n"
      "  text size   %.2f MiB\n"
      "  binary size %.2f MiB\n"
      "Iterations: %d (plus 3 warmup)\n\n",
      num_nodes,
      num_initializers,
      initializer_len,
      text.size() / 1048576.0,
      bytes.size() / 1048576.0,
      iters);

  // Warmup (parser static tables, page faults, cache).
  for (int i = 0; i < 3; ++i) {
    RunTextHeap(text);
    RunTextArena(text);
    RunBinaryHeap(bytes);
    RunBinaryArena(bytes);
  }

  PrintTable("[text parse]  OnnxParser::Parse", Average(RunTextHeap, text, iters), Average(RunTextArena, text, iters));
  PrintTable(
      "[binary]      ParseProtoFromBytes (cpp2py boundary path)",
      Average(RunBinaryHeap, bytes, iters),
      Average(RunBinaryArena, bytes, iters));
  return 0;
}

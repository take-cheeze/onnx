// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Micro-benchmark: parsing an ONNX text model into an arena-allocated
// ModelProto versus a heap-allocated (default) ModelProto.
//
// The ONNX text parser (OnnxParser::Parse) builds the whole ModelProto tree by
// calling mutable_*/add_* on the caller-supplied root message. protobuf
// propagates the root message's arena to every sub-message created that way, so
// creating the root on a google::protobuf::Arena makes the entire node /
// attribute / tensor / value-info tree arena-allocated. This benchmark measures
// what that buys us for two phases that dominate the lifetime of a freshly
// parsed model:
//
//   * parse       -- constructing the tree (thousands of small allocations)
//   * destruction -- tearing it down (per-object dtors + free vs. one bulk free)
//
// It deliberately depends only on the public parser API and std::chrono so it
// pulls in no new third-party dependency.

#include <chrono>
#include <cstdint>
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
using ONNX_NAMESPACE::OnnxParser;
using Clock = std::chrono::steady_clock;

double MillisSince(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
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
  double parse_ms = 0;
  double destroy_ms = 0;
};

// Parse into a plain heap ModelProto and time construction + destruction.
PhaseTimes RunHeap(const std::string& text) {
  PhaseTimes t;
  auto* model = new ModelProto();
  auto start = Clock::now();
  OnnxParser parser(text);
  auto status = parser.Parse(*model);
  t.parse_ms = MillisSince(start);
  if (!status.IsOK()) {
    std::fprintf(stderr, "parse failed: %s\n", status.ErrorMessage().c_str());
    std::exit(1);
  }
  start = Clock::now();
  delete model;
  t.destroy_ms = MillisSince(start);
  return t;
}

// Parse into an arena-allocated ModelProto and time construction + destruction.
PhaseTimes RunArena(const std::string& text) {
  PhaseTimes t;
  auto arena = std::make_unique<google::protobuf::Arena>();
  auto* model = google::protobuf::Arena::CreateMessage<ModelProto>(arena.get());
  auto start = Clock::now();
  OnnxParser parser(text);
  auto status = parser.Parse(*model);
  t.parse_ms = MillisSince(start);
  if (!status.IsOK()) {
    std::fprintf(stderr, "parse failed: %s\n", status.ErrorMessage().c_str());
    std::exit(1);
  }
  // Freeing the arena reclaims the whole ModelProto tree in one shot.
  start = Clock::now();
  arena.reset();
  t.destroy_ms = MillisSince(start);
  return t;
}

struct Stats {
  double parse_ms = 0;
  double destroy_ms = 0;
};

template <typename Fn>
Stats Average(Fn&& fn, const std::string& text, int iters) {
  Stats total;
  for (int i = 0; i < iters; ++i) {
    PhaseTimes t = fn(text);
    total.parse_ms += t.parse_ms;
    total.destroy_ms += t.destroy_ms;
  }
  total.parse_ms /= iters;
  total.destroy_ms /= iters;
  return total;
}

} // namespace

int main(int argc, char** argv) {
  int num_nodes = argc > 1 ? std::atoi(argv[1]) : 20000;
  int num_initializers = argc > 2 ? std::atoi(argv[2]) : 500;
  int initializer_len = argc > 3 ? std::atoi(argv[3]) : 64;
  int iters = argc > 4 ? std::atoi(argv[4]) : 20;

  std::string text = GenerateModelText(num_nodes, num_initializers, initializer_len);
  std::printf(
      "Model: %d nodes, %d initializers (len %d), text size %.2f MiB\n",
      num_nodes,
      num_initializers,
      initializer_len,
      text.size() / 1048576.0);
  std::printf("Iterations: %d (plus 3 warmup)\n\n", iters);

  // Warmup (parser static tables, page faults, cache).
  for (int i = 0; i < 3; ++i) {
    RunHeap(text);
    RunArena(text);
  }

  Stats heap = Average(RunHeap, text, iters);
  Stats arena = Average(RunArena, text, iters);

  auto pct = [](double base, double other) { return (base - other) / base * 100.0; };

  std::printf("%-10s %12s %12s %12s\n", "", "parse (ms)", "destroy (ms)", "total (ms)");
  std::printf("%-10s %12.3f %12.3f %12.3f\n", "heap", heap.parse_ms, heap.destroy_ms, heap.parse_ms + heap.destroy_ms);
  std::printf(
      "%-10s %12.3f %12.3f %12.3f\n", "arena", arena.parse_ms, arena.destroy_ms, arena.parse_ms + arena.destroy_ms);
  std::printf(
      "%-10s %11.1f%% %11.1f%% %11.1f%%\n",
      "faster",
      pct(heap.parse_ms, arena.parse_ms),
      pct(heap.destroy_ms, arena.destroy_ms),
      pct(heap.parse_ms + heap.destroy_ms, arena.parse_ms + arena.destroy_ms));
  return 0;
}

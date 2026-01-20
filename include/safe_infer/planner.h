#pragma once

#include "safe_infer/graph.h"
#include <vector>

namespace safe_infer {

// Returns a topological execution order of node IDs
// Throws std::domain_error the graph is invalid or cyclic
std::vector<NodeId> plan_execution(const Graph& g);

} // namespace safe_infer

/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <utilities/cuda_helpers.cuh>
#include "../routing_helpers.cuh"
#include "routing/fleet_info.hpp"

namespace cuopt {
namespace routing {
namespace detail {

// Per-vehicle hard limit on total number of service-node visits in a route.
//
// State layout:
//   count_increment : 1 for service nodes, 0 for depot nodes (set at creation)
//   fwd_count[k]    : number of service-node visits in prefix [0..k] (inclusive)
//   bwd_count[k]    : number of service-node visits in suffix [k..n] (inclusive)
//
// Forward and backward sums are over disjoint sets (prefix [0..k] and suffix
// [k+1..n]) so combine is plain addition — no double-counting correction is
// needed at the boundary, unlike the capacity dimension where the in-transit
// state is "seen" by both sides at the join.
template <typename i_t, typename f_t>
class route_size_node_t {
 public:
  HDI route_size_node_t() = default;

  // Per-node contribution to the count. 1 for service nodes, 0 for depot.
  i_t count_increment{0};
  // Propagated counts, set by calculate_forward / calculate_backward.
  i_t fwd_count{0};
  i_t bwd_count{0};

  void HDI calculate_forward(route_size_node_t& next, [[maybe_unused]] f_t dummy = 0) const noexcept
  {
    next.fwd_count = fwd_count + next.count_increment;
  }

  void HDI calculate_backward(route_size_node_t& prev,
                              [[maybe_unused]] f_t dummy = 0) const noexcept
  {
    prev.bwd_count = prev.count_increment + bwd_count;
  }

  // Combined excess for a route split between prefix ending at `prev` and
  // suffix beginning at `next`. fwd_count and bwd_count are over disjoint
  // index ranges, so their sum equals the route's total service-node count.
  static i_t HDI combine(const route_size_node_t& prev,
                         const route_size_node_t& next,
                         const VehicleInfo<f_t>& vehicle_info,
                         [[maybe_unused]] const f_t dummy = 0.) noexcept
  {
    return max(0, prev.fwd_count + next.bwd_count - vehicle_info.max_route_size);
  }

  HDI double forward_excess(const VehicleInfo<f_t>& vehicle_info) const noexcept
  {
    return max(0, fwd_count - vehicle_info.max_route_size);
  }

  HDI double backward_excess(const VehicleInfo<f_t>& vehicle_info) const noexcept
  {
    return max(0, bwd_count - vehicle_info.max_route_size);
  }

  HDI bool forward_feasible(const VehicleInfo<f_t>& vehicle_info,
                            const double weight    = 1.,
                            const f_t excess_limit = 0.) const noexcept
  {
    return forward_excess(vehicle_info) * weight <= excess_limit;
  }

  HDI bool backward_feasible(const VehicleInfo<f_t>& vehicle_info,
                             const double weight    = 1.,
                             const f_t excess_limit = 0.) const noexcept
  {
    return backward_excess(vehicle_info) * weight <= excess_limit;
  }

  // get_cost is called on `next` with `prev` passed in — mirrors combine().
  // prev.fwd_count is reconstructed from this->fwd_count by removing this
  // node's own contribution, mirroring capacity_node_t::get_cost.
  template <bool is_device = true>
  HDI void get_cost(const route_size_node_t& prev,
                    const VehicleInfo<f_t, is_device>& vehicle_info,
                    [[maybe_unused]] const route_size_dimension_info_t& dim_info,
                    [[maybe_unused]] objective_cost_t& obj_cost,
                    infeasible_cost_t& inf_cost) const noexcept
  {
    const i_t prev_fwd_count    = fwd_count - count_increment;
    inf_cost[dim_t::ROUTE_SIZE] = max(0, prev_fwd_count + bwd_count - vehicle_info.max_route_size);
  }
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt

/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include "routing/dimensions.cuh"
#include "routing/vehicle_info.hpp"

#include <utilities/cuda_helpers.cuh>

#include <cstdint>

namespace cuopt {
namespace routing {
namespace detail {

// Order-type incompatibility ("INCOMPAT") dimension — event-based.
//
// Each order carries a uint64 bitmask of tags (up to max_incompat_tags = 32
// bits used). A symmetric n_tags x n_tags float matrix M with M[t][t] = 0
// assigns a weight to each pair of tag IDs that are incompatible.
//
// Cost is accumulated at pickup events ONLY:
//   pickup_cost(k) = fwd_count[k-1] . (M . tag_mask[k])
// where fwd_count[t] is the number of currently-active orders carrying tag t
// (BEFORE the new pickup is applied — order matters; see calculate_forward).
//
// Combine across a stitched prefix + suffix:
//   combine = prev.fwd_excess
//           + next.bwd_excess
//           + prev.fwd_count . M . next.bwd_pickup_tag_sum
// The cross-term boosts each pickup in the suffix by prev's carry-over.
//
// Backward state stores suffix-LOCAL bookkeeping (fresh-start at suffix
// origin). bwd_excess may be NEGATIVE for suffixes that begin with
// deliveries; this is mathematically correct and is restored to the right
// total via combine. Do NOT clamp it.
template <typename i_t, typename f_t>
class incompat_node_t {
 public:
  HDI incompat_node_t(const incompat_dimension_info_t& dim_info)
    : n_tags(dim_info.n_tags), incompat_matrix(dim_info.incompat_matrix)
  {
  }

  HDI incompat_node_t() = delete;

  //
  // Problem-level config (copied from dim_info at construction).
  //
  // Matrix is row-major, size n_tags*n_tags, symmetric with diagonal = 0.
  int n_tags                   = 0;
  float const* incompat_matrix = nullptr;

  //
  // Per-node fixed data (populated by set_route_data from order_info).
  //
  // tag_mask: bits set for each tag the order carries. 0 at depots.
  // delta:    +1 at pickup, -1 at delivery, 0 at depots.
  uint64_t tag_mask = 0;
  int delta         = 0;

  //
  // Forward state.
  //
  // fwd_count[t] = number of active orders with tag t after processing this
  // node. fwd_excess = sum over pickups in [0..this] of pickup_cost.
  int fwd_count[max_incompat_tags] = {0};
  double fwd_excess                = 0.0;

  //
  // Backward state (suffix-local fresh-start convention).
  //
  // bwd_pickup_tag_sum[t] = sum over pickups in [this..N] of bit_t(tag_mask).
  // bwd_excess            = sum over pickups in [this..N] of local pickup_cost
  //                         measured with local_count starting at 0 just
  //                         before `this`. May be negative when suffix starts
  //                         with deliveries.
  int bwd_pickup_tag_sum[max_incompat_tags] = {0};
  double bwd_excess                         = 0.0;

  //
  // pickup_cost_at_this_with_count_before(count_before):
  //   evaluated only if delta == +1; sum_{t in tag_mask} count_before . M_row_t
  //   = count_before . (M . tag_mask).
  //
  // The matrix lookup iterates over set bits of tag_mask; cost is
  // O(popcount(tag_mask) * n_tags).
  HDI double pickup_cost_with_count_before(const int* count_before) const noexcept
  {
    if (delta != +1) return 0.0;
    double accum = 0.0;
    // Iterate set bits of tag_mask via linear scan (n_tags <= 32 — cheap).
    for (int j = 0; j < n_tags; ++j) {
      if (((tag_mask >> j) & 1ULL) == 0) continue;
      // contribute sum_t count_before[t] * M[t][j]  (M is symmetric)
      const float* mat_col = incompat_matrix + j;  // strided across rows
      for (int t = 0; t < n_tags; ++t) {
        accum += static_cast<double>(count_before[t]) * mat_col[t * n_tags];
      }
    }
    return accum;
  }

  //
  // Forward propagation: this -> next.
  //
  // Order-critical: compute pickup cost from `this.fwd_count` (the state
  // BEFORE the new pickup is applied), THEN update next.fwd_count by adding
  // this node's contribution. Reversing this would self-pair the new order.
  HDI void calculate_forward(incompat_node_t& next,
                             [[maybe_unused]] f_t arc_value = 0) const noexcept
  {
    // 1) pickup_cost uses `this.fwd_count`, which is the state at `this` (the
    //    last position before `next`). This is "count BEFORE next is applied".
    double incr     = (next.delta == +1) ? next.pickup_cost_with_count_before(fwd_count) : 0.0;
    next.fwd_excess = fwd_excess + incr;

    // 2) Now update next.fwd_count.
    for (int t = 0; t < n_tags; ++t) {
      next.fwd_count[t] = fwd_count[t];
      if (next.delta != 0 && ((next.tag_mask >> t) & 1ULL)) { next.fwd_count[t] += next.delta; }
    }
  }

  //
  // Backward propagation: this (suffix [k+1..N]) -> prev (prepend node k,
  // suffix [k..N]).
  //
  // CRITICAL ORDER: boost is computed from `this.bwd_pickup_tag_sum` (the OLD
  // suffix's pickup-sum). If you instead use the already-updated
  // `prev.bwd_pickup_tag_sum`, a prepended pickup would self-pair against
  // its own tags. Boost must be computed BEFORE updating the sum.
  HDI void calculate_backward(incompat_node_t& prev,
                              [[maybe_unused]] f_t arc_value = 0) const noexcept
  {
    // 1) boost = prev.delta * (bit_vec(prev.tag_mask) . M . this.bwd_pickup_tag_sum)
    //          = sign-of-prev.delta * sum_{t in prev.tag_mask, s} M[t][s] *
    //            this.bwd_pickup_tag_sum[s]
    double boost = 0.0;
    if (prev.delta != 0) {
      for (int t = 0; t < n_tags; ++t) {
        if (((prev.tag_mask >> t) & 1ULL) == 0) continue;
        const float* mat_row = incompat_matrix + t * n_tags;
        for (int s = 0; s < n_tags; ++s) {
          boost += mat_row[s] * static_cast<double>(bwd_pickup_tag_sum[s]);
        }
      }
      if (prev.delta == -1) boost = -boost;
    }
    prev.bwd_excess = bwd_excess + boost;

    // 2) Update suffix pickup sum: only pickups (delta == +1) contribute.
    for (int t = 0; t < n_tags; ++t) {
      prev.bwd_pickup_tag_sum[t] = bwd_pickup_tag_sum[t];
      if (prev.delta == +1 && ((prev.tag_mask >> t) & 1ULL)) { prev.bwd_pickup_tag_sum[t] += 1; }
    }
  }

  HDI double forward_excess([[maybe_unused]] const VehicleInfo<f_t>& vehicle_info) const noexcept
  {
    // Cumulative excess up to and including this node; non-negative for valid
    // forward sweeps with M >= 0.
    return fwd_excess;
  }

  HDI double backward_excess([[maybe_unused]] const VehicleInfo<f_t>& vehicle_info) const noexcept
  {
    // bwd_excess is a suffix-local algebraic accumulator and may be negative
    // for suffixes starting with deliveries. Surface as-is; the framework's
    // weighted-excess machinery is only meaningful when applied to full-route
    // forward excess for this dim.
    return bwd_excess;
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

  //
  // Combine: total INCOMPAT cost of the stitched prefix [..prev] + suffix [next..].
  //
  // O(n_tags^2) — bwd_pickup_tag_sum is dense; no popcount shortcut. At
  // n_tags = 32 this is ~1024 multiply-adds. The matrix is symmetric so we
  // iterate row-major over M.
  template <bool is_device = true>
  static HDI double combine(const incompat_node_t& prev,
                            const incompat_node_t& next,
                            [[maybe_unused]] const VehicleInfo<f_t, is_device>& vehicle_info,
                            [[maybe_unused]] f_t arc_value = 0) noexcept
  {
    const int n             = prev.n_tags;
    const float* M          = prev.incompat_matrix;
    const int* p_count      = prev.fwd_count;
    const int* n_pickup_sum = next.bwd_pickup_tag_sum;

    // cross = sum_{i,j} p_count[i] * M[i][j] * n_pickup_sum[j]
    double cross = 0.0;
    for (int i = 0; i < n; ++i) {
      if (p_count[i] == 0) continue;
      const float* mat_row = M + i * n;
      double partial       = 0.0;
      for (int j = 0; j < n; ++j) {
        partial += mat_row[j] * static_cast<double>(n_pickup_sum[j]);
      }
      cross += static_cast<double>(p_count[i]) * partial;
    }
    return prev.fwd_excess + next.bwd_excess + cross;
  }

  template <bool is_device = true>
  HDI void get_cost([[maybe_unused]] const incompat_node_t& prev_node,
                    const VehicleInfo<f_t, is_device>& vehicle_info,
                    [[maybe_unused]] const incompat_dimension_info_t& dim_info,
                    [[maybe_unused]] objective_cost_t& obj_cost,
                    infeasible_cost_t& inf_cost) const noexcept
  {
    // Per-node-pair view of the total route cost: equivalent to
    // combine(prev_node, *this) per §Math (route_t::compute_cost reads
    // fwd_excess[N] directly, but here in the local-search delta path we
    // expose the same value via combine).
    inf_cost[dim_t::INCOMPAT] = combine(prev_node, *this, vehicle_info, static_cast<f_t>(0));
  }
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt

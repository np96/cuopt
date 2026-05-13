/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <utilities/cuda_helpers.cuh>
#include "../node/incompat_node.cuh"
#include "../routing_helpers.cuh"
#include "../solution/solution_handle.cuh"
#include "routing/routing_helpers.cuh"

#include <raft/core/handle.hpp>
#include <raft/core/nvtx.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/tuple.h>

#include <cstdint>

namespace cuopt {
namespace routing {
namespace detail {

// Storage + device view for the INCOMPAT (order-tag incompatibility) dimension.
//
// Stride convention: each per-position array is sized `stride = N + 1`, where
// N = end-depot index (= n_nodes_route). Per-tag arrays are row-major
// [tag][position] = tag * stride + position.
//
// Memory ownership: rmm::device_uvector members own device storage; view_t
// borrows it via raft::device_span (matches capacity_route_t).
template <typename i_t, typename f_t>
class incompat_route_t {
 public:
  incompat_route_t(solution_handle_t<i_t, f_t> const* sol_handle_,
                   const incompat_dimension_info_t& dim_info_)
    : tag_mask(0, sol_handle_->get_stream()),
      delta(0, sol_handle_->get_stream()),
      fwd_count(0, sol_handle_->get_stream()),
      fwd_excess(0, sol_handle_->get_stream()),
      bwd_pickup_tag_sum(0, sol_handle_->get_stream()),
      bwd_excess(0, sol_handle_->get_stream()),
      dim_info(dim_info_)
  {
    raft::common::nvtx::range fun_scope("zero incompat route ctr");
  }

  incompat_route_t(const incompat_route_t& other, solution_handle_t<i_t, f_t> const* sol_handle_)
    : tag_mask(other.tag_mask, sol_handle_->get_stream()),
      delta(other.delta, sol_handle_->get_stream()),
      fwd_count(other.fwd_count, sol_handle_->get_stream()),
      fwd_excess(other.fwd_excess, sol_handle_->get_stream()),
      bwd_pickup_tag_sum(other.bwd_pickup_tag_sum, sol_handle_->get_stream()),
      bwd_excess(other.bwd_excess, sol_handle_->get_stream()),
      dim_info(other.dim_info)
  {
    raft::common::nvtx::range fun_scope("incompat route copy_ctr");
  }

  incompat_route_t& operator=(incompat_route_t&& other) = default;

  void resize(i_t max_nodes_per_route, rmm::cuda_stream_view stream)
  {
    tag_mask.resize(max_nodes_per_route, stream);
    delta.resize(max_nodes_per_route, stream);
    fwd_excess.resize(max_nodes_per_route, stream);
    bwd_excess.resize(max_nodes_per_route, stream);
    fwd_count.resize(dim_info.n_tags * max_nodes_per_route, stream);
    bwd_pickup_tag_sum.resize(dim_info.n_tags * max_nodes_per_route, stream);
  }

  struct view_t {
    bool is_empty() const { return tag_mask.empty(); }

    DI incompat_node_t<i_t, f_t> get_node(i_t idx) const
    {
      incompat_node_t<i_t, f_t> n(dim_info);
      n.tag_mask   = tag_mask[idx];
      n.delta      = delta[idx];
      n.fwd_excess = fwd_excess[idx];
      n.bwd_excess = bwd_excess[idx];
      for (int t = 0; t < dim_info.n_tags; ++t) {
        n.fwd_count[t]          = fwd_count[t * stride + idx];
        n.bwd_pickup_tag_sum[t] = bwd_pickup_tag_sum[t * stride + idx];
      }
      return n;
    }

    DI void set_node(i_t idx, const incompat_node_t<i_t, f_t>& node)
    {
      tag_mask[idx] = node.tag_mask;
      delta[idx]    = node.delta;
      set_forward_data(idx, node);
      set_backward_data(idx, node);
    }

    DI void set_forward_data(i_t idx, const incompat_node_t<i_t, f_t>& node)
    {
      fwd_excess[idx] = node.fwd_excess;
      for (int t = 0; t < dim_info.n_tags; ++t) {
        fwd_count[t * stride + idx] = node.fwd_count[t];
      }
    }

    DI void set_backward_data(i_t idx, const incompat_node_t<i_t, f_t>& node)
    {
      bwd_excess[idx] = node.bwd_excess;
      for (int t = 0; t < dim_info.n_tags; ++t) {
        bwd_pickup_tag_sum[t * stride + idx] = node.bwd_pickup_tag_sum[t];
      }
    }

    DI void copy_forward_data(const view_t& orig_route, i_t start_idx, i_t end_idx, i_t write_start)
    {
      i_t size = end_idx - start_idx;
      block_copy(fwd_excess.subspan(write_start), orig_route.fwd_excess.subspan(start_idx), size);

      i_t my_offset   = write_start;
      i_t orig_offset = start_idx;
      for (int t = 0; t < dim_info.n_tags; ++t) {
        block_copy(fwd_count.subspan(my_offset), orig_route.fwd_count.subspan(orig_offset), size);
        my_offset += this->stride;
        orig_offset += orig_route.stride;
      }
    }

    DI void copy_backward_data(const view_t& orig_route,
                               i_t start_idx,
                               i_t end_idx,
                               i_t write_start)
    {
      i_t size = end_idx - start_idx;
      block_copy(bwd_excess.subspan(write_start), orig_route.bwd_excess.subspan(start_idx), size);

      i_t my_offset   = write_start;
      i_t orig_offset = start_idx;
      for (int t = 0; t < dim_info.n_tags; ++t) {
        block_copy(bwd_pickup_tag_sum.subspan(my_offset),
                   orig_route.bwd_pickup_tag_sum.subspan(orig_offset),
                   size);
        my_offset += this->stride;
        orig_offset += orig_route.stride;
      }
    }

    DI void copy_fixed_route_data(const view_t& orig_route,
                                  i_t from_idx,
                                  i_t to_idx,
                                  i_t write_start)
    {
      i_t size = to_idx - from_idx;
      block_copy(tag_mask.subspan(write_start), orig_route.tag_mask.subspan(from_idx), size);
      block_copy(delta.subspan(write_start), orig_route.delta.subspan(from_idx), size);
    }

    DI void compute_cost([[maybe_unused]] const VehicleInfo<f_t>& vehicle_info,
                         const i_t n_nodes,
                         [[maybe_unused]] objective_cost_t& obj_cost,
                         infeasible_cost_t& inf_cost) const noexcept
    {
      // Total INCOMPAT cost = fwd_excess at the end-depot position
      // (= index n_nodes per cuOpt convention; arrays are sized n_nodes + 1).
      // For non-PDP routes fwd_count[N] need not be zero — but the cost
      // (which is what we report) is well-defined regardless.
      inf_cost[dim_t::INCOMPAT] = fwd_excess[n_nodes];
    }

    // Shared-memory allocator for a single route. Allocates highest-alignment
    // arrays first (double, uint64_t) to avoid padding overhead, then int
    // arrays.
    static DI thrust::tuple<view_t, i_t*> create_shared_route(
      i_t* shmem, const incompat_dimension_info_t dim_info, i_t n_nodes_route)
    {
      view_t v;
      v.dim_info = dim_info;
      v.stride   = n_nodes_route + 1;

      size_t per_pos        = static_cast<size_t>(v.stride);
      size_t per_tag_arr_sz = static_cast<size_t>(v.stride) * dim_info.n_tags;

      i_t* sh_ptr = shmem;

      // 8-byte alignment first.
      thrust::tie(v.fwd_excess, sh_ptr) = wrap_ptr_as_span<double>(sh_ptr, per_pos);
      thrust::tie(v.bwd_excess, sh_ptr) = wrap_ptr_as_span<double>(sh_ptr, per_pos);
      thrust::tie(v.tag_mask, sh_ptr)   = wrap_ptr_as_span<uint64_t>(sh_ptr, per_pos);

      // 4-byte alignment next.
      thrust::tie(v.fwd_count, sh_ptr)          = wrap_ptr_as_span<i_t>(sh_ptr, per_tag_arr_sz);
      thrust::tie(v.bwd_pickup_tag_sum, sh_ptr) = wrap_ptr_as_span<i_t>(sh_ptr, per_tag_arr_sz);
      thrust::tie(v.delta, sh_ptr)              = wrap_ptr_as_span<i_t>(sh_ptr, per_pos);

      return thrust::make_tuple(v, sh_ptr);
    }

    raft::device_span<double> fwd_excess;
    raft::device_span<double> bwd_excess;
    raft::device_span<uint64_t> tag_mask;
    raft::device_span<i_t> fwd_count;           // [n_tags * stride], row-major by tag
    raft::device_span<i_t> bwd_pickup_tag_sum;  // [n_tags * stride], row-major by tag
    raft::device_span<i_t> delta;               // per-position; values in {-1, 0, +1}
    incompat_dimension_info_t dim_info;
    i_t stride;
  };

  view_t view()
  {
    view_t v;
    v.tag_mask   = raft::device_span<uint64_t>{tag_mask.data(), tag_mask.size()};
    v.delta      = raft::device_span<i_t>{delta.data(), delta.size()};
    v.fwd_excess = raft::device_span<double>{fwd_excess.data(), fwd_excess.size()};
    v.bwd_excess = raft::device_span<double>{bwd_excess.data(), bwd_excess.size()};
    v.fwd_count  = raft::device_span<i_t>{fwd_count.data(), fwd_count.size()};
    v.bwd_pickup_tag_sum =
      raft::device_span<i_t>{bwd_pickup_tag_sum.data(), bwd_pickup_tag_sum.size()};
    v.dim_info = dim_info;
    v.stride   = static_cast<i_t>(tag_mask.size());
    return v;
  }

  // Shared memory required to host this dimension's per-route state.
  HDI static size_t get_shared_size(i_t route_size, incompat_dimension_info_t dim_info)
  {
    if (!dim_info.has_incompat) return 0;
    return route_size * sizeof(double)                   // fwd_excess
           + route_size * sizeof(double)                 // bwd_excess
           + route_size * sizeof(uint64_t)               // tag_mask
           + dim_info.n_tags * route_size * sizeof(i_t)  // fwd_count
           + dim_info.n_tags * route_size * sizeof(i_t)  // bwd_pickup_tag_sum
           + route_size * sizeof(i_t);                   // delta
  }

  //! Fixed per-node data (populated from problem)
  rmm::device_uvector<uint64_t> tag_mask;
  rmm::device_uvector<i_t> delta;
  //! Forward state
  rmm::device_uvector<i_t> fwd_count;  // [n_tags * stride]
  rmm::device_uvector<double> fwd_excess;
  //! Backward state (suffix-local)
  rmm::device_uvector<i_t> bwd_pickup_tag_sum;  // [n_tags * stride]
  rmm::device_uvector<double> bwd_excess;

  incompat_dimension_info_t dim_info;
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt

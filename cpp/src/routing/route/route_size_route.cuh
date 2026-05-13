/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <utilities/cuda_helpers.cuh>
#include "../node/route_size_node.cuh"
#include "../routing_helpers.cuh"
#include "../solution/solution_handle.cuh"
#include "routing/routing_helpers.cuh"

#include <raft/core/handle.hpp>
#include <raft/core/nvtx.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/tuple.h>

namespace cuopt {
namespace routing {
namespace detail {

template <typename i_t, typename f_t>
class route_size_route_t {
 public:
  route_size_route_t(solution_handle_t<i_t, f_t> const* sol_handle_,
                     const route_size_dimension_info_t& dim_info_)
    : count_increment(0, sol_handle_->get_stream()),
      fwd_count(0, sol_handle_->get_stream()),
      bwd_count(0, sol_handle_->get_stream()),
      dim_info(dim_info_)
  {
    raft::common::nvtx::range fun_scope("zero route_size route copy_ctr");
  }

  route_size_route_t(const route_size_route_t& route_size_route,
                     solution_handle_t<i_t, f_t> const* sol_handle_)
    : count_increment(route_size_route.count_increment, sol_handle_->get_stream()),
      fwd_count(route_size_route.fwd_count, sol_handle_->get_stream()),
      bwd_count(route_size_route.bwd_count, sol_handle_->get_stream()),
      dim_info(route_size_route.dim_info)
  {
    raft::common::nvtx::range fun_scope("route_size route copy_ctr");
  }

  route_size_route_t& operator=(route_size_route_t&& route_size_route) = default;

  void resize(i_t max_nodes_per_route, rmm::cuda_stream_view stream)
  {
    count_increment.resize(max_nodes_per_route, stream);
    fwd_count.resize(max_nodes_per_route, stream);
    bwd_count.resize(max_nodes_per_route, stream);
  }

  struct view_t {
    bool is_empty() const { return count_increment.empty(); }

    DI route_size_node_t<i_t, f_t> get_node(i_t idx) const
    {
      route_size_node_t<i_t, f_t> node;
      node.count_increment = count_increment[idx];
      node.fwd_count       = fwd_count[idx];
      node.bwd_count       = bwd_count[idx];
      return node;
    }

    DI void set_node(i_t idx, const route_size_node_t<i_t, f_t>& node)
    {
      count_increment[idx] = node.count_increment;
      set_forward_data(idx, node);
      set_backward_data(idx, node);
    }

    DI void set_forward_data(i_t idx, const route_size_node_t<i_t, f_t>& node)
    {
      fwd_count[idx] = node.fwd_count;
    }

    DI void set_backward_data(i_t idx, const route_size_node_t<i_t, f_t>& node)
    {
      bwd_count[idx] = node.bwd_count;
    }

    DI void copy_forward_data(const view_t& orig_route, i_t start_idx, i_t end_idx, i_t write_start)
    {
      i_t size = end_idx - start_idx;
      block_copy(fwd_count.subspan(write_start), orig_route.fwd_count.subspan(start_idx), size);
    }

    DI void copy_backward_data(const view_t& orig_route,
                               i_t start_idx,
                               i_t end_idx,
                               i_t write_start)
    {
      i_t size = end_idx - start_idx;
      block_copy(bwd_count.subspan(write_start), orig_route.bwd_count.subspan(start_idx), size);
    }

    DI void copy_fixed_route_data(const view_t& orig_route,
                                  i_t from_idx,
                                  i_t to_idx,
                                  i_t write_start)
    {
      i_t size = to_idx - from_idx;
      block_copy(
        count_increment.subspan(write_start), orig_route.count_increment.subspan(from_idx), size);
    }

    DI void compute_cost(const VehicleInfo<f_t>& vehicle_info,
                         const i_t n_nodes,
                         [[maybe_unused]] objective_cost_t& obj_cost,
                         infeasible_cost_t& inf_cost) const noexcept
    {
      // At the return depot (index n_nodes), fwd_count equals total service-node visits.
      inf_cost[dim_t::ROUTE_SIZE] = max(0, fwd_count[n_nodes] - vehicle_info.max_route_size);
    }

    static DI thrust::tuple<view_t, i_t*> create_shared_route(
      i_t* shmem, const route_size_dimension_info_t dim_info, i_t n_nodes_route)
    {
      view_t v;
      v.dim_info = dim_info;

      const size_t sz = static_cast<size_t>(n_nodes_route + 1);
      i_t* sh_ptr     = shmem;

      thrust::tie(v.count_increment, sh_ptr) = wrap_ptr_as_span<i_t>(sh_ptr, sz);
      thrust::tie(v.fwd_count, sh_ptr)       = wrap_ptr_as_span<i_t>(sh_ptr, sz);
      thrust::tie(v.bwd_count, sh_ptr)       = wrap_ptr_as_span<i_t>(sh_ptr, sz);

      return thrust::make_tuple(v, sh_ptr);
    }

    raft::device_span<i_t> count_increment;
    raft::device_span<i_t> fwd_count;
    raft::device_span<i_t> bwd_count;
    route_size_dimension_info_t dim_info;
  };

  view_t view()
  {
    view_t v;
    v.count_increment = raft::device_span<i_t>{count_increment.data(), count_increment.size()};
    v.fwd_count       = raft::device_span<i_t>{fwd_count.data(), fwd_count.size()};
    v.bwd_count       = raft::device_span<i_t>{bwd_count.data(), bwd_count.size()};
    v.dim_info        = dim_info;
    return v;
  }

  HDI static size_t get_shared_size(i_t route_size,
                                    [[maybe_unused]] route_size_dimension_info_t dim_info)
  {
    // count_increment, fwd_count, bwd_count
    return 3 * route_size * sizeof(i_t);
  }

  // Per-node 1-for-service, 0-for-depot increment (immutable after creation).
  rmm::device_uvector<i_t> count_increment;
  // Propagated cumulative counts.
  rmm::device_uvector<i_t> fwd_count;
  rmm::device_uvector<i_t> bwd_count;

  route_size_dimension_info_t dim_info;
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt

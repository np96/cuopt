/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <gtest/gtest.h>
#include <cuopt/routing/solve.hpp>
#include <utilities/copy_helpers.hpp>

#include <numeric>
#include <unordered_map>
#include <vector>

namespace cuopt {
namespace routing {
namespace test {

namespace {

// Build a uniform symmetric cost matrix over `nlocations` locations.
std::vector<float> uniform_cost_matrix(int nlocations, float off_diag)
{
  std::vector<float> m(nlocations * nlocations, off_diag);
  for (int i = 0; i < nlocations; ++i) {
    m[i * nlocations + i] = 0.f;
  }
  return m;
}

// Per-vehicle service-node visit count: stops in `host_route` whose node_type is not DEPOT.
std::vector<int> per_vehicle_visit_count(host_assignment_t<int> const& host_route, int nvehicles)
{
  std::vector<int> counts(nvehicles, 0);
  for (size_t i = 0; i < host_route.truck_id.size(); ++i) {
    const auto truck = host_route.truck_id[i];
    const auto type  = host_route.node_types[i];
    if (static_cast<node_type_t>(type) != node_type_t::DEPOT) { counts[truck] += 1; }
  }
  return counts;
}

}  // namespace

// 6 simple deliveries, 3 vehicles, max_route_size = 2 per vehicle. Total demand
// = 6 visits, total capacity = 6, so the solver must use all three vehicles
// with exactly two stops each. Without ROUTE_SIZE the solver would normally
// pack all deliveries onto one cheap vehicle when capacity allows.
TEST(route_size_limit, vrp_hard_limit_two_per_vehicle)
{
  constexpr int nlocations = 7;  // 1 depot + 6 customers
  constexpr int nvehicles  = 3;
  constexpr int norders    = 6;
  constexpr int max_size   = 2;

  auto cost_matrix = uniform_cost_matrix(nlocations, /*off_diag=*/10.f);

  // Orders are at locations 1..6; depot at 0.
  std::vector<int> order_locations(norders);
  std::iota(order_locations.begin(), order_locations.end(), 1);

  std::vector<int> vehicle_start_locations(nvehicles, 0);
  std::vector<int> vehicle_return_locations(nvehicles, 0);
  std::vector<int> vehicle_max_route_sizes(nvehicles, max_size);

  raft::handle_t handle;
  auto stream = handle.get_stream();

  auto v_cost            = cuopt::device_copy(cost_matrix, stream);
  auto v_order_locations = cuopt::device_copy(order_locations, stream);
  auto v_start           = cuopt::device_copy(vehicle_start_locations, stream);
  auto v_return          = cuopt::device_copy(vehicle_return_locations, stream);
  auto v_max_sizes       = cuopt::device_copy(vehicle_max_route_sizes, stream);

  data_model_view_t<int, float> data_model(&handle, nlocations, nvehicles, norders);
  data_model.add_cost_matrix(v_cost.data());
  data_model.set_order_locations(v_order_locations.data());
  data_model.set_vehicle_locations(v_start.data(), v_return.data());
  data_model.set_vehicle_max_route_sizes(v_max_sizes.data());

  auto routing_solution = solve(data_model);
  handle.sync_stream();

  ASSERT_EQ(routing_solution.get_status(), solution_status_t::SUCCESS);

  host_assignment_t<int> host_route(routing_solution);
  const auto visits = per_vehicle_visit_count(host_route, nvehicles);

  // All orders must be served.
  ASSERT_EQ(std::accumulate(visits.begin(), visits.end(), 0), norders);
  // No vehicle may exceed its per-route size limit.
  for (int v = 0; v < nvehicles; ++v) {
    EXPECT_LE(visits[v], max_size)
      << "vehicle " << v << " visited " << visits[v] << " orders, max_route_size=" << max_size;
  }
}

// Mixed vehicles: vehicle 0 is tight (max_route_size = 1), vehicles 1 & 2 are
// unconstrained (large limit). Constrained vehicles respect the limit while
// unconstrained ones may take more.
TEST(route_size_limit, vrp_heterogeneous_limits)
{
  constexpr int nlocations = 7;  // 1 depot + 6 customers
  constexpr int nvehicles  = 3;
  constexpr int norders    = 6;

  auto cost_matrix = uniform_cost_matrix(nlocations, /*off_diag=*/10.f);

  std::vector<int> order_locations(norders);
  std::iota(order_locations.begin(), order_locations.end(), 1);

  std::vector<int> vehicle_start_locations(nvehicles, 0);
  std::vector<int> vehicle_return_locations(nvehicles, 0);

  // Tight on vehicle 0, effectively unconstrained on the rest.
  std::vector<int> vehicle_max_route_sizes = {1, norders, norders};

  raft::handle_t handle;
  auto stream = handle.get_stream();

  auto v_cost            = cuopt::device_copy(cost_matrix, stream);
  auto v_order_locations = cuopt::device_copy(order_locations, stream);
  auto v_start           = cuopt::device_copy(vehicle_start_locations, stream);
  auto v_return          = cuopt::device_copy(vehicle_return_locations, stream);
  auto v_max_sizes       = cuopt::device_copy(vehicle_max_route_sizes, stream);

  data_model_view_t<int, float> data_model(&handle, nlocations, nvehicles, norders);
  data_model.add_cost_matrix(v_cost.data());
  data_model.set_order_locations(v_order_locations.data());
  data_model.set_vehicle_locations(v_start.data(), v_return.data());
  data_model.set_vehicle_max_route_sizes(v_max_sizes.data());

  auto routing_solution = solve(data_model);
  handle.sync_stream();

  ASSERT_EQ(routing_solution.get_status(), solution_status_t::SUCCESS);

  host_assignment_t<int> host_route(routing_solution);
  const auto visits = per_vehicle_visit_count(host_route, nvehicles);

  ASSERT_EQ(std::accumulate(visits.begin(), visits.end(), 0), norders);
  EXPECT_LE(visits[0], vehicle_max_route_sizes[0]);
  EXPECT_LE(visits[1], vehicle_max_route_sizes[1]);
  EXPECT_LE(visits[2], vehicle_max_route_sizes[2]);
}

// No-regression: not calling set_vehicle_max_route_sizes leaves the dimension
// disabled, so the solver behaves exactly as before for a simple VRP.
TEST(route_size_limit, vrp_disabled_when_unset)
{
  constexpr int nlocations = 5;
  constexpr int nvehicles  = 2;
  constexpr int norders    = 4;

  auto cost_matrix = uniform_cost_matrix(nlocations, /*off_diag=*/10.f);

  std::vector<int> order_locations(norders);
  std::iota(order_locations.begin(), order_locations.end(), 1);

  std::vector<int> vehicle_start_locations(nvehicles, 0);
  std::vector<int> vehicle_return_locations(nvehicles, 0);

  raft::handle_t handle;
  auto stream = handle.get_stream();

  auto v_cost            = cuopt::device_copy(cost_matrix, stream);
  auto v_order_locations = cuopt::device_copy(order_locations, stream);
  auto v_start           = cuopt::device_copy(vehicle_start_locations, stream);
  auto v_return          = cuopt::device_copy(vehicle_return_locations, stream);

  data_model_view_t<int, float> data_model(&handle, nlocations, nvehicles, norders);
  data_model.add_cost_matrix(v_cost.data());
  data_model.set_order_locations(v_order_locations.data());
  data_model.set_vehicle_locations(v_start.data(), v_return.data());

  auto routing_solution = solve(data_model);
  handle.sync_stream();

  ASSERT_EQ(routing_solution.get_status(), solution_status_t::SUCCESS);

  host_assignment_t<int> host_route(routing_solution);
  const auto visits = per_vehicle_visit_count(host_route, nvehicles);
  EXPECT_EQ(std::accumulate(visits.begin(), visits.end(), 0), norders);
}

}  // namespace test
}  // namespace routing
}  // namespace cuopt

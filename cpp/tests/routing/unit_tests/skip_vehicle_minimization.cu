/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <gtest/gtest.h>
#include <cuopt/routing/solve.hpp>
#include <routing/utilities/check_constraints.hpp>
#include <utilities/copy_helpers.hpp>

#include <vector>

namespace cuopt {
namespace routing {
namespace test {

TEST(skip_vehicle_minimization, setter_getter)
{
  solver_settings_t<int, float> settings;
  ASSERT_FALSE(settings.get_skip_vehicle_minimization());

  settings.set_skip_vehicle_minimization(true);
  ASSERT_TRUE(settings.get_skip_vehicle_minimization());

  settings.set_skip_vehicle_minimization(false);
  ASSERT_FALSE(settings.get_skip_vehicle_minimization());
}

// Behavioural test: the cost matrix is non-metric — depot↔order edges are
// cheap (1) while order↔order edges are expensive (100). The 1-route
// solution that visits all four orders costs 1+100+100+100+1 = 302; the
// 4-route solution (one order per vehicle) costs 4*(1+1) = 8.
//
// Default solver behaviour first minimises the route count, which yields a
// 1-route initial solution that local search cannot escape (every merge it
// considers undoing would *increase* travel cost relative to the current
// 1-route state, but reducing to 1 route was found via feasibility-only
// ejection that ignored cost).
//
// With skip_vehicle_minimization enabled, the solver seeds with
// min(fleet_size, num_orders) routes and lets cost-minimisation choose the
// final route count — landing near the 4-route optimum.
TEST(skip_vehicle_minimization, dispersed_cost_optimum)
{
  constexpr int nlocations = 5;  // depot + 4 order locations
  constexpr int norders    = 4;
  constexpr int nvehicles  = 8;

  std::vector<float> cost_matrix(nlocations * nlocations, 100.f);
  for (int i = 0; i < nlocations; ++i) {
    cost_matrix[i * nlocations + i] = 0.f;
  }
  for (int i = 1; i < nlocations; ++i) {
    cost_matrix[0 * nlocations + i] = 1.f;
    cost_matrix[i * nlocations + 0] = 1.f;
  }

  std::vector<int> order_locations = {1, 2, 3, 4};

  raft::handle_t handle;
  auto stream = handle.get_stream();

  auto v_cost_matrix     = cuopt::device_copy(cost_matrix, stream);
  auto v_order_locations = cuopt::device_copy(order_locations, stream);

  data_model_view_t<int, float> data_model(&handle, nlocations, nvehicles, norders);
  data_model.add_cost_matrix(v_cost_matrix.data());
  data_model.set_order_locations(v_order_locations.data());

  // Default: vehicle-count minimisation collapses to a single route.
  float default_cost        = 0.f;
  int default_vehicle_count = 0;
  {
    solver_settings_t<int, float> settings;
    settings.set_time_limit(5);
    ASSERT_FALSE(settings.get_skip_vehicle_minimization());

    auto routing_solution = cuopt::routing::solve(data_model, settings);
    handle.sync_stream();
    ASSERT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);

    auto host_route = cuopt::routing::host_assignment_t(routing_solution);
    check_route(data_model, host_route);
    default_cost          = routing_solution.get_total_objective();
    default_vehicle_count = routing_solution.get_vehicle_count();
  }

  // Skip vehicle minimisation: solver seeds with 4 routes and stays there
  // because cost rises sharply if any pair of orders shares a vehicle.
  float skip_cost        = 0.f;
  int skip_vehicle_count = 0;
  {
    solver_settings_t<int, float> settings;
    settings.set_time_limit(5);
    settings.set_skip_vehicle_minimization(true);

    auto routing_solution = cuopt::routing::solve(data_model, settings);
    handle.sync_stream();
    ASSERT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);

    auto host_route = cuopt::routing::host_assignment_t(routing_solution);
    check_route(data_model, host_route);
    skip_cost          = routing_solution.get_total_objective();
    skip_vehicle_count = routing_solution.get_vehicle_count();
  }

  // 4-route optimum is 8; 3-route is 1+100+1 + 1+100+1 + 1+1 = 206; 2-route
  // is at least 1+100+1 + 1+100+100+1 = 204; 1-route is 302. The 4-route
  // optimum should be clearly distinguishable.
  ASSERT_LT(skip_cost, 50.f) << "skip_vehicle_minimization solution cost " << skip_cost
                             << " is not near the 4-route optimum (8)";
  ASSERT_EQ(skip_vehicle_count, norders);

  ASSERT_GT(default_cost, 100.f) << "Default solver cost " << default_cost
                                 << " unexpectedly avoids the 1-route trap; "
                                 << "the flag may have become a no-op.";
  ASSERT_LT(default_vehicle_count, norders);

  // The flag should yield a strictly better objective on this problem.
  ASSERT_LT(skip_cost, default_cost);
}

// PDP variant: the inverse of dispersed_cost_optimum. Here the cost matrix
// makes per-route depot trips expensive (50) and inter-stop edges cheap (1),
// so the cost-optimal solution merges all pickup-delivery pairs onto a
// single vehicle (cost ~107) rather than running them in parallel
// (4-vehicle cost ~404).
//
// This test exercises the route-reduction path in PDP local search
// (compute_insertions.cu:697). With skip_vehicle_minimization the solver
// seeds with N=4 routes. To reach the cost optimum it must MERGE pairs onto
// fewer vehicles — which requires LS to record cycle edges for routes that
// would be left empty after ejection. Without route-reduction enabled, that
// filter blocks the merges and the solver is stuck at 4 vehicles. With it
// enabled, LS can collapse routes.
TEST(skip_vehicle_minimization, pdp_route_reduction)
{
  constexpr int nlocations = 9;  // depot + 4 pickups + 4 deliveries
  constexpr int norders    = 8;  // 4 pickup-delivery pairs
  constexpr int nvehicles  = 4;

  // Depot-far / inter-stop-near cost matrix.
  std::vector<float> cost_matrix(nlocations * nlocations, 1.f);
  for (int i = 0; i < nlocations; ++i) {
    cost_matrix[i * nlocations + i] = 0.f;
  }
  for (int i = 1; i < nlocations; ++i) {
    cost_matrix[0 * nlocations + i] = 50.f;
    cost_matrix[i * nlocations + 0] = 50.f;
  }

  // orders 0..3 = pickups (locations 1..4); orders 4..7 = deliveries (5..8).
  std::vector<int> order_locations = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int> pickup_orders   = {0, 1, 2, 3};
  std::vector<int> delivery_orders = {4, 5, 6, 7};
  std::vector<int> demands         = {1, 1, 1, 1, -1, -1, -1, -1};
  std::vector<int> capacities(nvehicles, 100);
  std::vector<int> vehicle_start(nvehicles, 0);
  std::vector<int> vehicle_end(nvehicles, 0);

  raft::handle_t handle;
  auto stream = handle.get_stream();

  auto v_cost_matrix     = cuopt::device_copy(cost_matrix, stream);
  auto v_order_locations = cuopt::device_copy(order_locations, stream);
  auto v_pickup_orders   = cuopt::device_copy(pickup_orders, stream);
  auto v_delivery_orders = cuopt::device_copy(delivery_orders, stream);
  auto v_demands         = cuopt::device_copy(demands, stream);
  auto v_capacities      = cuopt::device_copy(capacities, stream);
  auto v_vehicle_start   = cuopt::device_copy(vehicle_start, stream);
  auto v_vehicle_end     = cuopt::device_copy(vehicle_end, stream);

  data_model_view_t<int, float> data_model(&handle, nlocations, nvehicles, norders);
  data_model.add_cost_matrix(v_cost_matrix.data());
  data_model.set_order_locations(v_order_locations.data());
  data_model.set_pickup_delivery_pairs(v_pickup_orders.data(), v_delivery_orders.data());
  data_model.set_vehicle_locations(v_vehicle_start.data(), v_vehicle_end.data());
  data_model.add_capacity_dimension("demand", v_demands.data(), v_capacities.data());

  solver_settings_t<int, float> settings;
  settings.set_time_limit(5);
  settings.set_skip_vehicle_minimization(true);

  auto routing_solution = cuopt::routing::solve(data_model, settings);
  handle.sync_stream();
  ASSERT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);

  auto host_route = cuopt::routing::host_assignment_t(routing_solution);
  check_route(data_model, host_route);

  // 4-vehicle stuck cost is ~404; any meaningful merge brings it under 300.
  // The 2-vehicle solution costs ~206, the 1-vehicle solution ~107.
  ASSERT_LT(routing_solution.get_total_objective(), 300.f)
    << "skip_vehicle_minimization failed to merge pickup-delivery pairs onto "
       "fewer vehicles (cost "
    << routing_solution.get_total_objective()
    << "). Likely the route-reduction filter in compute_insertions.cu is still "
       "blocking the necessary moves.";

  ASSERT_LT(routing_solution.get_vehicle_count(), nvehicles)
    << "skip_vehicle_minimization left every vehicle in use; route reduction "
       "did not happen.";
}

}  // namespace test
}  // namespace routing
}  // namespace cuopt

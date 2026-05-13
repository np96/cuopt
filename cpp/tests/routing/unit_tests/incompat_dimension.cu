/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

// Host-only unit tests for the INCOMPAT dimension's per-node math:
//   - forward propagation (pickup_cost summed in fwd_excess)
//   - backward propagation (suffix-local algebraic accumulators)
//   - combine invariance across all split points
//   - multi-tag intra/cross-order semantics
//   - non-PDP fallback (all delta = +1)
//
// These tests exercise incompat_node_t directly on the host (its methods are
// HDI). No device kernels, no full route/problem setup — math only.

#include <routing/node/incompat_node.cuh>
#include <routing/vehicle_info.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

namespace cuopt {
namespace routing {
namespace test {

namespace {

using i_t = int;
using f_t = float;
using detail::incompat_dimension_info_t;
using detail::incompat_node_t;
using detail::VehicleInfo;

constexpr double EPS = 1e-9;

// Holds a host-allocated matrix and a dim_info pointing at it.
struct fixture_t {
  std::vector<float> matrix;
  incompat_dimension_info_t info{};

  fixture_t(int n_tags, std::vector<float> m_row_major) : matrix(std::move(m_row_major))
  {
    EXPECT_EQ((int)matrix.size(), n_tags * n_tags) << "matrix size must be n_tags x n_tags";
    info.has_incompat    = true;
    info.n_tags          = n_tags;
    info.incompat_matrix = matrix.data();
  }
};

incompat_node_t<i_t, f_t> make_node(const incompat_dimension_info_t& info,
                                    int delta,
                                    uint64_t tag_mask)
{
  incompat_node_t<i_t, f_t> n(info);
  n.delta    = delta;
  n.tag_mask = tag_mask;
  return n;
}

void forward_sweep(std::vector<incompat_node_t<i_t, f_t>>& nodes)
{
  for (size_t k = 1; k < nodes.size(); ++k) {
    nodes[k - 1].calculate_forward(nodes[k]);
  }
}

void backward_sweep(std::vector<incompat_node_t<i_t, f_t>>& nodes)
{
  // The end-depot's bwd state is zero-init at construction; we just need to
  // propagate backward through the rest.
  for (int k = static_cast<int>(nodes.size()) - 2; k >= 0; --k) {
    nodes[k + 1].calculate_backward(nodes[k]);
  }
}

double combine_at(const incompat_node_t<i_t, f_t>& prev, const incompat_node_t<i_t, f_t>& next)
{
  VehicleInfo<f_t, false> v{};
  return incompat_node_t<i_t, f_t>::combine(prev, next, v, 0.0f);
}

// Build a fresh balanced PDP route:
//   [depot, pickup A, pickup B, delivery A, delivery B, depot]
// with tags as specified.
std::vector<incompat_node_t<i_t, f_t>> two_order_route(const incompat_dimension_info_t& info,
                                                       uint64_t mask_A,
                                                       uint64_t mask_B)
{
  return {
    make_node(info, 0, 0),        // 0: depot
    make_node(info, +1, mask_A),  // 1: pickup A
    make_node(info, +1, mask_B),  // 2: pickup B
    make_node(info, -1, mask_A),  // 3: delivery A
    make_node(info, -1, mask_B),  // 4: delivery B
    make_node(info, 0, 0),        // 5: depot
  };
}

}  // namespace

// ----- Forward propagation correctness -----

TEST(incompat_dim, forward_two_incompatible_singletons)
{
  // M = [[0, 1], [1, 0]] — tag 0 and tag 1 are incompatible (weight 1).
  fixture_t fx(2, {0.f, 1.f, 1.f, 0.f});
  auto nodes = two_order_route(fx.info, /*A=*/1ULL << 0, /*B=*/1ULL << 1);

  forward_sweep(nodes);

  // fwd_excess accumulates pickup_cost = count_before . (M . T).
  EXPECT_NEAR(nodes[0].fwd_excess, 0.0, EPS);  // depot
  EXPECT_NEAR(nodes[1].fwd_excess, 0.0, EPS);  // pickup A: no active orders
  EXPECT_NEAR(nodes[2].fwd_excess, 1.0, EPS);  // pickup B: count[0]=1 * M[0][1]
  EXPECT_NEAR(nodes[3].fwd_excess, 1.0, EPS);  // delivery A
  EXPECT_NEAR(nodes[4].fwd_excess, 1.0, EPS);  // delivery B
  EXPECT_NEAR(nodes[5].fwd_excess, 1.0, EPS);  // end depot

  // Active counts are balanced at end depot.
  EXPECT_EQ(nodes[5].fwd_count[0], 0);
  EXPECT_EQ(nodes[5].fwd_count[1], 0);
}

TEST(incompat_dim, forward_excess_is_non_negative)
{
  // Regression test for the self_pair_cost bug we previously had.
  fixture_t fx(2, {0.f, 1.f, 1.f, 0.f});

  // Lone multi-tag pickup must NOT produce negative excess.
  std::vector<incompat_node_t<i_t, f_t>> nodes = {
    make_node(fx.info, 0, 0),                           // depot
    make_node(fx.info, +1, (1ULL << 0) | (1ULL << 1)),  // pickup A {0,1}
    make_node(fx.info, -1, (1ULL << 0) | (1ULL << 1)),  // delivery A {0,1}
    make_node(fx.info, 0, 0),                           // depot
  };
  forward_sweep(nodes);
  for (size_t k = 0; k < nodes.size(); ++k) {
    EXPECT_GE(nodes[k].fwd_excess, 0.0) << "negative excess at pos " << k;
  }
  // No other order to pair against — total excess must be 0.
  EXPECT_NEAR(nodes.back().fwd_excess, 0.0, EPS);
}

TEST(incompat_dim, forward_multi_tag_cross_order)
{
  // A carries {0, 1}; B carries {2}. M[0][2] = M[1][2] = 1.
  // At B's pickup, count = [1, 1, 0]. M . bit(2) gives col 2 = [1, 1, 0].
  // pickup_cost = 1*1 + 1*1 + 0*0 = 2.
  fixture_t fx(3, {0.f, 0.f, 1.f, 0.f, 0.f, 1.f, 1.f, 1.f, 0.f});

  auto nodes = two_order_route(fx.info, /*A=*/(1ULL << 0) | (1ULL << 1), /*B=*/1ULL << 2);

  forward_sweep(nodes);
  EXPECT_NEAR(nodes[2].fwd_excess, 2.0, EPS);
  EXPECT_NEAR(nodes.back().fwd_excess, 2.0, EPS);
}

TEST(incompat_dim, forward_weighted_matrix)
{
  // Non-binary weights.
  fixture_t fx(2, {0.f, 2.5f, 2.5f, 0.f});
  auto nodes = two_order_route(fx.info, /*A=*/1ULL << 0, /*B=*/1ULL << 1);
  forward_sweep(nodes);
  EXPECT_NEAR(nodes.back().fwd_excess, 2.5, EPS);
}

TEST(incompat_dim, forward_duration_independence)
{
  // Same incompatible pair, longer overlap (more dummy nodes between
  // A's pickup and the deliveries). Event-based cost should be unchanged.
  fixture_t fx(2, {0.f, 1.f, 1.f, 0.f});

  // [depot, pickup A {0}, pickup B {1}, ... dummies ..., deliv A, deliv B, depot]
  // dummies: depot-like nodes (delta=0, tag_mask=0).
  std::vector<incompat_node_t<i_t, f_t>> nodes;
  nodes.push_back(make_node(fx.info, 0, 0));
  nodes.push_back(make_node(fx.info, +1, 1ULL << 0));
  nodes.push_back(make_node(fx.info, +1, 1ULL << 1));
  for (int i = 0; i < 5; ++i)
    nodes.push_back(make_node(fx.info, 0, 0));
  nodes.push_back(make_node(fx.info, -1, 1ULL << 0));
  nodes.push_back(make_node(fx.info, -1, 1ULL << 1));
  nodes.push_back(make_node(fx.info, 0, 0));

  forward_sweep(nodes);
  EXPECT_NEAR(nodes.back().fwd_excess, 1.0, EPS);
}

// ----- Backward propagation correctness -----

TEST(incompat_dim, backward_sweep_matches_forward_via_compute_cost)
{
  fixture_t fx(2, {0.f, 1.f, 1.f, 0.f});
  auto nodes = two_order_route(fx.info, /*A=*/1ULL << 0, /*B=*/1ULL << 1);
  forward_sweep(nodes);
  backward_sweep(nodes);

  // For a complete balanced route, fwd_excess[N] should equal
  // combine(node[N-1], node[N]) — invariant required by the framework.
  size_t N        = nodes.size() - 1;
  double from_fwd = nodes[N].fwd_excess;
  double from_cmb = combine_at(nodes[N - 1], nodes[N]);
  EXPECT_NEAR(from_fwd, from_cmb, EPS);
}

TEST(incompat_dim, backward_pickup_tag_sum_at_start_depot_matches_total_pickups)
{
  // After backward sweep, bwd_pickup_tag_sum at position 0 should equal the
  // total per-tag pickup count over the whole route (i.e., sum of bits at
  // every pickup node).
  fixture_t fx(3, {0.f, 0.f, 1.f, 0.f, 0.f, 1.f, 1.f, 1.f, 0.f});

  auto nodes = two_order_route(fx.info, /*A=*/(1ULL << 0) | (1ULL << 1), /*B=*/1ULL << 2);
  backward_sweep(nodes);

  // Pickups in this route: A at pos 1 (tags 0,1), B at pos 2 (tag 2).
  // Total per-tag pickup count: tag 0 = 1, tag 1 = 1, tag 2 = 1.
  EXPECT_EQ(nodes[0].bwd_pickup_tag_sum[0], 1);
  EXPECT_EQ(nodes[0].bwd_pickup_tag_sum[1], 1);
  EXPECT_EQ(nodes[0].bwd_pickup_tag_sum[2], 1);
}

// ----- Combine invariant (the most important math test) -----

TEST(incompat_dim, combine_invariant_across_all_split_points)
{
  fixture_t fx(2, {0.f, 1.f, 1.f, 0.f});
  auto nodes = two_order_route(fx.info, /*A=*/1ULL << 0, /*B=*/1ULL << 1);

  forward_sweep(nodes);
  backward_sweep(nodes);

  // combine(node[k], node[k+1]) must return the same total route excess for
  // every split point k.
  double expected = nodes.back().fwd_excess;  // total route excess
  for (size_t k = 0; k + 1 < nodes.size(); ++k) {
    double c = combine_at(nodes[k], nodes[k + 1]);
    EXPECT_NEAR(c, expected, EPS) << "split at position " << k;
  }
}

TEST(incompat_dim, combine_invariant_multi_tag_cross_order)
{
  fixture_t fx(3, {0.f, 0.f, 1.f, 0.f, 0.f, 1.f, 1.f, 1.f, 0.f});

  auto nodes = two_order_route(fx.info, /*A=*/(1ULL << 0) | (1ULL << 1), /*B=*/1ULL << 2);
  forward_sweep(nodes);
  backward_sweep(nodes);

  double expected = nodes.back().fwd_excess;
  for (size_t k = 0; k + 1 < nodes.size(); ++k) {
    double c = combine_at(nodes[k], nodes[k + 1]);
    EXPECT_NEAR(c, expected, EPS) << "split at position " << k;
  }
}

TEST(incompat_dim, combine_invariant_with_longer_overlap)
{
  fixture_t fx(2, {0.f, 1.f, 1.f, 0.f});

  std::vector<incompat_node_t<i_t, f_t>> nodes;
  nodes.push_back(make_node(fx.info, 0, 0));
  nodes.push_back(make_node(fx.info, +1, 1ULL << 0));
  nodes.push_back(make_node(fx.info, +1, 1ULL << 1));
  for (int i = 0; i < 4; ++i)
    nodes.push_back(make_node(fx.info, 0, 0));
  nodes.push_back(make_node(fx.info, -1, 1ULL << 0));
  nodes.push_back(make_node(fx.info, -1, 1ULL << 1));
  nodes.push_back(make_node(fx.info, 0, 0));

  forward_sweep(nodes);
  backward_sweep(nodes);

  double expected = nodes.back().fwd_excess;
  for (size_t k = 0; k + 1 < nodes.size(); ++k) {
    double c = combine_at(nodes[k], nodes[k + 1]);
    EXPECT_NEAR(c, expected, EPS) << "split at position " << k;
  }
}

// ----- Backward step ordering guard -----

TEST(incompat_dim, backward_step_no_self_pair)
{
  // Construct a suffix whose first prepended node is a pickup of an order
  // whose tags exactly match a later pickup's tags. If the implementation
  // erroneously uses the already-updated prev.bwd_pickup_tag_sum, the
  // prepended pickup would self-pair, producing extra cost.
  fixture_t fx(2, {0.f, 1.f, 1.f, 0.f});

  // Route: depot, pickup A {0}, pickup B {1}, deliv A, deliv B, depot
  // Combine at split 0 should match split 5 = total excess.
  auto nodes = two_order_route(fx.info, /*A=*/1ULL << 0, /*B=*/1ULL << 1);
  forward_sweep(nodes);
  backward_sweep(nodes);

  // Specifically: combine after prepending the pickup (split at position 1)
  // must equal total excess (1.0), not 2.0 (which would indicate self-pair).
  double c1 = combine_at(nodes[1], nodes[2]);
  EXPECT_NEAR(c1, 1.0, EPS);
}

// ----- Non-PDP fallback (all delta = +1) -----

TEST(incompat_dim, non_pdp_route_counts_pairs_anywhere_on_route)
{
  // Three orders, all on the same route, all "delta = +1" (no deliveries).
  // Expected cost = sum_{i<j} T_i . M . T_j  =  M[0][1] + M[0][2] + M[1][2].
  fixture_t fx(3, {0.f, 1.f, 2.f, 1.f, 0.f, 3.f, 2.f, 3.f, 0.f});

  std::vector<incompat_node_t<i_t, f_t>> nodes;
  nodes.push_back(make_node(fx.info, 0, 0));
  nodes.push_back(make_node(fx.info, +1, 1ULL << 0));
  nodes.push_back(make_node(fx.info, +1, 1ULL << 1));
  nodes.push_back(make_node(fx.info, +1, 1ULL << 2));
  nodes.push_back(make_node(fx.info, 0, 0));

  forward_sweep(nodes);
  // Active count never drops; cost at each pickup =
  //   pos 1 (first pickup): 0
  //   pos 2: count=[1,0,0], M.bit(1)=col1=[1,0,3] -> 1
  //   pos 3: count=[1,1,0], M.bit(2)=col2=[2,3,0] -> 1*2 + 1*3 = 5
  EXPECT_NEAR(nodes[1].fwd_excess, 0.0, EPS);
  EXPECT_NEAR(nodes[2].fwd_excess, 1.0, EPS);
  EXPECT_NEAR(nodes[3].fwd_excess, 6.0, EPS);

  // fwd_count does NOT return to zero — that's the non-PDP property.
  EXPECT_EQ(nodes.back().fwd_count[0], 1);
  EXPECT_EQ(nodes.back().fwd_count[1], 1);
  EXPECT_EQ(nodes.back().fwd_count[2], 1);

  // Combine invariant still holds.
  backward_sweep(nodes);
  double expected = nodes.back().fwd_excess;
  for (size_t k = 0; k + 1 < nodes.size(); ++k) {
    double c = combine_at(nodes[k], nodes[k + 1]);
    EXPECT_NEAR(c, expected, EPS) << "non-PDP split at " << k;
  }
}

// ----- Sanity: zero matrix yields zero excess -----

TEST(incompat_dim, zero_matrix_yields_zero_excess)
{
  fixture_t fx(3, std::vector<float>(9, 0.0f));
  auto nodes =
    two_order_route(fx.info, /*A=*/(1ULL << 0) | (1ULL << 2), /*B=*/(1ULL << 1) | (1ULL << 2));
  forward_sweep(nodes);
  EXPECT_NEAR(nodes.back().fwd_excess, 0.0, EPS);
}

}  // namespace test
}  // namespace routing
}  // namespace cuopt

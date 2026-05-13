# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for the order-tag incompatibility dimension (INCOMPAT).

These tests exercise the Python API surface and the end-to-end wiring:
  - setters accept device-array inputs and store them
  - Python-side validation rejects malformed inputs at setter time
  - C++-side validation rejects malformed inputs at solve time
  - Solve completes with the dim enabled (smoke)
  - The solver responds to incompatibility by splitting incompatible orders
    when extra vehicles are available

Math-level invariants (combine, propagation, edge cases) are covered by the
C++ unit tests in `cpp/tests/routing/unit_tests/incompat_dimension.cu`.
"""

import math

import numpy as np
import pytest

import cudf

from cuopt import routing
from cuopt.routing.vehicle_routing_wrapper import ErrorStatus


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _make_pdptw_dm(n_locations=5, n_vehicles=2, cost_value=2):
    """
    Build a minimal PDPTW data model:
        depot 0; pickup A at 1, pickup B at 2; delivery A at 3, delivery B at 4.
    Cost matrix is uniform so that route choice is dominated by INCOMPAT.
    """
    costs = cudf.DataFrame(
        {
            i: [0 if i == j else cost_value for j in range(n_locations)]
            for i in range(n_locations)
        },
        dtype=np.float32,
    )
    times = costs.astype(np.float32)

    pickup_indices = cudf.Series([1, 2], dtype=np.int32)
    delivery_indices = cudf.Series([3, 4], dtype=np.int32)
    demand = cudf.Series([0, 1, 1, -1, -1], dtype=np.int32)
    capacities = cudf.Series([2] * n_vehicles, dtype=np.int32)
    earliest = cudf.Series([0] * n_locations, dtype=np.int32)
    latest = cudf.Series([100] * n_locations, dtype=np.int32)

    dm = routing.DataModel(n_locations, n_vehicles)
    dm.add_cost_matrix(costs)
    dm.add_transit_time_matrix(times)
    dm.set_pickup_delivery_pairs(pickup_indices, delivery_indices)
    dm.add_capacity_dimension("demand", demand, capacities)
    dm.set_order_time_windows(earliest, latest)
    return dm


def _solve(dm, time_limit_s=2):
    settings = routing.SolverSettings()
    settings.set_time_limit(time_limit_s)
    return routing.Solve(dm, settings)


def _truck_for_location(route_df, loc):
    """Return the truck_id that visited location `loc` (or None)."""
    rows = route_df[route_df["route"] == loc]
    if len(rows) == 0:
        return None
    return rows["truck_id"].to_arrow().to_pylist()[0]


# -----------------------------------------------------------------------------
# Python-side validation (set_incompatibility_matrix)
# -----------------------------------------------------------------------------


def test_set_incompatibility_matrix_rejects_non_square():
    dm = _make_pdptw_dm()
    bad = cudf.DataFrame(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    with pytest.raises(ValueError, match="square"):
        dm.set_incompatibility_matrix(bad)


def test_set_incompatibility_matrix_rejects_asymmetric():
    dm = _make_pdptw_dm()
    asym = cudf.DataFrame(
        [[0.0, 1.0], [2.0, 0.0]],
        dtype=np.float32,
    )
    with pytest.raises(ValueError, match="symmetric"):
        dm.set_incompatibility_matrix(asym)


def test_set_incompatibility_matrix_rejects_nonzero_diagonal():
    dm = _make_pdptw_dm()
    bad_diag = cudf.DataFrame(
        [[1.0, 1.0], [1.0, 0.0]],
        dtype=np.float32,
    )
    with pytest.raises(ValueError, match="diagonal"):
        dm.set_incompatibility_matrix(bad_diag)


def test_set_incompatibility_matrix_rejects_negative_values():
    dm = _make_pdptw_dm()
    neg = cudf.DataFrame(
        [[0.0, -1.0], [-1.0, 0.0]],
        dtype=np.float32,
    )
    with pytest.raises(ValueError, match="non-negative"):
        dm.set_incompatibility_matrix(neg)


def test_set_incompatibility_matrix_rejects_too_many_tags():
    dm = _make_pdptw_dm()
    # 33 x 33 — over the max_incompat_tags=32 cap
    big = cudf.DataFrame(np.zeros((33, 33), dtype=np.float32))
    with pytest.raises(ValueError, match="32"):
        dm.set_incompatibility_matrix(big)


# -----------------------------------------------------------------------------
# Python-side validation (set_order_tag_masks / set_order_tags)
# -----------------------------------------------------------------------------


def test_set_order_tag_masks_rejects_wrong_size():
    dm = _make_pdptw_dm()
    # PDPTW with depot+4 orders ⇒ 5 entries expected; pass 4.
    bad_size = cudf.Series([0, 1, 2, 3], dtype=np.uint64)
    with pytest.raises(ValueError):
        dm.set_order_tag_masks(bad_size)


def test_set_order_tags_rejects_oob_tag_id():
    dm = _make_pdptw_dm()
    # Tag id 32 is out of range (mask is uint64 but cap is 32 tag IDs).
    tag_lists = [[], [32], [], [], []]
    with pytest.raises(ValueError, match=r"\[0, 32\)"):
        dm.set_order_tags(tag_lists)


# -----------------------------------------------------------------------------
# C++-side validation (mixed presence / out-of-range bits)
# -----------------------------------------------------------------------------


def test_setting_only_masks_surfaces_validation_error():
    # cuOpt routes C++ validation errors through Assignment.error_status
    # rather than re-raising as Python exceptions.
    dm = _make_pdptw_dm()
    masks = cudf.Series([0, 0b01, 0b10, 0b01, 0b10], dtype=np.uint64)
    dm.set_order_tag_masks(masks)
    sol = _solve(dm)
    assert sol.get_error_status() == ErrorStatus.ValidationError, (
        f"expected ValidationError, got {sol.get_error_status()} "
        f"(msg: {sol.get_error_message()})"
    )


def test_setting_only_matrix_surfaces_validation_error():
    dm = _make_pdptw_dm()
    M = cudf.DataFrame([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    dm.set_incompatibility_matrix(M)
    sol = _solve(dm)
    assert sol.get_error_status() == ErrorStatus.ValidationError, (
        f"expected ValidationError, got {sol.get_error_status()} "
        f"(msg: {sol.get_error_message()})"
    )


def test_mask_bit_out_of_range_surfaces_validation_error():
    # n_tags = 2 ⇒ valid bits are 0 and 1; bit 2 is out of range.
    dm = _make_pdptw_dm()
    masks = cudf.Series(
        [0, 0b001, 0b010, 0b001, 0b100],  # last delivery's mask has bit 2 set
        dtype=np.uint64,
    )
    M = cudf.DataFrame([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    dm.set_order_tag_masks(masks)
    dm.set_incompatibility_matrix(M)
    sol = _solve(dm)
    assert sol.get_error_status() == ErrorStatus.ValidationError, (
        f"expected ValidationError, got {sol.get_error_status()} "
        f"(msg: {sol.get_error_message()})"
    )


def test_pickup_delivery_mask_mismatch_surfaces_validation_error():
    # Pickup mask must equal delivery mask for the same order in PDP.
    dm = _make_pdptw_dm()
    # depot, pickup A (bit 0), pickup B (bit 1), delivery A (bit 1!), delivery B (bit 1)
    bad = cudf.Series([0, 0b01, 0b10, 0b10, 0b10], dtype=np.uint64)
    M = cudf.DataFrame([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    dm.set_order_tag_masks(bad)
    dm.set_incompatibility_matrix(M)
    sol = _solve(dm)
    assert sol.get_error_status() == ErrorStatus.ValidationError, (
        f"expected ValidationError, got {sol.get_error_status()} "
        f"(msg: {sol.get_error_message()})"
    )


# -----------------------------------------------------------------------------
# Behavioral / solver-level
# -----------------------------------------------------------------------------


def test_solver_runs_with_dim_enabled_smoke():
    """
    Smoke test: dim enabled, valid inputs, single vehicle (no choice but
    co-load), solve completes successfully and returns a route.
    """
    dm = _make_pdptw_dm(n_vehicles=1)
    masks = cudf.Series([0, 0b01, 0b10, 0b01, 0b10], dtype=np.uint64)
    M = cudf.DataFrame([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    dm.set_order_tag_masks(masks)
    dm.set_incompatibility_matrix(M)
    sol = _solve(dm)
    assert sol.get_status() == 0, sol.get_message()
    route_df = sol.get_route()
    # All 4 orders + 2 depot visits should be in the route output (1 vehicle).
    assert len(route_df) >= 4


def test_solver_runs_with_dim_disabled_baseline():
    """Sanity: setting NEITHER tag_masks nor matrix is a no-op."""
    dm = _make_pdptw_dm(n_vehicles=2)
    sol = _solve(dm)
    assert sol.get_status() == 0, sol.get_message()


def test_solver_avoids_incompatible_overlap():
    """
    With strong incompatibility and 2 vehicles available, the solver must
    avoid pickup/delivery overlap of A and B. Acceptable outcomes:
      (a) A and B on different trucks (split), OR
      (b) Both on the same truck but sequenced non-overlapping
          (pA → dA → pB → dB  or  pB → dB → pA → dA).
    The bad outcome is co-loading: pA … pB … dA … dB or any pattern where
    A and B are simultaneously "in transit".

    Because the cost matrix is uniform, the solver may prefer fewer trucks
    (5 edges co-routed vs. 6 edges split) — that's fine as long as INCOMPAT
    cost stays 0 by careful sequencing.
    """
    dm = _make_pdptw_dm(n_vehicles=2)
    # Order A: pickup loc 1, delivery loc 3 — tag bit 0.
    # Order B: pickup loc 2, delivery loc 4 — tag bit 1.
    masks = cudf.Series([0, 0b01, 0b10, 0b01, 0b10], dtype=np.uint64)
    # Strong incompatibility weight.
    M = cudf.DataFrame([[0.0, 1000.0], [1000.0, 0.0]], dtype=np.float32)
    dm.set_order_tag_masks(masks)
    dm.set_incompatibility_matrix(M)

    sol = _solve(dm, time_limit_s=10)
    assert sol.get_error_status() == ErrorStatus.Success, (
        f"unexpected error: {sol.get_error_message()}"
    )
    assert sol.get_status() == 0, sol.get_message()

    route_df = sol.get_route()
    truck_A_pickup = _truck_for_location(route_df, 1)
    truck_A_delivery = _truck_for_location(route_df, 3)
    truck_B_pickup = _truck_for_location(route_df, 2)
    truck_B_delivery = _truck_for_location(route_df, 4)
    assert truck_A_pickup is not None
    assert truck_B_pickup is not None
    assert truck_A_pickup == truck_A_delivery, "Order A split across trucks"
    assert truck_B_pickup == truck_B_delivery, "Order B split across trucks"

    if truck_A_pickup != truck_B_pickup:
        # Outcome (a): split. We're done.
        return

    # Outcome (b): same truck. Verify non-overlapping sequence.
    # Find each location's row in this truck's route, sorted by arrival.
    truck = truck_A_pickup
    truck_rows = route_df[route_df["truck_id"] == truck].sort_values(
        "arrival_stamp"
    )
    seq = truck_rows["route"].to_arrow().to_pylist()
    # Locations: 1 = pA, 2 = pB, 3 = dA, 4 = dB.
    pA_idx, pB_idx = seq.index(1), seq.index(2)
    dA_idx, dB_idx = seq.index(3), seq.index(4)
    # Non-overlapping <=> dA comes before pB OR dB comes before pA.
    non_overlapping = (dA_idx < pB_idx) or (dB_idx < pA_idx)
    assert non_overlapping, (
        f"Both orders on truck {truck} but overlapping. "
        f"Sequence: {seq}; need pA→dA before pB or pB→dB before pA."
    )


def test_incompat_zero_matrix_no_op_vs_baseline():
    """
    Diagnostic: with M = all zeros, the dim is enabled but every entry
    contributes 0 cost. The solution should be identical (same objective)
    to solving with the dim disabled. This isolates "is the dim being
    enabled and read" from "is the matrix value actually consulted".
    """
    # Baseline — dim disabled.
    dm_base = _make_pdptw_dm(n_vehicles=2)
    sol_base = _solve(dm_base, time_limit_s=5)
    assert sol_base.get_error_status() == ErrorStatus.Success

    # Dim enabled with zero matrix.
    dm_dim = _make_pdptw_dm(n_vehicles=2)
    masks = cudf.Series([0, 0b01, 0b10, 0b01, 0b10], dtype=np.uint64)
    M = cudf.DataFrame([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    dm_dim.set_order_tag_masks(masks)
    dm_dim.set_incompatibility_matrix(M)
    sol_dim = _solve(dm_dim, time_limit_s=5)
    assert sol_dim.get_error_status() == ErrorStatus.Success

    # The two objectives should be identical (dim with M=0 is a no-op).
    assert math.isclose(
        sol_base.get_total_objective(),
        sol_dim.get_total_objective(),
        rel_tol=1e-5,
        abs_tol=1e-6,
    ), (
        f"M=0 should be equivalent to dim disabled, but got "
        f"baseline={sol_base.get_total_objective()} vs "
        f"M=0={sol_dim.get_total_objective()}"
    )


# -----------------------------------------------------------------------------
# Ergonomic helper
# -----------------------------------------------------------------------------


def test_set_order_tags_helper_round_trips_to_masks():
    """
    set_order_tags([[0,1], [0], [1], []]) should be equivalent to passing
    the OR-folded uint64 masks directly. We assert this indirectly by
    confirming the solver gives the same final cost in both setups.
    """
    M = cudf.DataFrame([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)

    # Helper path
    dm1 = _make_pdptw_dm(n_vehicles=1)
    dm1.set_order_tags(
        [[], [0], [1], [0], [1]]  # depot, pickups, deliveries (matched)
    )
    dm1.set_incompatibility_matrix(M)
    sol1 = _solve(dm1)
    assert sol1.get_status() == 0, sol1.get_message()

    # Direct mask path
    dm2 = _make_pdptw_dm(n_vehicles=1)
    dm2.set_order_tag_masks(
        cudf.Series([0, 0b01, 0b10, 0b01, 0b10], dtype=np.uint64)
    )
    dm2.set_incompatibility_matrix(M)
    sol2 = _solve(dm2)
    assert sol2.get_status() == 0, sol2.get_message()

    # Same setup => same objective.
    assert math.isclose(
        sol1.get_total_objective(),
        sol2.get_total_objective(),
        rel_tol=1e-5,
        abs_tol=1e-6,
    ), (
        f"set_order_tags should round-trip to set_order_tag_masks "
        f"(got {sol1.get_total_objective()} vs {sol2.get_total_objective()})"
    )

# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import cudf

from cuopt import routing
from cuopt.routing import utils

filename = utils.RAPIDS_DATASET_ROOT_DIR + "/solomon/In/r107.txt"


def test_time_windows():
    vehicle_num = 5
    d = utils.create_data_model(
        filename, num_vehicles=vehicle_num * 2, run_nodes=10
    )

    vehicle_earliest = []
    vehicle_latest = []
    latest_time = d.get_order_time_windows()[1].max()
    buffer_time = 50.0  # Time to travel back to or from the depot
    for i in range(vehicle_num):
        vehicle_earliest.append(0)
        vehicle_latest.append(latest_time / 2 + buffer_time)
    for i in range(vehicle_num):
        vehicle_earliest.append(latest_time / 2 - buffer_time)
        vehicle_latest.append(latest_time + buffer_time)
    d.set_vehicle_time_windows(
        cudf.Series(vehicle_earliest).astype(np.int32),
        cudf.Series(vehicle_latest).astype(np.int32),
    )

    s = routing.SolverSettings()
    s.set_time_limit(10)
    routing_solution = routing.Solve(d, s)

    ret_vehicle_time_windows = d.get_vehicle_time_windows()
    assert (ret_vehicle_time_windows[0] == cudf.Series(vehicle_earliest)).all()
    assert (ret_vehicle_time_windows[1] == cudf.Series(vehicle_latest)).all()

    assert routing_solution.get_status() == 0

    routes = routing_solution.get_route()
    truck_ids = routing_solution.get_route()["truck_id"].unique()

    for i in range(len(truck_ids)):
        truck_id = truck_ids.iloc[i]
        vehicle_route = routes[routes["truck_id"] == truck_id]
        assert (
            vehicle_route["arrival_stamp"].iloc[0]
            >= vehicle_earliest[truck_id]
        )
        assert (
            vehicle_route["arrival_stamp"].iloc[-1] <= vehicle_latest[truck_id]
        )


def test_vehicle_locations():
    d = utils.create_data_model(filename, run_nodes=10)
    num_vehicles = d.get_fleet_size()
    v_start_locations = cudf.Series([4] * num_vehicles)
    v_end_locations = cudf.Series([10] * num_vehicles)
    d.set_vehicle_locations(v_start_locations, v_end_locations)
    ret_start_locations, ret_end_locations = d.get_vehicle_locations()

    assert (v_start_locations == ret_start_locations).all()
    assert (v_end_locations == ret_end_locations).all()

    s = routing.SolverSettings()
    s.set_time_limit(10)
    routing_solution = routing.Solve(d, s)

    routes = routing_solution.get_route()
    truck_ids = routing_solution.get_route()["truck_id"].unique()

    for i in range(len(truck_ids)):
        truck_id = truck_ids.iloc[i]
        vehicle_route = routes[routes["truck_id"] == truck_id]
        assert vehicle_route["location"].iloc[0] == 4
        assert vehicle_route["location"].iloc[-1] == 10


# ----- Vehicle max route sizes -----


def test_vehicle_max_route_sizes():
    """
    Hard per-vehicle limit on number of service-node visits. With 6 orders
    and 3 vehicles each capped at 2, every vehicle must take exactly 2 stops.
    """
    n_locations = 7  # depot + 6 customers
    n_vehicles = 3
    max_size = 2

    # Uniform symmetric cost matrix; depot self-cost = 0.
    cost = cudf.DataFrame(
        [
            [0 if i == j else 10 for j in range(n_locations)]
            for i in range(n_locations)
        ],
        dtype=np.float32,
    )
    vehicle_max_route_sizes = cudf.Series(
        [max_size] * n_vehicles, dtype=np.int32
    )

    d = routing.DataModel(n_locations, n_vehicles)
    d.add_cost_matrix(cost)
    d.set_vehicle_max_route_sizes(vehicle_max_route_sizes)
    assert (d.get_vehicle_max_route_sizes() == vehicle_max_route_sizes).all()

    s = routing.SolverSettings()
    s.set_time_limit(5)

    routing_solution = routing.Solve(d, s)
    assert routing_solution.get_status() == 0

    route_df = routing_solution.get_route()
    for truck_id in route_df["truck_id"].unique().to_arrow().to_pylist():
        truck_rows = route_df[route_df["truck_id"] == truck_id]
        # Depot node id is 0 when order_locations is unset; non-depot rows are
        # the per-vehicle service-node count.
        service_count = int((truck_rows["route"] != 0).sum())
        assert service_count <= max_size, (
            f"truck {truck_id} served {service_count} orders (limit {max_size})"
        )


def test_vehicle_max_route_sizes_validation_fails_on_zero():
    """validate_positive should reject a zero or negative entry."""
    n_locations = 4
    n_vehicles = 2
    cost = cudf.DataFrame(
        [
            [0 if i == j else 1 for j in range(n_locations)]
            for i in range(n_locations)
        ],
        dtype=np.float32,
    )
    d = routing.DataModel(n_locations, n_vehicles)
    d.add_cost_matrix(cost)
    try:
        d.set_vehicle_max_route_sizes(cudf.Series([2, 0], dtype=np.int32))
    except ValueError:
        return
    raise AssertionError("expected ValueError for non-positive max_route_size")

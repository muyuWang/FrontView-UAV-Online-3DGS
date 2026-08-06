import torch
from scipy.spatial.transform import Rotation

from scripts.refine_360dvo_rotation_spline import (
    interpolate_knots,
    matrix_to_6d,
    rotation_6d_to_matrix,
)


def test_rotation_6d_round_trip_preserves_so3_matrices():
    matrices = torch.tensor(
        Rotation.random(20, random_state=3).as_matrix(), dtype=torch.float32
    )

    reconstructed = rotation_6d_to_matrix(matrix_to_6d(matrices))

    torch.testing.assert_close(reconstructed, matrices, rtol=1.0e-5, atol=1.0e-5)


def test_identical_knots_produce_a_constant_rotation_sequence():
    matrix = torch.tensor(
        Rotation.from_euler("xyz", [20.0, -10.0, 35.0], degrees=True).as_matrix(),
        dtype=torch.float32,
    )
    knots = matrix_to_6d(matrix).repeat(5, 1)

    frames = interpolate_knots(knots, frame_count=37, spacing=8)

    torch.testing.assert_close(
        frames,
        matrix.unsqueeze(0).expand_as(frames),
        rtol=1.0e-5,
        atol=1.0e-5,
    )

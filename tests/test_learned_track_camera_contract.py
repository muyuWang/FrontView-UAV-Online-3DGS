import numpy as np

from scripts import build_panoair_learned_tracks as tracks


def test_yaw_180_virtual_camera_extrinsic_preserves_center_and_flips_forward():
    source_c2w = np.eye(4, dtype=np.float64)
    source_c2w[:3, 3] = np.asarray([1.0, 2.0, 3.0])
    source_w2c = np.linalg.inv(source_c2w)

    virtual_w2c = tracks.apply_virtual_camera_extrinsic(
        source_w2c, yaw_deg=180.0, pitch_deg=0.0, roll_deg=0.0
    )
    virtual_c2w = np.linalg.inv(virtual_w2c)

    np.testing.assert_allclose(virtual_c2w[:3, 3], source_c2w[:3, 3], atol=1.0e-12)
    np.testing.assert_allclose(
        virtual_c2w[:3, :3] @ np.asarray([0.0, 0.0, 1.0]),
        np.asarray([0.0, 0.0, -1.0]),
        atol=1.0e-12,
    )


def test_camera_model_is_loaded_from_dataset_metadata(monkeypatch):
    camera = {
        "intrinsic": {"fx": 640.0, "fy": 641.0, "cx": 638.0, "cy": 359.0},
        "width": 1280,
        "height": 720,
    }

    monkeypatch.setattr(tracks, "K", tracks.K.copy())
    monkeypatch.setattr(tracks, "WIDTH", tracks.WIDTH)
    monkeypatch.setattr(tracks, "HEIGHT", tracks.HEIGHT)
    tracks.configure_camera_model([camera, dict(camera)])

    np.testing.assert_allclose(
        tracks.K,
        np.asarray([[640.0, 0.0, 638.0], [0.0, 641.0, 359.0], [0.0, 0.0, 1.0]]),
    )
    assert tracks.WIDTH == 1280
    assert tracks.HEIGHT == 720

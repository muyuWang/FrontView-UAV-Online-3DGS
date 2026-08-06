from collections import OrderedDict

from utils_new.eval_utils import pose_index_by_name


def test_subset_camera_name_resolves_to_reconstruction_pose_index():
    keyframes = OrderedDict(
        (f"aria_{index:05d}.png", index % 3 == 0) for index in range(12)
    )
    pose_indices = pose_index_by_name(keyframes)

    assert pose_indices["aria_00007.png"] == 7

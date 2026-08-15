import pytest

from utils_new.causal_depth_audit import validate_causal_depth_audit_config


def test_causal_depth_audit_is_default_disabled():
    config = validate_causal_depth_audit_config()
    assert config["enabled"] is False
    assert config["start_frame"] == -1
    assert config["audit_opacity_pruning"] is False


def test_causal_depth_audit_validates_boundary_and_isolation_contract():
    with pytest.raises(ValueError, match="start_frame"):
        validate_causal_depth_audit_config({"enabled": True})
    with pytest.raises(ValueError, match="requires"):
        validate_causal_depth_audit_config({"isolate_future_births": True})
    config = validate_causal_depth_audit_config(
        {
            "enabled": True,
            "start_frame": 620,
            "freeze_existing_geometry": True,
            "isolate_future_births": True,
        }
    )
    assert config["start_frame"] == 620
    assert config["freeze_existing_geometry"] is True


def test_causal_depth_audit_rejects_unknown_options():
    with pytest.raises(ValueError, match="Unknown"):
        validate_causal_depth_audit_config({"future_information": True})

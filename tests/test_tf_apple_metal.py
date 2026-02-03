# tests/test_tf_metal.py
import platform
import pytest

def _dist_installed(dist_name: str) -> bool:
    try:
        import importlib.metadata as md
        md.version(dist_name)
        return True
    except Exception:
        return False

@pytest.mark.skipif(platform.system() != "Darwin", reason="macOS only")
@pytest.mark.skipif(platform.machine() != "arm64", reason="Apple Silicon only")
def test_metal_plugin_present_implies_tf_sees_gpu():
    if not _dist_installed("tensorflow-metal"):
        pytest.skip("tensorflow-metal distribution not installed")

    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    assert gpus, "tensorflow-metal is installed but TF reports no GPU devices"

    details = tf.config.experimental.get_device_details(gpus[0])
    # This is what you observed: {'device_name': 'METAL'}
    assert details.get("device_name") == "METAL"

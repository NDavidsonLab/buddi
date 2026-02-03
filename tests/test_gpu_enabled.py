"""
Tests if TensorFlow is able to detect GPU devices when running on Apple Silicon with
the `tensorflow-metal` package installed (e.g., via the `gpu-metal` extra).

Also tests the gpu-nvidia extra on systems with NVIDIA GPUs, but this is not included here
because the CI does not run on such systems.
"""

import tensorflow as tf
import importlib.util
import pytest
import platform

has_tf_metal = (
    importlib.util.find_spec("tensorflow_metal") is not None
)

@pytest.mark.skipif(
    not has_tf_metal,
    reason="Requires Apple Silicon + tensorflow-metal (gpu-metal extra)"
)
def test_metal_enabled():
    gpus = tf.config.list_physical_devices('GPU')
    assert len(gpus) > 0, "No GPU devices found, but metal support test was marked."

has_nvidia_gpu = (
    importlib.util.find_spec("tensorflow") is not None
    and tf.config.list_physical_devices('GPU')
)

# def test_simple_model():
#     cifar = tf.keras.datasets.cifar100
#     (x_train, y_train), (x_test, y_test) = cifar.load_data()
#     model = tf.keras.applications.ResNet50(
#         include_top=True,
#         weights=None,
#         input_shape=(32, 32, 3),
#         classes=10,
#     )

#     loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
#     model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
#     model.fit(x_train, y_train, epochs=1, batch_size=64)

# def test_tensorflow_devices():
#     import tensorflow as tf

#     devices = tf.config.list_physical_devices()
#     print("Available devices:")
#     for d in devices:
#         print(f"  {d.name} ({d.device_type})")

#     assert any(
#         d.device_type == "GPU" and "METAL" in d.name.upper()
#         for d in devices
#     )

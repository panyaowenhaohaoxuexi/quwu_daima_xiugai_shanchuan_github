import importlib
import sys


def test_model_package_exports_only_formal_model_and_router():
    for name in list(sys.modules):
        if name == "model" or name.startswith("model."):
            del sys.modules[name]

    package = importlib.import_module("model")

    assert package.__all__ == ["FogRoutedRGBTIRDehazer", "MonotonicFogRouter"]
    assert "model.Teacher" not in sys.modules
    assert "model.gumbel_sigmoid" not in sys.modules
    assert "model.teacher_color" not in sys.modules

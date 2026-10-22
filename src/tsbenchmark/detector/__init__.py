import importlib

# Detectors are imported lazily, on first access, so that using one detector
# does not require the dependencies of all the others (merlion, prophet, tranad).
_LAZY_IMPORTS = {
    "gaussian_point_detector": "dummy_detectors",
    "random_detector": "dummy_detectors",
}

__all__ = list(_LAZY_IMPORTS)


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        module = importlib.import_module(f"{__name__}.{_LAZY_IMPORTS[name]}")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

import importlib

# Submodules are imported lazily, on first access (e.g. `tsbenchmark.metric`),
# so importing the package does not pull in heavy dependencies such as prophet.
__all__ = ["data_process", "metric", "utils", "choose_ts"]


def __getattr__(name):
    if name in __all__:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

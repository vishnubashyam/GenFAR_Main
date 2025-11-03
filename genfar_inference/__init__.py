from importlib.metadata import PackageNotFoundError, version


def _get_version() -> str:
    try:
        return version("genfar-inference")
    except PackageNotFoundError:
        return "0.0.0"


__all__ = ["__version__"]
__version__ = _get_version()

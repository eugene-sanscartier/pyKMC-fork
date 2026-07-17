import importlib
import pkgutil


def autodiscover(package_name: str, package_path) -> None:
    """Import every submodule so strategies register themselves."""
    for _, module_name, _ in pkgutil.iter_modules(package_path):
        try:
            importlib.import_module(f"{package_name}.{module_name}")
        except ImportError as e:
            raise ImportError(
                f"Strategy module '{module_name}' unavailable: {e}"
            ) from e

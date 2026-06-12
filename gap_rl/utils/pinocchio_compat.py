import sys
import warnings


import sys
import warnings


def _patch_pinocchio_model():
    try:
        import sapien.wrapper.pinocchio_model as pm
    except ImportError:
        return

    try:
        import pinocchio as pin_module
    except ImportError:
        pin_module = None

    if pin_module is not None and hasattr(pin_module, "buildModelFromXML"):
        return

    # The installed 'pinocchio' module lacks 'buildModelFromXML'.
    # This usually means the wrong pip package is installed
    # (the CLI tool 'pinocchio' v0.4.3, not the robotics library).
    if pin_module is not None and not hasattr(pin_module, "buildModelFromXML"):
        import importlib.metadata
        try:
            ver = importlib.metadata.version("pinocchio")
        except Exception:
            ver = "unknown"
        if ver == "0.4.3":
            warnings.warn(
                "The PyPI package 'pinocchio==0.4.3' (a CLI tool) shadows the real "
                "pinocchio robotics library. Uninstall it with:\n"
                "  pip uninstall pinocchio\n"
                "Then install the robotics library via conda (recommended on Windows):\n"
                "  conda install -c conda-forge pinocchio\n"
                "Or on Linux/Mac:\n"
                "  pip install pin"
            )
            return

    # Try alternative sources for the real pinocchio robotics library
    for candidate in ("pin",):
        try:
            real_pin = __import__(candidate)
            if hasattr(real_pin, "buildModelFromXML"):
                sys.modules["pinocchio"] = real_pin
                warnings.warn(
                    "Replaced wrong 'pinocchio' module with the correct one from 'pin'."
                )
                return
        except ImportError:
            continue

    if pin_module is None:
        warnings.warn(
            "Could not find a pinocchio library with 'buildModelFromXML'. "
            "Install the correct robotics pinocchio:\n"
            "  conda install -c conda-forge pinocchio\n\n"
            "Or on Linux/Mac:\n"
            "  pip install pin"
        )


_patch_pinocchio_model()

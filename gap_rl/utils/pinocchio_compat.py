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

    # The installed 'pinocchio' module is the wrong package (PyPI CLI tool v0.4.3),
    # not the robotics library. Try to import the real pinocchio from 'pin'.
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
            "Install the correct robotics pinocchio: `pip install pin`"
        )


_patch_pinocchio_model()

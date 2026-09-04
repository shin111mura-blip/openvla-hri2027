"""VLA package exports.

Keep RLDS dataset construction lazy so inference/evaluation code can import
token/layout utilities without requiring optional training-only dependencies
such as dlimp.
"""


def __getattr__(name):
    if name == "get_vla_dataset_and_collator":
        from .materialize import get_vla_dataset_and_collator

        return get_vla_dataset_and_collator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["get_vla_dataset_and_collator"]

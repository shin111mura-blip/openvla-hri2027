try:
    from .models import available_model_names, available_models, get_model_description, load
except ModuleNotFoundError as e:
    if e.name not in {"dlimp", "huggingface_hub"}:
        raise

    def _missing_optional_dependency(*args, **kwargs):
        raise ModuleNotFoundError(
            f"{e.name} is required for full Prismatic model loading/training APIs, but it is not installed in this "
            "lightweight test environment."
        ) from e

    available_model_names = _missing_optional_dependency
    available_models = _missing_optional_dependency
    get_model_description = _missing_optional_dependency
    load = _missing_optional_dependency

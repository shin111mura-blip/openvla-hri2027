try:
    from .models import available_model_names, available_models, get_model_description, load
except ModuleNotFoundError as e:
    if e.name != "dlimp":
        raise

    def _missing_dlimp(*args, **kwargs):
        raise ModuleNotFoundError(
            "dlimp is required for Prismatic training/RLDS dataset APIs, but it is not installed in this "
            "evaluation-only environment."
        ) from e

    available_model_names = _missing_dlimp
    available_models = _missing_dlimp
    get_model_description = _missing_dlimp
    load = _missing_dlimp

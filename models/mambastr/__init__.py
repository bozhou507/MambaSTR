from .mambastr import *


available_models = [
    'mambastr_tiny', 'mambastr_small', 'mambastr',
]


def get_model(model_name: str, args=None) -> MambaSTR:
    assert model_name in available_models, f'Unsupported model_name: {model_name}'
    import sys
    if args is not None and args.save_failure_cases:
        return getattr(sys.modules[__name__], model_name)(save_failure_cases=True)
    else:
        return getattr(sys.modules[__name__], model_name)()


__all__ = [*available_models, 'get_model']

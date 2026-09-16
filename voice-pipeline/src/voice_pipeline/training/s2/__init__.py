from importlib import import_module


_EXPORTS = {
    "TrainingCursor": (".checkpoint", "TrainingCursor"),
    "checkpoint_path": (".checkpoint", "checkpoint_path"),
    "load_checkpoint": (".checkpoint", "load_checkpoint"),
    "save_checkpoint": (".checkpoint", "save_checkpoint"),
    "S2TrainConfig": (".config", "S2TrainConfig"),
    "DeterministicEpochSampler": (".data", "DeterministicEpochSampler"),
    "S2Collate": (".data", "S2Collate"),
    "S2Dataset": (".data", "S2Dataset"),
    "build_optimizers": (".optim", "build_optimizers"),
    "build_schedulers": (".optim", "build_schedulers"),
    "S2StepResult": (".step", "S2StepResult"),
    "train_s2_step": (".step", "train_s2_step"),
    "S2Trainer": (".trainer", "S2Trainer"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value

from importlib import import_module


_EXPORTS = {
    "S1TrainConfig": (".config", "S1TrainConfig"),
    "S1Collate": (".data", "S1Collate"),
    "S1Dataset": (".data", "S1Dataset"),
    "S1Item": (".data", "S1Item"),
    "build_optimizer": (".optim", "build_optimizer"),
    "build_scheduler": (".optim", "build_scheduler"),
    "S1MiniBatchResult": (".step", "S1MiniBatchResult"),
    "S1OptimizerResult": (".step", "S1OptimizerResult"),
    "backward_s1_minibatch": (".step", "backward_s1_minibatch"),
    "finish_s1_optimizer_step": (".step", "finish_s1_optimizer_step"),
    "S1Trainer": (".trainer", "S1Trainer"),
    "S1TrainingCursor": (".checkpoint", "S1TrainingCursor"),
    "checkpoint_path": (".checkpoint", "checkpoint_path"),
    "load_checkpoint": (".checkpoint", "load_checkpoint"),
    "save_checkpoint": (".checkpoint", "save_checkpoint"),
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

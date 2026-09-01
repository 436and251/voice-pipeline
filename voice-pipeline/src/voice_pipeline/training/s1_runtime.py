from dataclasses import dataclass


@dataclass
class S1StepController:
    target_optimizer_steps: int
    optimizer_steps: int = 0

    def after_backward(self, batch_idx: int) -> bool:
        should_step = batch_idx > 0 and batch_idx % 4 == 0
        if should_step:
            self.optimizer_steps += 1
        return should_step

    @property
    def should_stop(self) -> bool:
        return self.optimizer_steps >= self.target_optimizer_steps

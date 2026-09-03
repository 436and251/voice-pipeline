from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader

from voice_pipeline.common.logging import PipelineLogger
from voice_pipeline.core.gpt_sovits.compatibility.s1_checkpoint import load_s1_checkpoint
from voice_pipeline.training.preprocess.cleanup import cleanup_after_training
from voice_pipeline.training.sampler import DeterministicEpochSampler

from .checkpoint import S1TrainingCursor, checkpoint_path, load_checkpoint, save_checkpoint
from .config import S1TrainConfig
from .data import S1Collate, S1Dataset
from .optim import build_optimizer, build_scheduler
from .step import backward_s1_minibatch, finish_s1_optimizer_step


class S1Trainer:
    def __init__(
        self,
        *,
        config: S1TrainConfig,
        model,
        optimizer,
        scheduler,
        scaler,
        loader,
        sampler,
        logger: PipelineLogger,
        cursor: S1TrainingCursor,
        cleanup=cleanup_after_training,
    ) -> None:
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.loader = loader
        self.sampler = sampler
        self.logger = logger
        self.cursor = cursor
        self.cleanup = cleanup

    @classmethod
    def from_pretrained(
        cls,
        config: S1TrainConfig,
        *,
        resume_from: Path | None = None,
        logger: PipelineLogger | None = None,
    ) -> "S1Trainer":
        config.validate()
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        device = torch.device(config.device)
        model = load_s1_checkpoint(config.base_s1_path, device)
        optimizer = build_optimizer(model, config)
        scheduler = build_scheduler(optimizer)
        scaler = torch.amp.GradScaler("cuda", enabled=config.precision == "fp16")
        dataset = S1Dataset(
            config.preprocess_dir,
            max_sec=config.max_sec,
            hz=config.hz,
            min_ps_ratio=config.min_ps_ratio,
            max_ps_ratio=config.max_ps_ratio,
        )
        sampler = DeterministicEpochSampler(dataset, config.seed)
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=config.num_workers,
            collate_fn=S1Collate(),
            pin_memory=device.type == "cuda",
            generator=torch.Generator().manual_seed(config.seed),
        )
        cursor = S1TrainingCursor()
        if resume_from is not None:
            cursor = load_checkpoint(
                resume_from,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                target_optimizer_steps=config.target_optimizer_steps,
            )
        optimizer.zero_grad()
        return cls(
            config=config,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            loader=loader,
            sampler=sampler,
            logger=logger or PipelineLogger(config.output_dir / "training" / "s1" / "events.jsonl"),
            cursor=cursor,
        )

    def train(self) -> S1TrainingCursor:
        self.model.train()
        self.optimizer.zero_grad()
        while self.cursor.optimizer_step < self.config.target_optimizer_steps:
            self.sampler.set_epoch(self.cursor.epoch)
            batch_count = len(self.loader)
            if batch_count == 0 or self.cursor.next_batch_index >= batch_count:
                raise ValueError("S1 resume batch cursor is outside the epoch")
            processed = False
            for batch_index, batch in enumerate(self.loader):
                if batch_index < self.cursor.next_batch_index:
                    continue
                result = backward_s1_minibatch(batch, self.model, self.scaler, self.config)
                processed = True
                next_batch = batch_index + 1
                epoch = self.cursor.epoch
                if next_batch == batch_count:
                    epoch += 1
                    next_batch = 0
                accumulation = self.cursor.accumulation_position + 1
                self.cursor = S1TrainingCursor(
                    self.cursor.optimizer_step, epoch, next_batch, accumulation
                )
                metrics = asdict(result)
                metrics["accumulation_position"] = accumulation
                self.logger.log(
                    "s1", "mini_batch", mini_step=batch_index + 1,
                    optimizer_step=self.cursor.optimizer_step, metrics=metrics,
                )
                if accumulation == self.config.gradient_accumulation:
                    update = finish_s1_optimizer_step(
                        self.model, self.optimizer, self.scheduler, self.scaler
                    )
                    self.cursor = S1TrainingCursor(
                        self.cursor.optimizer_step + 1, epoch, next_batch, 0
                    )
                    self.logger.log(
                        "s1", "optimizer", optimizer_step=self.cursor.optimizer_step,
                        metrics=asdict(update),
                    )
                    if (
                        self.cursor.optimizer_step % self.config.checkpoint_every_steps == 0
                        or self.cursor.optimizer_step == self.config.target_optimizer_steps
                    ):
                        destination = checkpoint_path(
                            self.config.output_dir, self.cursor.optimizer_step
                        )
                        save_checkpoint(
                            destination,
                            model=self.model,
                            optimizer=self.optimizer,
                            scheduler=self.scheduler,
                            scaler=self.scaler,
                            cursor=self.cursor,
                        )
                        self.logger.log(
                            "s1", "checkpoint", optimizer_step=self.cursor.optimizer_step,
                            metrics={"path": str(destination)},
                        )
                    if self.cursor.optimizer_step == self.config.target_optimizer_steps:
                        break
            if not processed:
                raise ValueError("S1 epoch contains no resumable batch")
        self.cleanup(self.config.preprocess_dir, True)
        return self.cursor


__all__ = ["S1Trainer"]

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader

from voice_pipeline.common.logging import PipelineLogger
from voice_pipeline.core.gpt_sovits.compatibility.s2_checkpoint import (
    load_s2_discriminator,
    load_s2_generator,
)
from voice_pipeline.training.preprocess.cleanup import cleanup_after_training

from .checkpoint import TrainingCursor, checkpoint_path, load_checkpoint, save_checkpoint
from .config import S2TrainConfig
from .data import DeterministicEpochSampler, S2Collate, S2Dataset
from .optim import build_optimizers, build_schedulers
from .step import train_s2_step


class S2Trainer:
    def __init__(
        self,
        *,
        config: S2TrainConfig,
        net_g,
        net_d,
        optim_g,
        optim_d,
        scheduler_g,
        scheduler_d,
        scaler,
        loader: DataLoader,
        sampler: DeterministicEpochSampler,
        logger: PipelineLogger,
        cursor: TrainingCursor,
    ) -> None:
        self.config = config
        self.net_g = net_g
        self.net_d = net_d
        self.optim_g = optim_g
        self.optim_d = optim_d
        self.scheduler_g = scheduler_g
        self.scheduler_d = scheduler_d
        self.scaler = scaler
        self.loader = loader
        self.sampler = sampler
        self.logger = logger
        self.cursor = cursor

    @classmethod
    def from_pretrained(
        cls,
        config: S2TrainConfig,
        *,
        resume_from: Path | None = None,
        logger: PipelineLogger | None = None,
    ) -> "S2Trainer":
        config.validate()
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        device = torch.device(config.device)
        net_g = load_s2_generator(config.base_s2g_path, device)
        net_d = load_s2_discriminator(config.base_s2d_path, device)
        optim_g, optim_d = build_optimizers(net_g, net_d, config)
        scheduler_g, scheduler_d = build_schedulers(optim_g, optim_d, config)
        scaler = torch.amp.GradScaler("cuda", enabled=config.precision == "fp16")
        dataset = S2Dataset(config.preprocess_dir)
        sampler = DeterministicEpochSampler(dataset, config.seed)
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=config.num_workers,
            collate_fn=S2Collate(),
            pin_memory=device.type == "cuda",
        )
        cursor = TrainingCursor()
        if resume_from is not None:
            cursor = load_checkpoint(
                resume_from,
                net_g=net_g,
                net_d=net_d,
                optim_g=optim_g,
                optim_d=optim_d,
                scheduler_g=scheduler_g,
                scheduler_d=scheduler_d,
                scaler=scaler,
                target_optimizer_steps=config.target_optimizer_steps,
            )
        return cls(
            config=config,
            net_g=net_g,
            net_d=net_d,
            optim_g=optim_g,
            optim_d=optim_d,
            scheduler_g=scheduler_g,
            scheduler_d=scheduler_d,
            scaler=scaler,
            loader=loader,
            sampler=sampler,
            logger=logger or PipelineLogger(config.output_dir / "training" / "s2" / "events.jsonl"),
            cursor=cursor,
        )

    def train(self) -> TrainingCursor:
        self.net_g.train()
        self.net_d.train()
        while self.cursor.global_step < self.config.target_optimizer_steps:
            self.sampler.set_epoch(self.cursor.epoch)
            batch_count = len(self.loader)
            if self.cursor.next_batch_index >= batch_count:
                raise ValueError("S2 resume batch cursor is outside the epoch")
            completed_batch = False
            for batch_index, batch in enumerate(self.loader):
                if batch_index < self.cursor.next_batch_index:
                    continue
                result = train_s2_step(
                    batch=batch,
                    net_g=self.net_g,
                    net_d=self.net_d,
                    optim_g=self.optim_g,
                    optim_d=self.optim_d,
                    scaler=self.scaler,
                    config=self.config,
                )
                global_step = self.cursor.global_step + 1
                if batch_index + 1 == batch_count:
                    self.scheduler_g.step()
                    self.scheduler_d.step()
                    self.cursor = TrainingCursor(global_step, self.cursor.epoch + 1, 0)
                else:
                    self.cursor = TrainingCursor(global_step, self.cursor.epoch, batch_index + 1)
                metrics = asdict(result)
                metrics["learning_rate"] = float(self.optim_g.param_groups[0]["lr"])
                self.logger.log(
                    "s2",
                    "batch",
                    mini_step=batch_index + 1,
                    optimizer_step=global_step,
                    metrics=metrics,
                )
                if (
                    global_step % self.config.checkpoint_every_steps == 0
                    or global_step == self.config.target_optimizer_steps
                ):
                    destination = checkpoint_path(self.config.output_dir, global_step)
                    save_checkpoint(
                        destination,
                        net_g=self.net_g,
                        net_d=self.net_d,
                        optim_g=self.optim_g,
                        optim_d=self.optim_d,
                        scheduler_g=self.scheduler_g,
                        scheduler_d=self.scheduler_d,
                        scaler=self.scaler,
                        cursor=self.cursor,
                    )
                    self.logger.log("s2", "checkpoint", optimizer_step=global_step, metrics={"path": str(destination)})
                completed_batch = True
                if global_step == self.config.target_optimizer_steps:
                    break
            if not completed_batch:
                raise ValueError("S2 epoch contains no resumable batch")
        cleanup_after_training(self.config.preprocess_dir, True)
        return self.cursor

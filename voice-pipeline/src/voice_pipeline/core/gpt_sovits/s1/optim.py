# Copyright 2022 Xiaomi Corp. (author: Daniel Povey)
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import contextlib
import logging
from collections import defaultdict

import torch
from torch import Tensor
from torch.optim import Optimizer


class BatchedOptimizer(Optimizer):
    @contextlib.contextmanager
    def batched_params(self, param_group, group_params_names):
        batches = defaultdict(list)
        batch_names = defaultdict(list)
        if len(param_group) != len(group_params_names):
            raise ValueError("parameter names do not match optimizer parameters")
        for parameter, name in zip(param_group, group_params_names):
            key = (str(parameter.dtype), *parameter.shape)
            batches[key].append(parameter)
            batch_names[key].append(name)
        rows = []
        grouped_parameters = []
        for key in sorted(batches):
            group = batches[key]
            stacked = torch.stack(group)
            stacked.grad = torch.stack(
                [torch.zeros_like(item) if item.grad is None else item.grad for item in group]
            )
            rows.append((stacked, self.state[group[0]], batch_names[key]))
            grouped_parameters.append(group)
        yield rows
        for (stacked, _state, _names), group in zip(rows, grouped_parameters):
            for index, parameter in enumerate(group):
                parameter.copy_(stacked[index])


class ScaledAdam(BatchedOptimizer):
    """The ScaledAdam implementation used by the pinned GPT-SoVITS S1 trainer."""

    def __init__(
        self,
        params,
        lr=3e-2,
        clipping_scale=None,
        betas=(0.9, 0.98),
        scalar_lr_scale=0.1,
        eps=1e-8,
        param_min_rms=1e-5,
        param_max_rms=3.0,
        scalar_max=10.0,
        size_update_period=4,
        clipping_update_period=100,
        parameters_names=None,
        show_dominant_parameters=True,
    ):
        if parameters_names is None:
            raise ValueError("parameters_names is required")
        defaults = {
            "lr": lr,
            "clipping_scale": clipping_scale,
            "betas": betas,
            "scalar_lr_scale": scalar_lr_scale,
            "eps": eps,
            "param_min_rms": param_min_rms,
            "param_max_rms": param_max_rms,
            "scalar_max": scalar_max,
            "size_update_period": size_update_period,
            "clipping_update_period": clipping_update_period,
        }
        super().__init__(params, defaults)
        if len(self.param_groups) != len(parameters_names):
            raise ValueError("parameter-name groups do not match optimizer groups")
        self.parameters_names = parameters_names
        self.show_dominant_parameters = show_dominant_parameters

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group, names in zip(self.param_groups, self.parameters_names):
            with self.batched_params(group["params"], names) as batches:
                clipping_scale = 1 if not batches[0][1] else self._get_clipping_scale(group, batches)
                for parameter, state, _ in batches:
                    if parameter.grad.is_sparse:
                        raise RuntimeError("ScaledAdam optimizer does not support sparse gradients")
                    if not state:
                        self._init_state(group, parameter, state)
                    self._step_one_batch(group, parameter, state, clipping_scale)
        return loss

    def _init_state(self, group: dict, parameter: Tensor, state: dict) -> None:
        period = group["size_update_period"]
        state["step"] = 0
        state["delta"] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
        if parameter.numel() > 1:
            dimensions = list(range(1, parameter.ndim))
            rms = (parameter**2).mean(dim=dimensions, keepdim=True).sqrt()
            state["param_rms"] = rms
            state["scale_exp_avg_sq"] = torch.zeros_like(rms)
            state["scale_grads"] = torch.zeros(
                period, *rms.shape, device=parameter.device, dtype=parameter.dtype
            )
        state["exp_avg_sq"] = torch.zeros_like(parameter, memory_format=torch.preserve_format)

    def _get_clipping_scale(self, group: dict, batches) -> float:
        first_parameter, first_state, _ = batches[0]
        step = first_state["step"]
        if group["clipping_scale"] is None or step == 0:
            return 1.0
        total = torch.tensor(0.0, device=first_parameter.device)
        for parameter, state, _ in batches:
            gradient = parameter.grad
            if gradient.is_sparse:
                raise RuntimeError("ScaledAdam optimizer does not support sparse gradients")
            if parameter.numel() == parameter.shape[0]:
                total += (gradient**2).sum()
            else:
                total += ((gradient * state["param_rms"]) ** 2).sum()
        norm = total.sqrt()
        period = group["clipping_update_period"]
        if "model_norms" not in first_state:
            first_state["model_norms"] = torch.zeros(period, device=first_parameter.device)
        first_state["model_norms"][step % period] = norm
        if step % period == 0:
            sorted_norms = first_state["model_norms"].sort()[0].cpu()
            quartiles = [sorted_norms[min(period - 1, period // 4 * index)].item() for index in range(5)]
            threshold = group["clipping_scale"] * quartiles[2]
            clipped = first_state.get("num_clipped", 0) * 100.0 / period
            first_state["model_norm_threshold"] = threshold
            first_state["num_clipped"] = 0
            logging.info(
                "Clipping_scale=%s, grad-norm quartiles %s, threshold=%.3e, percent-clipped=%.1f",
                group["clipping_scale"], " ".join(f"{value:.3e}" for value in quartiles), threshold, clipped
            )
        if step < period:
            return 1.0
        threshold = first_state.get("model_norm_threshold")
        if threshold is None:
            logging.info("Warning: model_norm_threshold missing from ScaledAdam state")
            return 1.0
        answer = min(1.0, (threshold / (norm + 1e-20)).item())
        if answer < 1.0:
            first_state["num_clipped"] += 1
        if answer < 0.1 and self.show_dominant_parameters:
            self._show_gradient_dominating_parameter(batches, total)
        return answer

    @staticmethod
    def _show_gradient_dominating_parameter(batches, total: Tensor) -> None:
        contributions = {}
        for parameter, state, names in batches:
            gradient = parameter.grad
            if parameter.numel() == parameter.shape[0]:
                sums = gradient**2
                rms_values = torch.ones(parameter.shape[0])
            else:
                rms_values = state["param_rms"]
                sums = ((gradient * rms_values) ** 2).sum(dim=list(range(1, gradient.ndim)))
            for name, value, rms, item_gradient in zip(names, sums, rms_values, gradient):
                contributions[name] = (value / total, value, rms, item_gradient)
        name, values = max(contributions.items(), key=lambda item: item[1][0])
        proportion, value, rms, gradient = values
        logging.info(
            "Parameter Dominating tot_sumsq %s with proportion %.2f, dominant_sumsq=%s, grad_sumsq=%s, orig_rms_sq=%s",
            name, proportion, value, (gradient**2).sum(), (rms**2).item()
        )

    def _step_one_batch(self, group: dict, parameter: Tensor, state: dict, clipping_scale: float) -> None:
        gradient = parameter.grad
        if clipping_scale != 1.0:
            gradient = gradient * clipping_scale
        step = state["step"]
        state["delta"].mul_(group["betas"][0])
        elements_per_parameter = parameter.numel() // parameter.shape[0]
        if elements_per_parameter > 1:
            period = group["size_update_period"]
            scale_grads = state["scale_grads"]
            scale_grads[step % period] = (parameter * gradient).sum(
                dim=list(range(1, parameter.ndim)), keepdim=True
            )
            if step % period == period - 1:
                state["param_rms"].copy_(
                    (parameter**2).mean(dim=list(range(1, parameter.ndim)), keepdim=True).sqrt()
                )
                if step > 0:
                    self._size_update(group, scale_grads, parameter, state)
        if elements_per_parameter == 1:
            self._step_scalar(group, parameter, state)
        else:
            self._step(group, parameter, state)
        state["step"] = step + 1

    @staticmethod
    def _size_update(group: dict, scale_grads: Tensor, parameter: Tensor, state: dict) -> None:
        beta2 = group["betas"][1]
        period = scale_grads.shape[0]
        corrected_beta2 = beta2**period
        average = state["scale_exp_avg_sq"]
        average.mul_(corrected_beta2).add_((scale_grads**2).mean(dim=0), alpha=1 - corrected_beta2)
        size_step = (state["step"] + 1) // period
        denominator = average.sqrt() + group["eps"]
        scale = -(group["lr"] * group["scalar_lr_scale"])
        scale *= (1 - corrected_beta2**size_step) ** 0.5
        update = scale * scale_grads.sum(dim=0) / denominator
        rms = state["param_rms"]
        update.masked_fill_(rms < group["param_min_rms"], 0)
        update.masked_fill_(
            rms > group["param_max_rms"],
            -group["lr"] * group["scalar_lr_scale"] * period,
        )
        state["delta"].add_(parameter * update, alpha=1 - group["betas"][0])

    @staticmethod
    def _step(group: dict, parameter: Tensor, state: dict) -> None:
        gradient = parameter.grad
        beta1, beta2 = group["betas"]
        average = state["exp_avg_sq"]
        average.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
        effective_step = state["step"] - state.get("zero_step", 0)
        bias_correction = 1 - beta2 ** (effective_step + 1)
        if bias_correction < 0.99:
            average = average / bias_correction
        denominator = average.sqrt() + group["eps"]
        alpha = -group["lr"] * (1 - beta1) * state["param_rms"].clamp(min=group["param_min_rms"])
        state["delta"].add_(gradient / denominator * alpha)
        parameter.add_(state["delta"])

    @staticmethod
    def _step_scalar(group: dict, parameter: Tensor, state: dict) -> None:
        gradient = parameter.grad
        beta1, beta2 = group["betas"]
        average = state["exp_avg_sq"]
        average.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
        denominator = (average / (1 - beta2 ** (state["step"] + 1))).sqrt() + group["eps"]
        state["delta"].add_(
            gradient / denominator,
            alpha=-group["lr"] * group["scalar_lr_scale"] * (1 - beta1),
        )
        parameter.clamp_(min=-group["scalar_max"], max=group["scalar_max"])
        parameter.add_(state["delta"])


__all__ = ["ScaledAdam"]

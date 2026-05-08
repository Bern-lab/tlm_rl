# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import copy
import math
from typing import Any

import torch
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups


class PPOTeacherKL(PPO):
    """PPO with an additional frozen-teacher KL regularization term."""

    teacher: MLPModel | None
    """The frozen teacher actor model."""

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        teacher_kl_cfg: dict | None = None,
        teacher_checkpoint_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize PPO and store teacher-KL configuration."""
        super().__init__(actor, critic, storage, **kwargs)

        self.teacher = None
        self.teacher_loaded = False
        self.teacher_kl_cfg = dict(teacher_kl_cfg or {})
        self._teacher_kl_shapes_checked = False

        # Allow checkpoint path to be supplied either inside teacher_kl_cfg or as a direct keyword.
        self.teacher_checkpoint_path: str | None = None
        self.set_teacher_checkpoint(teacher_checkpoint_path or self.teacher_kl_cfg.get("checkpoint_path"))

        self.teacher_kl_lambda_start = float(self.teacher_kl_cfg.get("lambda_start", 1.0))
        self.teacher_kl_lambda_end = float(self.teacher_kl_cfg.get("lambda_end", 0.0))
        self.teacher_kl_warmup_iters = int(self.teacher_kl_cfg.get("warmup_iters", 0))
        self.teacher_kl_constant_iters = int(self.teacher_kl_cfg.get("constant_iters", 0))
        self.teacher_kl_anneal_iters = int(self.teacher_kl_cfg.get("anneal_iters", 3000))
        self.teacher_kl_schedule = str(self.teacher_kl_cfg.get("schedule", "linear"))
        self.teacher_kl_iteration = int(self.teacher_kl_cfg.get("iteration", 0))

        self.teacher_kl_cfg.update(
            {
                "lambda_start": self.teacher_kl_lambda_start,
                "lambda_end": self.teacher_kl_lambda_end,
                "warmup_iters": self.teacher_kl_warmup_iters,
                "constant_iters": self.teacher_kl_constant_iters,
                "anneal_iters": self.teacher_kl_anneal_iters,
                "schedule": self.teacher_kl_schedule,
                "log_kl_when_lambda_zero": self.teacher_kl_cfg.get("log_kl_when_lambda_zero", True),
            }
        )

    def set_teacher_checkpoint(self, checkpoint_path: str | None) -> None:
        """Set or clear the checkpoint path used to initialize the frozen teacher."""
        self.teacher_checkpoint_path = checkpoint_path
        if checkpoint_path is None:
            self.teacher_kl_cfg.pop("checkpoint_path", None)
        else:
            self.teacher_kl_cfg["checkpoint_path"] = checkpoint_path

    def set_teacher_kl_schedule(
        self,
        lambda_start: float | None = None,
        lambda_end: float | None = None,
        warmup_iters: int | None = None,
        constant_iters: int | None = None,
        anneal_iters: int | None = None,
        schedule: str | None = None,
    ) -> None:
        """Update the teacher-KL weight schedule."""
        if lambda_start is not None:
            self.teacher_kl_lambda_start = float(lambda_start)
            self.teacher_kl_cfg["lambda_start"] = self.teacher_kl_lambda_start
        if lambda_end is not None:
            self.teacher_kl_lambda_end = float(lambda_end)
            self.teacher_kl_cfg["lambda_end"] = self.teacher_kl_lambda_end
        if warmup_iters is not None:
            self.teacher_kl_warmup_iters = int(warmup_iters)
            self.teacher_kl_cfg["warmup_iters"] = self.teacher_kl_warmup_iters
        if constant_iters is not None:
            self.teacher_kl_constant_iters = int(constant_iters)
            self.teacher_kl_cfg["constant_iters"] = self.teacher_kl_constant_iters
        if anneal_iters is not None:
            self.teacher_kl_anneal_iters = int(anneal_iters)
            self.teacher_kl_cfg["anneal_iters"] = self.teacher_kl_anneal_iters
        if schedule is not None:
            self.teacher_kl_schedule = str(schedule)
            self.teacher_kl_cfg["schedule"] = self.teacher_kl_schedule

    def get_teacher_kl_lambda(self) -> float:
        """Return the current teacher-KL loss weight.

        Warmup takes precedence over all schedules. After warmup, ``constant`` keeps
        ``lambda_start`` fixed, ``linear`` and ``cosine`` anneal toward ``lambda_end``,
        and ``constant_then_linear`` holds ``lambda_start`` before linear annealing.
        """
        if self.teacher_kl_iteration < self.teacher_kl_warmup_iters:
            return 0.0

        schedule = self.teacher_kl_schedule
        if schedule == "constant":
            return self.teacher_kl_lambda_start

        schedule_iteration = self.teacher_kl_iteration - self.teacher_kl_warmup_iters
        if schedule == "constant_then_linear":
            if schedule_iteration < self.teacher_kl_constant_iters:
                return self.teacher_kl_lambda_start
            schedule = "linear"
            schedule_iteration -= self.teacher_kl_constant_iters

        if self.teacher_kl_anneal_iters <= 0:
            return self.teacher_kl_lambda_end

        progress = min(schedule_iteration / self.teacher_kl_anneal_iters, 1.0)
        if schedule == "linear":
            alpha = progress
        elif schedule == "cosine":
            alpha = 0.5 * (1.0 - math.cos(math.pi * progress))
        else:
            raise ValueError(f"Unsupported teacher KL schedule: {self.teacher_kl_schedule}")

        return self.teacher_kl_lambda_start * (1.0 - alpha) + self.teacher_kl_lambda_end * alpha

    def _freeze_teacher(self) -> None:
        """Keep the teacher actor in inference mode and out of gradient updates."""
        if self.teacher is None:
            return

        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad_(False)

    def load_teacher_checkpoint(self, checkpoint_path: str | None = None, strict: bool = True) -> None:
        """Load the frozen teacher actor from an rsl_rl PPO checkpoint."""
        if self.teacher is None:
            raise RuntimeError("Cannot load a teacher checkpoint before constructing the teacher model.")

        if checkpoint_path is not None:
            self.set_teacher_checkpoint(checkpoint_path)
        if self.teacher_checkpoint_path is None:
            raise ValueError("teacher_kl_cfg.checkpoint_path is required for PPOTeacherKL.")

        loaded_dict = torch.load(self.teacher_checkpoint_path, weights_only=False, map_location=self.device)
        if "actor_state_dict" not in loaded_dict:
            raise KeyError(
                f"Cannot find 'actor_state_dict' in teacher checkpoint: {self.teacher_checkpoint_path}"
            )

        self.teacher.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
        self.teacher_loaded = True
        self._teacher_kl_shapes_checked = False
        self._freeze_teacher()
        print(
            "Loaded frozen teacher actor from "
            f"'{self.teacher_checkpoint_path}' with obs groups {self.teacher.obs_groups}."
        )

    def _validate_distribution_params(
        self,
        teacher_params: tuple[torch.Tensor, ...],
        student_params: tuple[torch.Tensor, ...],
    ) -> None:
        """Validate teacher and student distribution parameter compatibility."""
        if len(teacher_params) != len(student_params):
            raise RuntimeError(
                "Teacher/student distribution parameter count mismatch: "
                f"teacher={len(teacher_params)}, student={len(student_params)}"
            )

        for index, (teacher_param, student_param) in enumerate(zip(teacher_params, student_params)):
            if teacher_param.shape != student_param.shape:
                raise RuntimeError(
                    f"Teacher/student distribution parameter {index} shape mismatch: "
                    f"teacher={tuple(teacher_param.shape)}, student={tuple(student_param.shape)}"
                )

        if self.teacher_kl_cfg.get("debug_shapes", False):
            shape_summary = [tuple(param.shape) for param in teacher_params]
            print(f"Teacher/student distribution parameter shapes verified: {shape_summary}")

    def _distributed_mean_scalar(self, value: torch.Tensor) -> torch.Tensor:
        """Average a scalar tensor across distributed workers for logging."""
        value = value.detach()
        if self.is_multi_gpu:
            value = value.clone()
            torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
            value /= self.gpu_world_size
        return value

    def _compute_additional_loss(
        self,
        batch: RolloutStorage.Batch,
        original_batch_size: int,
        distribution_params: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the frozen-teacher KL regularization loss."""
        if self.teacher is None or not self.teacher_loaded:
            raise RuntimeError("Teacher KL loss requires a loaded teacher model.")
        if batch.observations is None:
            raise RuntimeError("Teacher KL loss requires observations in the rollout batch.")

        teacher_kl_lambda = self.get_teacher_kl_lambda()
        if teacher_kl_lambda == 0.0 and not self.teacher_kl_cfg.get("log_kl_when_lambda_zero", True):
            return torch.zeros((), device=self.device), {
                "teacher_kl": 0.0,
                "teacher_kl_for_loss": 0.0,
                "teacher_kl_loss": 0.0,
                "teacher_kl_lambda": 0.0,
            }

        observations = batch.observations
        teacher_observations = observations[:original_batch_size]
        teacher_masks = batch.masks[:original_batch_size] if batch.masks is not None else None

        with torch.no_grad():
            self.teacher(teacher_observations, masks=teacher_masks, stochastic_output=True)
            teacher_distribution_params = tuple(p.detach() for p in self.teacher.output_distribution_params)

        student_distribution_params = distribution_params
        if teacher_kl_lambda == 0.0:
            student_distribution_params = tuple(p.detach() for p in student_distribution_params)

        if self.teacher_kl_cfg.get("check_shapes", True) and not self._teacher_kl_shapes_checked:
            self._validate_distribution_params(teacher_distribution_params, student_distribution_params)
            self._teacher_kl_shapes_checked = True

        teacher_kl = self.actor.get_kl_divergence(teacher_distribution_params, student_distribution_params).mean()
        if self.teacher_kl_cfg.get("fail_on_nonfinite_kl", True) and not torch.isfinite(teacher_kl):
            raise FloatingPointError(
                f"Non-finite teacher KL detected: {teacher_kl.item()}. "
                "Check teacher observation normalization, action distribution parameters, and checkpoint compatibility."
            )

        max_kl_loss = self.teacher_kl_cfg.get("max_kl_loss")
        if max_kl_loss is not None:
            teacher_kl_for_loss = teacher_kl.clamp(max=float(max_kl_loss))
        else:
            teacher_kl_for_loss = teacher_kl

        teacher_kl_loss = teacher_kl_lambda * teacher_kl_for_loss
        teacher_kl_log = self._distributed_mean_scalar(teacher_kl)
        teacher_kl_for_loss_log = self._distributed_mean_scalar(teacher_kl_for_loss)
        teacher_kl_loss_log = self._distributed_mean_scalar(teacher_kl_loss)

        return teacher_kl_loss, {
            "teacher_kl": teacher_kl_log.item(),
            "teacher_kl_for_loss": teacher_kl_for_loss_log.item(),
            "teacher_kl_loss": teacher_kl_loss_log.item(),
            "teacher_kl_lambda": float(teacher_kl_lambda),
        }

    def update(self) -> dict[str, float]:
        """Run a PPO update and advance the teacher-KL schedule."""
        loss_dict = super().update()
        self.teacher_kl_iteration += 1
        return loss_dict

    def save(self) -> dict:
        """Return a dict of all learnable models and teacher-KL training state."""
        saved_dict = super().save()
        saved_dict["teacher_kl_iteration"] = self.teacher_kl_iteration
        saved_dict["teacher_kl_cfg"] = dict(self.teacher_kl_cfg)
        saved_dict["teacher_checkpoint_path"] = self.teacher_checkpoint_path
        saved_dict["teacher_kl_lambda_current"] = self.get_teacher_kl_lambda()
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load models and restore the teacher-KL schedule state."""
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        if load_iteration and "teacher_kl_iteration" in loaded_dict:
            self.teacher_kl_iteration = int(loaded_dict["teacher_kl_iteration"])
        return load_iteration

    def train_mode(self) -> None:
        """Set train mode for learnable models while keeping the teacher frozen."""
        super().train_mode()
        self._freeze_teacher()

    def eval_mode(self) -> None:
        """Set evaluation mode for all policy models."""
        super().eval_mode()
        self._freeze_teacher()

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> "PPOTeacherKL":
        """Construct the PPO + teacher-KL algorithm."""
        algorithm_cfg = copy.deepcopy(cfg["algorithm"])
        actor_cfg = copy.deepcopy(cfg["actor"])
        critic_cfg = copy.deepcopy(cfg["critic"])
        teacher_cfg = copy.deepcopy(cfg.get("teacher", cfg["actor"]))
        obs_groups_cfg = copy.deepcopy(cfg["obs_groups"])

        # Resolve class callables
        alg_class: type[PPOTeacherKL] = resolve_callable(algorithm_cfg.pop("class_name"))  # type: ignore
        actor_class: type[MLPModel] = resolve_callable(actor_cfg.pop("class_name"))  # type: ignore
        critic_class: type[MLPModel] = resolve_callable(critic_cfg.pop("class_name"))  # type: ignore
        teacher_class_name = teacher_cfg.pop("class_name", None)
        teacher_class: type[MLPModel] = resolve_callable(teacher_class_name) if teacher_class_name else actor_class

        # Resolve observation groups
        default_sets = ["actor", "critic", "teacher"]
        if "rnd_cfg" in algorithm_cfg and algorithm_cfg["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        obs_groups = resolve_obs_groups(obs, obs_groups_cfg, default_sets)

        # Resolve RND config if used
        algorithm_cfg = resolve_rnd_config(algorithm_cfg, obs, obs_groups, env)

        # Resolve symmetry config if used
        algorithm_cfg = resolve_symmetry_config(algorithm_cfg, env)

        # Initialize the student actor, critic, and frozen teacher actor
        actor: MLPModel = actor_class(obs, obs_groups, "actor", env.num_actions, **actor_cfg).to(device)
        print(f"Actor Model: {actor}")
        if algorithm_cfg.pop("share_cnn_encoders", None):  # Share CNN encoders between actor and critic
            critic_cfg["cnns"] = actor.cnns  # type: ignore
        critic: MLPModel = critic_class(obs, obs_groups, "critic", 1, **critic_cfg).to(device)
        print(f"Critic Model: {critic}")
        teacher: MLPModel = teacher_class(obs, obs_groups, "teacher", env.num_actions, **teacher_cfg).to(device)
        if teacher.is_recurrent:
            raise ValueError("PPOTeacherKL currently supports feedforward teacher actors only.")
        print(f"Teacher Model: {teacher}")

        # Initialize the storage
        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)

        # Initialize the algorithm. The teacher is intentionally assigned after construction so it is not part of the
        # PPO optimizer created by the parent class.
        alg: PPOTeacherKL = alg_class(
            actor, critic, storage, device=device, **algorithm_cfg, multi_gpu_cfg=cfg["multi_gpu"]
        )
        alg.teacher = teacher
        alg.load_teacher_checkpoint()

        # Compile the algorithm's learnable models if requested. The teacher remains an uncompiled frozen reference.
        alg.compile(cfg.get("torch_compile_mode"))

        return alg

# TLM-RL: PPO Teacher-KL Regularization

**TLM-RL** is a small fork of [RSL-RL](https://github.com/leggedrobotics/rsl_rl) focused on PPO training with frozen
teacher KL regularization.

The added training path keeps the student actor in the normal PPO loop: the student samples actions, controls the
environment, stores rollouts, and is optimized by PPO. During `update()`, a frozen teacher actor reads the `teacher`
observation group, computes its action distribution, and adds:

```text
loss += lambda(t) * KL(teacher || student)
```

## Main Changes

- Added `PPOTeacherKL`, a PPO variant with a frozen teacher actor.
- Added teacher checkpoint loading from `actor_state_dict`.
- Added `actor` / `critic` / `teacher` observation group support.
- Added configurable teacher-KL schedules: `linear`, `cosine`, `constant`, and `constant_then_linear`.
- Added warmup, KL clipping, finite checks, distribution-shape checks, and teacher-KL logging.
- Added checkpoint metadata for teacher-KL state and resume-safe KL schedule progress.

## Expected Observation Interface

The environment should provide stable observation keys:

```text
obs["actor"]    -> student actor input
obs["critic"]   -> critic input
obs["teacher"]  -> frozen teacher actor input
```

The teacher observation must match the actor observation used when training the teacher checkpoint: same terms, order,
dimensions, scaling, clipping, history length, and sensor settings.

## Example Configuration

```yaml
obs_groups:
  actor: ["actor"]
  critic: ["critic"]
  teacher: ["teacher"]

algorithm:
  class_name: PPOTeacherKL
  teacher_kl_cfg:
    checkpoint_path: /abs/path/to/teacher/model.pt
    lambda_start: 1.0
    lambda_end: 0.0
    warmup_iters: 0
    constant_iters: 0
    anneal_iters: 3000
    schedule: cosine
    max_kl_loss: 10.0
    check_shapes: true
    fail_on_nonfinite_kl: true
    log_kl_when_lambda_zero: true
```

## Download

For the upstream RSL-RL package and installation instructions, see
[leggedrobotics/rsl_rl](https://github.com/leggedrobotics/rsl_rl).

## Roadmap

Transformer-based policy components are expected to be made public next month.

## License

This project follows the BSD 3-Clause license inherited from RSL-RL. See [LICENSE](LICENSE).

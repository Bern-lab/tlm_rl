# TLM-RL: Modified RSL-RL

**TLM-RL** is a modified version of [RSL-RL](https://github.com/leggedrobotics/rsl_rl), a GPU-accelerated, lightweight
learning library for robotics research. This fork keeps the compact PPO training pipeline of RSL-RL while focusing on
sequence-aware policy and value networks.

The main modification is replacing the default feed-forward MLP policy/value networks with Transformer- and LSTM-based
architectures. This is useful for tasks where the policy benefits from temporal context, partial observability, or
history-dependent robot behavior.

## Key Features

- **Minimal, readable codebase** with clear extension points for rapid prototyping.
- **Robotics-first methods** including PPO and Student-Teacher Distillation.
- **High-throughput training** with native Multi-GPU support.
- **Sequence-aware models** using Transformer and LSTM architectures instead of the default MLP networks.
- **Proven performance** in numerous research publications.

## Learning Environments

RSL-RL is currently used by the following robot learning libraries:

- [Isaac Lab](https://github.com/isaac-sim/IsaacLab) (built on top of NVIDIA Isaac Sim)
- [Legged Gym](https://github.com/leggedrobotics/legged_gym) (built on top of NVIDIA Isaac Gym)
- [mjlab](https://github.com/mujocolab/mjlab) (built on top of MuJoCo Warp)
- [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) (built on top of MuJoCo MJX and Warp)

## Installation

Before installing RSL-RL, ensure that Python `3.9+` is available. It is recommended to install the library in a virtual
environment (e.g. using `venv` or `conda`), which is often already created by the used environment library (e.g.
Isaac Lab). If so, make sure to activate it before installing RSL-RL.

### Installing the original RSL-RL package

```bash
pip install rsl-rl-lib
```

### Installing this modified version for development

```bash
git clone https://github.com/Bern-lab/tlm_rl.git
cd tlm_rl
pip install -e .
```

## Citation

If you use RSL-RL in your research, please cite the [paper](https://arxiv.org/abs/2509.10771):

```text
@article{schwarke2025rslrl,
  title={RSL-RL: A Learning Library for Robotics Research},
  author={Schwarke, Clemens and Mittal, Mayank and Rudin, Nikita and Hoeller, David and Hutter, Marco},
  journal={arXiv preprint arXiv:2509.10771},
  year={2025}
}
```

## License

This project is a modified version of RSL-RL and follows the BSD 3-Clause license. See [LICENSE](LICENSE) for the full
license text and `licenses/dependencies/` for third-party dependency license information.

Please retain the original RSL-RL copyright and citation notices when redistributing or publishing work based on this
repository.

"""Opt-in HD position-action contract; stock/raw policies are unchanged.

Training history and exported output both represent the applied, bounded delta
from HOME. PPO still optimizes its latent Gaussian actions through clipping.
"""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import onnx
import torch
from onnx import helper, numpy_helper
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

CONTRACT = 'bounded_home_delta_v1'
SLEW_CONTRACT = 'bounded_slew_home_delta_v2'


class BoundedPositionAction(JointPositionAction):
    def process_actions(self, actions):
        step = getattr(self.cfg, 'max_step_rad', None)
        previous = self._raw_actions.clone() if step is not None else None
        if step is not None:
            self.previous_step = getattr(self, 'applied_step', previous * 0).clone()
        super().process_actions(actions)
        # last_action(action_name=...) must match the exported graph's output.
        self._raw_actions[:] = self._processed_actions - self._offset
        self.range_delta = self._raw_actions.clone()
        if step is not None:
            self.previous_delta = previous
            self._raw_actions[:] = self._raw_actions.clamp(previous-step, previous+step)
            self._processed_actions[:] = self._raw_actions + self._offset
            self.applied_step = self._raw_actions - previous
            if getattr(self.cfg, 'command_hold_max_steps', 0):
                start = (self._hold_left == 0) & (torch.rand_like(self._hold_rate) < self._hold_rate)
                durations = torch.randint(1, self.cfg.command_hold_max_steps + 1,
                    self._hold_left.shape, device=self._hold_left.device)
                self._hold_left = torch.where(start, durations, self._hold_left)
                hold = self._hold_left > 0
                self._hold_left[hold] -= 1
                self._raw_actions[hold] = previous[hold]
                self._processed_actions[hold] = previous[hold] + self._offset[hold]
                self.applied_step[hold] = 0.

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if getattr(self.cfg, 'command_hold_max_steps', 0):
            if not hasattr(self, '_hold_left'):
                self._hold_left = torch.zeros(self._raw_actions.shape[0], dtype=torch.long, device=self._raw_actions.device)
                self._hold_rate = torch.zeros_like(self._hold_left, dtype=torch.float32)
            ids = slice(None) if env_ids is None else env_ids
            self._hold_left[ids] = 0
            low, high = self.cfg.command_loss_probability_range
            self._hold_rate[ids] = low + (high - low) * torch.rand_like(self._hold_rate[ids])
        if getattr(self.cfg, 'reset_from_joint_state', False):
            ids = slice(None) if env_ids is None else env_ids
            position = self._entity.data.joint_pos[:, self._target_ids]
            bounded = position.clamp(self._clip[:, :, 0], self._clip[:, :, 1])
            self._raw_actions[ids] = (bounded-self._offset)[ids]
            self._processed_actions[ids] = bounded[ids]
        for name in ('previous_step', 'applied_step'):
            if hasattr(self, name):
                getattr(self, name)[slice(None) if env_ids is None else env_ids] = 0


@dataclass(kw_only=True)
class BoundedPositionActionCfg(JointPositionActionCfg):
    max_step_rad: float | None = None
    reset_from_joint_state: bool = False
    command_loss_probability_range: tuple[float, float] = (0., 0.)
    command_hold_max_steps: int = 0

    def __post_init__(self):
        super().__post_init__()
        if self.max_step_rad is not None and not 0 < self.max_step_rad <= .12:
            raise ValueError('max_step_rad must be finite and within (0, 0.12]')
        low, high = self.command_loss_probability_range
        if not 0 <= low <= high < .5 or not 0 <= self.command_hold_max_steps <= 5:
            raise ValueError('invalid command hold randomization')

    def build(self, env):
        return BoundedPositionAction(self, env)


def bound_export(path, term):
    """Append the exact training action transform after the normalized actor."""
    if not isinstance(term, BoundedPositionAction):
        return
    low = (term._clip[0, :, 0] - term._offset[0]).detach().cpu().numpy().astype(np.float32)
    high = (term._clip[0, :, 1] - term._offset[0]).detach().cpu().numpy().astype(np.float32)
    model = onnx.load(path)
    name = model.graph.output[0].name
    raw = name + '_unbounded'
    if any(n.name == 'hd1910_bound_output' for n in model.graph.node):
        raise ValueError('bounded export already applied')
    for node in model.graph.node:
        for i, value in enumerate(node.output):
            if value == name:
                node.output[i] = raw
        for i, value in enumerate(node.input):
            if value == name:
                node.input[i] = raw
    model.graph.initializer.extend([numpy_helper.from_array(low, 'hd_delta_low'),
                                    numpy_helper.from_array(high, 'hd_delta_high')])
    # ONNX Clip takes scalar limits; Min/Max support per-joint broadcasting.
    step = term.cfg.max_step_rad if hasattr(term.cfg, 'max_step_rad') else None
    bounded = name if step is None else 'hd_bounded_delta'
    model.graph.node.extend([
        helper.make_node('Max', [raw, 'hd_delta_low'], [raw+'_lower'], name='hd1910_bound_lower'),
        helper.make_node('Min', [raw+'_lower', 'hd_delta_high'], [bounded], name='hd1910_bound_output'),
    ])
    if step is not None:
        # Use the unnormalized observation history, identical to the action term.
        for key, value in [('hd_history_ids', np.arange(34,48,dtype=np.int64)),
                           ('hd_step', np.asarray(step,dtype=np.float32))]:
            model.graph.initializer.append(numpy_helper.from_array(value,key))
        model.graph.node.extend([
            helper.make_node('Gather', [model.graph.input[0].name,'hd_history_ids'], ['hd_history'], axis=1),
            helper.make_node('Sub', ['hd_history','hd_step'], ['hd_step_low']),
            helper.make_node('Add', ['hd_history','hd_step'], ['hd_step_high']),
            helper.make_node('Max', [bounded,'hd_step_low'], ['hd_slew_lower']),
            helper.make_node('Min', ['hd_slew_lower','hd_step_high'], [name]),
        ])
    metadata = {p.key: p.value for p in model.metadata_props}
    metadata.update(action_semantics=CONTRACT, previous_action_semantics=CONTRACT,
                    action_delta_low=json.dumps(low.tolist()), action_delta_high=json.dumps(high.tolist()),
                    deployment_ready='false', hardware_tested='false')
    if step is not None:
        metadata.update(action_semantics=SLEW_CONTRACT, previous_action_semantics=SLEW_CONTRACT,
                        max_action_step_rad=str(step), policy_period_s='0.02')
    helper.set_model_props(model, metadata)
    onnx.checker.check_model(model)
    onnx.save(model, Path(path))

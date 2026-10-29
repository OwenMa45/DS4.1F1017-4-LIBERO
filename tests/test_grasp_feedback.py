import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from ds_libero.artifacts import Store
from ds_libero.client import Client, ModelPolicy
from ds_libero.config import Config
from ds_libero.errors import ExperimentError
from ds_libero.feedback import decision_feedback, feedback_prefix, gripper_motion_diagnostics
from ds_libero.geometry import camera_calibration, observe_gripper_geometry
from ds_libero.protocol import system_prompt
from ds_libero.report import build_report
from test_experiment import FakeEnv, FixedPolicy, MemoryStore, observation, reply, run_quiet

CLOSE = [0., 0., 0., 0., 0., 0., 1.]
OPEN = [0., 0., 0., 0., 0., 0., -1.]
LIFT = [0., 0., .2, 0., 0., 0., 1.]
APPROACH = [.1, 0., -.1, 0., 0., 0., -1.]


class GraspFeedbackTests(unittest.TestCase):
    def test_old_profile_hashes_stay_exact_and_v1_cannot_enable_new_features(self):
        self.assertEqual(Config().fingerprint, '911c4f579ce0bd922aee8e2472ddb658b58fdf42f3fd76cd00be3887017b37ff')
        self.assertEqual(Config.load('configs/direct_v2_high.json').fingerprint,
                         '234d3b33d5757faffc5490cdcb7fbb5245f3ad476c4ae64f788de5bb31aed3fb')
        with self.assertRaises(ExperimentError):
            Config.from_dict({'grasp_feedback': True})
        for name in ('direct_v2_grasp_high', 'v2_grasp_feedback_only_high'):
            cfg = Config.load('configs/' + name + '.json')
            self.assertEqual(cfg.reasoning_effort, 'high')
            self.assertIsNone(cfg.max_tokens)
            self.assertEqual(Config.from_dict(cfg.to_dict()), cfg)

    def test_boundary_splits_close_lift_and_approach_close_without_inventing_actions(self):
        for actions, expected in (([CLOSE]*2+[LIFT]*2, 2), ([APPROACH]*2+[CLOSE]*4, 2),
                                   ([OPEN]*2+[CLOSE]*2, 2), ([CLOSE]*8, 8), ([LIFT]*8, 8),
                                   ([CLOSE, [0.]*7, LIFT], 2)):
            with self.subTest(actions=actions):
                self.assertEqual(feedback_prefix(actions)[0], expected)

    def test_invalid_suffix_is_rejected_before_any_prefix_executes(self):
        cfg = replace(Config.load('configs/direct_v2_grasp_high.json'), max_steps=8)
        env = FakeEnv()
        report = run_quiet(env, FixedPolicy([CLOSE, [2.] + [0.]*6]), MemoryStore(), cfg)
        self.assertEqual(report['status'], 'invalid_output')
        self.assertEqual(env.steps, 0)

    def test_new_observation_is_requested_before_lifting_and_suffix_is_discarded(self):
        cfg = Config.load('configs/direct_v2_grasp_high.json')
        class Env(FakeEnv):
            def reset(self):
                return replace(super().reset(), geometry={'cameras': {}})
            def step(self, action):
                obs, done = super().step(action)
                return replace(obs, geometry={'cameras': {}}), done
        env = Env(success_at=3)
        payloads = []
        def send(endpoint, body, key, timeout):
            payloads.append(json.loads(body))
            self.assertEqual(env.steps, 0 if len(payloads) == 1 else 2)
            return reply(json.dumps({'actions': [CLOSE]*2+[LIFT]*2 if len(payloads) == 1 else [LIFT]}))
        with tempfile.TemporaryDirectory() as temp:
            store = Store(Path(temp)/'run', 'run', cfg)
            policy = ModelPolicy(cfg, 'task', Client(cfg, store, api_key='test', send=send), store)
            result = run_quiet(env, policy, store, cfg)
            store.finish(result)
            report = build_report(store.path, replay=False)
            self.assertEqual(env.actions, [CLOSE, CLOSE, LIFT])
            self.assertEqual(result['policy_decisions'], 2)
            self.assertEqual(report['feedback_boundaries'], 1)
            self.assertEqual(report['discarded_actions_at_feedback_boundaries'], 2)
            self.assertEqual(report['executed_actions_per_decision']['total'], 3)
            feedback = json.loads(payloads[1]['messages'][1]['content'][0]['text'])['control_feedback']
            self.assertEqual(feedback['previous_decision']['executed_actions'], [CLOSE, CLOSE])
            self.assertEqual(feedback['previous_decision']['discarded_actions'], 2)
            self.assertEqual(feedback['previous_decision']['simulated_seconds'], .1)

    def test_disabled_boundary_preserves_full_old_chunk(self):
        cfg = Config.load('configs/direct_v2_high.json')
        env = FakeEnv(success_at=4)
        result = run_quiet(env, FixedPolicy([CLOSE]*2+[LIFT]*2), MemoryStore(), cfg)
        self.assertEqual(result['policy_decisions'], 1)
        self.assertEqual(env.actions, [CLOSE]*2+[LIFT]*2)

    def test_measured_feedback_uses_actual_displacement_not_requested_target(self):
        before = observation(0)
        after = observation(3)
        result = decision_feedback(after, 3, (0, before, [APPROACH]*8), 20)
        self.assertEqual(result['previous_decision']['eef_delta_world_m'], [.003, 0., 0.])
        self.assertEqual(result['previous_decision']['discarded_actions'], 5)
        with self.assertRaises(ValueError):
            decision_feedback(after, 9, (0, before, [CLOSE]*8), 20)

    def test_robot_geometry_reads_only_named_fingerpads_and_projects_with_camera(self):
        import numpy as np
        names = []
        def geom_id(name):
            names.append(name)
            return {'robot_left_pad': 0, 'robot_right_pad': 1}[name]
        sim = SimpleNamespace(model=SimpleNamespace(geom_name2id=geom_id),
                              data=SimpleNamespace(geom_xpos=np.array([[-.02,0,-1],[.02,0,-1]])))
        gripper = SimpleNamespace(important_geoms={'left_fingerpad':['robot_left_pad'], 'right_fingerpad':['robot_right_pad']})
        cal = camera_calibration([0,0,0], np.eye(3), 90, 256)
        geometry = observe_gripper_geometry(sim, gripper, {'cameras':{'agentview':cal}})
        self.assertEqual(names, ['robot_left_pad', 'robot_right_pad'])
        self.assertEqual(geometry['points_world_m']['pad_midpoint'], [0.,0.,-1.])
        self.assertEqual(geometry['pixels_uv']['agentview']['pad_midpoint'], [128.,128.])
        self.assertAlmostEqual(geometry['pad_centre_distance_m'], .04)

    def test_robot_geometry_rejects_nonfinite_state(self):
        import numpy as np
        sim = SimpleNamespace(model=SimpleNamespace(geom_name2id=lambda _:0),
                              data=SimpleNamespace(geom_xpos=[[float('nan'),0,0]]))
        gripper = SimpleNamespace(important_geoms={'left_fingerpad':['pad']})
        with self.assertRaises(ValueError):
            observe_gripper_geometry(sim, gripper, {'cameras':{}})

    def test_command_diagnostics_do_not_label_grasp_success(self):
        state = observation().state
        events = [{'event':'step','step':i,'action':a,'before':state,'after':state}
                  for i,a in enumerate([CLOSE,CLOSE,LIFT,OPEN,CLOSE])]
        result = gripper_motion_diagnostics(events)
        self.assertEqual(len(result['closing_bouts']), 2)
        self.assertEqual(result['closing_bouts'][0]['stationary_close_steps_before_first_arm_motion'], 2)
        self.assertEqual(result['closing_bouts'][0]['first_arm_motion_step'], 2)
        self.assertIsNone(result['closing_bouts'][1]['first_arm_motion_step'])
        self.assertNotIn('grasp_success', result)


if __name__ == '__main__':
    unittest.main()

import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ds_libero.artifacts import Store
from ds_libero.client import Client, HTTPResponse, ModelPolicy
from ds_libero.config import Config
from ds_libero.environment import LiberoEnvironment
from ds_libero.errors import ExperimentError
from ds_libero.protocol import request_payload
from ds_libero.report import build_campaign_report, build_report
from test_accounting import usage
from test_experiment import FakeEnv, FixedPolicy, MemoryStore, observation, run_quiet
from test_geometry_contract import fixture


PROFILE = 'configs/direct_v2_geometry_chunk16_high.json'
ACTION = [0., 0., 0., 0., 0., 0., -1.]


class LongChunkTests(unittest.TestCase):
    def test_extended_pilot_does_not_change_v1_or_benchmark_limits(self):
        cfg = Config.load(PROFILE)
        self.assertEqual((cfg.action_horizon, cfg.max_steps), (16, 2048))
        self.assertEqual(cfg.max_http_requests, cfg.max_steps * (cfg.max_retries + 1))
        for values in ({**cfg.to_dict(), 'action_horizon': 17},
                       {**cfg.to_dict(), 'max_steps': 2049},
                       {**cfg.to_dict(), 'max_steps': 0},
                       {**cfg.to_dict(), 'task_id': 1},
                       {**cfg.to_dict(), 'evaluation_scope': 'benchmark'},
                       {'max_steps': 2048}, {'action_horizon': 16}):
            with self.subTest(values=values), self.assertRaises(ExperimentError):
                Config.from_dict(values)
        self.assertEqual(Config().fingerprint,
                         '911c4f579ce0bd922aee8e2472ddb658b58fdf42f3fd76cd00be3887017b37ff')

    def test_2048_actions_with_settling_cross_old_simulator_horizon(self):
        cfg = Config.load(PROFILE)
        adapter = LiberoEnvironment.__new__(LiberoEnvironment)
        adapter.cfg, adapter.np = cfg, np
        adapter.initial_state = np.zeros(1)
        adapter._observe = lambda raw: raw

        class Simulator(FakeEnv):
            def __init__(self, **kwargs):
                super().__init__()
                self.horizon = kwargs['horizon']
                self.ignore_done = kwargs['ignore_done']
            def seed(self, seed): pass
            def set_init_state(self, state): return observation()
            def step(self, action):
                if self.steps >= self.horizon:
                    raise RuntimeError('Simulator already terminated')
                super().step(action)
                return observation(self.steps - cfg.settle_steps), 0., self.steps >= self.horizon, {}

        adapter.env = adapter._create_simulator(Simulator, 'unused.bddl')
        self.assertEqual(adapter.env.horizon, 2059)
        self.assertFalse(adapter.env.ignore_done)
        seen = []

        class Policy:
            def act(inner, obs, history, step, refs):
                self.assertAlmostEqual(obs.state['eef_position_world_m'][0], step * .001)
                if step:
                    self.assertEqual(history[-1]['step'], step - 1)
                seen.append(step)
                return [ACTION] * min(16, cfg.max_steps - step)

        result = run_quiet(adapter, Policy(), MemoryStore(), cfg)
        self.assertEqual(seen, list(range(0, 2048, 16)))
        self.assertEqual(adapter.env.steps, 2058)  # Includes 10 unscored settling steps.
        self.assertEqual((result['status'], result['reason']), ('failure', 'control_step_limit'))
        self.assertEqual((result['control_steps'], result['policy_decisions']), (2048, 128))
        self.assertEqual(result['simulated_control_seconds'], 102.4)
        adapter.cfg = Config.load('configs/direct_v2_geometry_high.json')
        self.assertEqual(adapter._create_simulator(Simulator, 'unused.bddl').horizon, 1000)

    def test_last_array_is_validated_before_any_action_and_success_stops_chunk(self):
        cfg = Config.load(PROFILE)
        for actions in ([ACTION] * 17, [ACTION] * 15 + [[2.] + [0.] * 6]):
            env = FakeEnv()
            result = run_quiet(env, FixedPolicy(actions), MemoryStore(), cfg)
            self.assertEqual((result['status'], env.steps), ('invalid_output', 0))
        result = run_quiet(FakeEnv(success_at=9), FixedPolicy([ACTION] * 16), MemoryStore(), cfg)
        self.assertEqual((result['status'], result['control_steps'], result['policy_decisions']),
                         ('success', 9, 1))

    def test_phase_boundary_reobserves_and_does_not_replay_suffix(self):
        cfg = Config.load(PROFILE)
        seen = []
        close = [0.] * 6 + [1.]
        lift = [0., 0., .1, 0., 0., 0., 1.]
        class Policy:
            def act(inner, obs, history, step, refs):
                seen.append(step)
                return [close] * 10 + [lift] * 6 if step == 0 else [close]
        env, store = FakeEnv(success_at=11), MemoryStore()
        result = run_quiet(env, Policy(), store, cfg)
        self.assertEqual(result['status'], 'success')
        self.assertEqual(seen, [0, 10])
        self.assertEqual(env.actions, [close] * 11)
        self.assertEqual(next(e for e in store.events if e['event'] == 'chunk_feedback_boundary')
                         ['discarded_actions'], 6)

    def test_chunk_cost_latency_retry_and_final_remaining_budget(self):
        cfg = replace(Config.load(PROFILE), max_steps=20)
        adapter, raw, _, _ = fixture()
        base_obs = adapter._observe(raw)
        class Clock:
            value = 0.
            def now(self): return self.value
            def advance(self, seconds): self.value += seconds
        clock = Clock()
        class Env(FakeEnv):
            def reset(self):
                super().reset(); clock.advance(5.)
                return copy.deepcopy(base_obs)
            def step(self, action):
                super().step(action); clock.advance(.25)
                obs = copy.deepcopy(base_obs)
                obs.state['eef_position_world_m'][0] += self.steps * .001
                return obs, False
        env, requests = Env(), []
        with tempfile.TemporaryDirectory() as temp:
            store = Store(Path(temp) / 'episode', 'run', cfg)
            def send(endpoint, body, key, timeout):
                payload = json.loads(body)
                self.assertEqual(payload['reasoning_effort'], 'high')
                self.assertNotIn('max_tokens', payload)
                data = json.loads(payload['messages'][1]['content'][0]['text'])
                self.assertEqual(data['control_step'], env.steps)
                requests.append(payload)
                clock.advance(10.)
                if len(requests) == 1:
                    return HTTPResponse(503, b'{}')
                actions = [ACTION] * data['max_actions_this_response']
                return HTTPResponse(200, json.dumps({'model': 'deepseek-flash', 'usage': usage(),
                    'choices': [{'finish_reason': 'stop', 'message': {
                        'content': json.dumps({'actions': actions})}}]}).encode())
            with patch('time.perf_counter', side_effect=clock.now):
                client = Client(cfg, store, api_key='dummy', send=send, sleep=clock.advance)
                result = run_quiet(env, ModelPolicy(cfg, 'task', client, store), store, cfg)
            self.assertEqual(requests[0], requests[1])
            latest = json.loads(requests[-1]['messages'][1]['content'][0]['text'])
            self.assertEqual(latest['max_actions_this_response'], 4)
            self.assertEqual(latest['control_feedback']['previous_decision']['executed_steps'], 16)
            self.assertEqual(latest['recent_history'][-1]['step'], 15)
            self.assertEqual(result['status'], 'failure')
            store.finish(result)
            report = build_report(store.path, replay=False)
            self.assertEqual((report['http_requests'], report['policy_decisions']), (3, 2))
            self.assertEqual(report['executed_action_count'], 20)
            self.assertEqual(report['task_execution_wall_seconds'], 37.)
            self.assertEqual(report['http_latency_seconds']['total'], 30.)
            self.assertEqual(report['decision_latency_seconds']['total'], 32.)
            self.assertEqual(report['retry_backoff_seconds'], 2.)
            self.assertEqual(report['reported_usage']['total_tokens'], 35222)
            self.assertTrue(report['usage_incomplete'])  # Failed HTTP attempt has unknown usage.
            self.assertGreater(report['api_cost']['known_cost_range_cny'][0], 0)
            self.assertEqual(report['executed_actions_per_decision']['mean'], 10.)
            campaign = build_campaign_report(temp)
            self.assertEqual(campaign['reported_usage'], report['reported_usage'])
            self.assertEqual(campaign['api_cost'], report['api_cost'])
            self.assertEqual(campaign['api_wait_seconds'], 30.)
            self.assertTrue((store.path / 'api_calls.csv').is_file())
        payload = request_payload(Config.load(PROFILE), 'task', base_obs, [], 2047)
        self.assertEqual(json.loads(payload['messages'][1]['content'][0]['text'])['max_actions_this_response'], 1)


if __name__ == '__main__':
    unittest.main()

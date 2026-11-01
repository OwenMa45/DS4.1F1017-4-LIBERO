import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ds_libero.__main__ import main
from ds_libero.artifacts import Store, write_json
from ds_libero.benchmark import create_five_plan, load_plan, report_plan, run_plan_all
from ds_libero.config import Config, digest
from ds_libero.environment import LiberoEnvironment, verify_task_name
from ds_libero.errors import ExperimentError
from ds_libero.five_tasks import FIVE_TASKS
from ds_libero.protocol import request_payload, system_prompt
from test_accounting import attempt
from test_experiment import FakeEnv, FixedPolicy, MemoryStore, run_quiet
from test_geometry_contract import fixture


PROFILE = 'configs/direct_v2_geometry_1024_high.json'
ACTION = [0., 0., 0., 0., 0., 0., -1.]


class FiveTaskTests(unittest.TestCase):
    def make_plan(self, root):
        return create_five_plan(Config.load(PROFILE), root)

    def save(self, root, entry, status, phase='episode'):
        cfg = Config.from_dict(entry['config'])
        store = Store(root/entry['run_directory']/phase, 'run' if phase == 'episode' else 'api-check', cfg)
        store.environment({'renderer': 'test', 'task_name': cfg.expected_task_name})
        store.event(attempt())
        scored = phase == 'episode' and status in ('success', 'failure', 'invalid_output')
        store.finish({'status': status, 'scored': scored, 'task_success': status == 'success' if scored else None,
                      'control_steps': 8 if phase == 'episode' else 0, 'policy_decisions': 1,
                      'task_execution_wall_seconds': 105. if phase == 'episode' else None,
                      'command_wall_seconds': 110.})
        return store

    def test_offline_cli_plan_preserves_eight_step_prompt_and_five_distinct_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'plan'
            with patch('ds_libero.client.Client', side_effect=AssertionError('No paid API')):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(main(['plan-five', '--config', PROFILE, '--out', str(root)]), 0)
            _, plan = load_plan(root)
            baseline = Config.load('configs/direct_v2_geometry_high.json')
            self.assertEqual(len(plan['episodes']), 5)
            self.assertEqual(plan['max_http_requests_per_episode_including_api_check'], 1027)
            adapter, raw, _, _ = fixture()
            obs = adapter._observe(raw)
            for e, selected in zip(plan['episodes'], FIVE_TASKS):
                cfg = Config.from_dict(e['config'])
                self.assertEqual((cfg.suite, cfg.task_id, cfg.expected_task_name), selected)
                self.assertEqual((cfg.max_steps, cfg.action_horizon, cfg.init_state_id, cfg.seed), (1024, 8, 0, 0))
                self.assertEqual(system_prompt(cfg), system_prompt(baseline))
                payload = request_payload(cfg, 'task', obs, [], 1021)
                self.assertEqual(json.loads(payload['messages'][1]['content'][0]['text'])['max_actions_this_response'], 3)
                self.assertEqual(payload['reasoning_effort'], 'high')
                self.assertNotIn('max_tokens', payload)
            with self.assertRaises(FileExistsError): self.make_plan(root)

    def test_control_limit_and_simulator_horizon_are_not_decision_counts(self):
        cfg = Config.load(PROFILE)
        result = run_quiet(FakeEnv(), FixedPolicy([ACTION]*8), MemoryStore(), cfg)
        self.assertEqual((result['control_steps'], result['policy_decisions']), (1024, 128))
        self.assertEqual(result['simulated_control_seconds'], 51.2)
        self.assertEqual(result['reason'], 'control_step_limit')
        adapter = LiberoEnvironment.__new__(LiberoEnvironment); adapter.cfg = cfg
        kwargs = adapter._create_simulator(lambda **kwargs: kwargs, 'unused.bddl')
        self.assertEqual(kwargs['horizon'], 1035)

    def test_names_and_standard_benchmark_horizons_are_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = self.make_plan(Path(tmp)/'plan')
            cfg = Config.from_dict(plan['episodes'][0]['config'])
            verify_task_name(cfg, FIVE_TASKS[0][2])
            with self.assertRaises(ExperimentError): verify_task_name(cfg, 'wrong_task')
            with self.assertRaises(ExperimentError):
                Config.from_dict({**cfg.to_dict(), 'evaluation_scope': 'benchmark'})
            with self.assertRaises(ExperimentError):
                Config.from_dict({**cfg.to_dict(), 'evaluation_scope': 'diagnostic', 'max_steps': 0})

    def test_mixed_suite_task_ids_never_pool_cost_or_latency(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'plan'; plan = self.make_plan(root)
            for e, status in zip(plan['episodes'], ('success', 'failure', 'invalid_output', 'api_error')):
                self.save(root, e, status)
                self.save(root, e, 'pass', 'api-check')
            report = report_plan(root)
            self.assertEqual((report['planned'], report['scored'], report['successes']), (5, 3, 1))
            self.assertEqual(len(report['per_task']), 5)
            for key in ('libero_spatial:0', 'libero_object:0', 'libero_goal:0'):
                s = report['per_task'][key]
                self.assertEqual(s['reported_usage']['total_tokens'], 17611)
                self.assertEqual(s['campaign_reported_usage']['total_tokens'], 35222)
                self.assertEqual(s['api_wait_seconds'], 100.)
                self.assertEqual(s['task_execution_wall_seconds']['total'], 105.)
                self.assertGreater(s['campaign_api_cost']['total_cost_range_cny'][0], s['api_cost']['total_cost_range_cny'][0])
            self.assertEqual(report['per_task']['libero_10:5']['status_counts'], {'not_started': 1})
            self.assertIsNone(report['macro_task_success_rate'])
            self.assertIn('policy_decisions', (root/'episodes.csv').read_text())

    def test_batch_continues_model_failures_skips_recorded_trials_and_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'plan'; plan = self.make_plan(root); calls=[]
            def run(directory, index):
                calls.append(index)
                status = ('failure', 'invalid_output', 'invalid_output', 'budget_exhausted', 'success')[index]
                self.save(root, plan['episodes'][index], status, 'api-check' if index == 2 else 'episode')
                return 2 if index in (2, 3) else 0
            with patch('ds_libero.benchmark.run_plan_episode', side_effect=run), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run_plan_all(root), 0)
                self.assertEqual(run_plan_all(root), 0)
            self.assertEqual(calls, [0,1,2,3,4])
            self.assertTrue((root/'aggregate.json').is_file())

    def test_batch_stops_on_infrastructure_failure_and_preserves_partial_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'plan'; plan=self.make_plan(root)
            def run(directory, index):
                self.save(root, plan['episodes'][index], 'api_error'); return 2
            with patch('ds_libero.benchmark.run_plan_episode', side_effect=run) as call:
                self.assertEqual(run_plan_all(root), 2)
                self.assertEqual(call.call_count, 1)
                with self.assertRaises(ExperimentError): run_plan_all(root)
            self.assertEqual(json.loads((root/'aggregate.json').read_text())['status_counts']['not_started'], 4)

    def test_rehashed_wrong_selection_or_policy_drift_is_rejected(self):
        for field, value in [('expected_task_name', 'wrong'), ('max_steps', 520), ('action_horizon', 16)]:
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)/'plan'; plan=self.make_plan(root)
                e=plan['episodes'][1]; e['config'][field]=value; e['config_sha256']=digest(e['config'])
                write_json(root/e['config_file'], e['config'])
                plan['plan_sha256']=digest({k:v for k,v in plan.items() if k!='plan_sha256'})
                write_json(root/'plan.json',plan)
                with self.assertRaises(ExperimentError): load_plan(root)


if __name__ == '__main__':
    unittest.main()

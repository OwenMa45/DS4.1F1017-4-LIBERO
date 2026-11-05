import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ds_libero.__main__ import main
from ds_libero.artifacts import write_json
from ds_libero.config import Config, digest
from ds_libero.errors import APIError, ExperimentError, InvalidAction
from ds_libero.microbench import MicroBench, PolicyInput, TASKS, external_config
from test_experiment import FakeEnv


ACTION = [0., 0., 0., 0., 0., 0., -1.]


class StubEnvironment(FakeEnv):
    def __init__(self, cfg, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.task = cfg.expected_task_name.replace('_', ' ')
        self.provenance = {"renderer": "OFFLINE_TEST_DOUBLE", "task_name": cfg.expected_task_name,
                           "task": self.task}
        self.closed = False

    def close(self):
        self.closed = True


class HoldPolicy:
    def __init__(self):
        self.requests = []
        self.closed = False

    def act(self, request):
        self.requests.append(copy.deepcopy(request))
        return [ACTION[:] for _ in range(request.max_actions)]

    def close(self):
        self.closed = True


def create_hold(context):
    return HoldPolicy()


class MicroBenchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.temp.name) / 'bench'
        self.bench = MicroBench(max_steps=5, action_horizon=3)
        self.envs = []
        self.env_options = {}

        def environment(cfg):
            env = StubEnvironment(cfg, **self.env_options)
            self.envs.append(env)
            return env

        self.environment = self.stack.enter_context(patch('ds_libero.environment.LiberoEnvironment', side_effect=environment))
        self.calibration = self.stack.enter_context(patch('ds_libero.calibration.calibrate',
            return_value={"status": "pass", "scored": False, "task_success": None}))
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.stack.enter_context(contextlib.redirect_stderr(io.StringIO()))

    def evaluate(self, factory=create_hold, **kwargs):
        return self.bench.evaluate(factory, out=self.root, policy_id='test-policy', revision='fixture-v1',
                                   replay=False, **kwargs)

    def test_offline_plan_freezes_selection_and_does_not_import_policy(self):
        with patch('ds_libero.microbench.importlib.import_module', side_effect=AssertionError('offline')):
            self.assertEqual(main(['microbench', 'plan', '--out', str(self.root),
                '--policy', 'missing_package:factory', '--policy-id', 'experiment', '--revision', 'v1']), 0)
        self.environment.assert_not_called()
        plan = json.loads((self.root / 'plan.json').read_text(encoding='utf-8'))
        self.assertEqual(len(plan['episodes']), 5)
        self.assertEqual([t.key for t in TASKS], ['libero_spatial:0', 'libero_object:0',
                         'libero_goal:0', 'libero_goal:5', 'libero_10:5'])
        for task, entry in zip(TASKS, plan['episodes']):
            cfg = Config.from_dict(entry['config'])
            self.assertEqual((cfg.suite, cfg.task_id, cfg.expected_task_name),
                             (task.suite, task.task_id, task.task_name))
            self.assertEqual((cfg.init_state_id, cfg.seed, cfg.max_steps, cfg.action_horizon), (0, 0, 220, 8))
        self.assertEqual(main(['microbench', 'report', str(self.root)]), 0)
        result = MicroBench.report(self.root)
        self.assertIsNone(result['score'])
        self.assertEqual(result['coverage'], 0.)
        self.assertEqual(result['planned_success_bounds'], [0., 1.])

    def test_public_policy_contract_budgets_fresh_instances_and_artifacts(self):
        policies, contexts = [], []

        def factory(context):
            contexts.append(context)
            policy = HoldPolicy()
            policies.append(policy)
            return policy

        result = self.evaluate(factory, policy_config={'checkpoint': 'fixture'})
        self.assertTrue(result['complete'])
        self.assertEqual((result['planned'], result['scored'], result['score']), (5, 5, 0.))
        self.assertEqual(result['batch_exit_code'], 0)
        self.assertEqual([c.task for c in contexts], list(TASKS))
        self.assertEqual(len({id(p) for p in policies}), 5)
        for context, policy in zip(contexts, policies):
            self.assertEqual(context.policy_config, {'checkpoint': 'fixture'})
            self.assertTrue(policy.closed)
            first, last = policy.requests
            self.assertIsInstance(first, PolicyInput)
            self.assertEqual((first.step, first.max_actions, first.remaining_steps), (0, 3, 5))
            self.assertEqual((last.step, last.max_actions, last.remaining_steps), (3, 2, 2))
            self.assertEqual(len(last.history), 3)
            self.assertEqual(len(last.visual_history), 1)
            self.assertEqual(last.control_feedback['previous_decision']['executed_steps'], 3)
            self.assertEqual(set(first.observation.images), {'agentview', 'robot0_eye_in_hand'})
            self.assertFalse(hasattr(first, 'env'))
        self.assertTrue(all(e.closed for e in self.envs))
        self.assertTrue((self.root / 'scores.csv').is_file())
        self.assertEqual(result['inference_accounting'], 'not_instrumented')
        for row in result['episodes']:
            self.assertEqual(row['control_steps'], 5)
            self.assertEqual(row['policy_decisions'], 2)
            self.assertEqual(row['decision_latency_seconds']['count'], 2)
            self.assertIsNone(row['usage'])
            self.assertIsNone(row['api_cost'])
            self.assertIsNone(row['http_requests'])

    def test_success_stops_mid_chunk_and_uses_official_checker(self):
        self.env_options = {'success_at': 1}
        result = self.evaluate()
        self.assertEqual(result['score'], 1.)
        self.assertEqual([r['control_steps'] for r in result['episodes']], [1] * 5)

    def test_invalid_action_is_scored_without_executing_any_prefix(self):
        invalid = [[], [ACTION, [float('nan')] + ACTION[1:]],
                   [[True] + ACTION[1:]], [ACTION] * 4, [[0.] * 6], ACTION]
        for i, output in enumerate(invalid):
            self.root = Path(self.temp.name) / f'invalid-{i}'

            class InvalidPolicy:
                def act(self, request):
                    return output

            result = self.evaluate(lambda context: InvalidPolicy())
            self.assertEqual(result['status_counts'], {'invalid_output': 5})
            self.assertEqual(result['score'], 0.)
            self.assertTrue(all(row['control_steps'] == 0 for row in result['episodes']))
            self.assertTrue(all(row['action_validity_rate'] == 0 for row in result['episodes']))

    def test_policy_cannot_mutate_runner_observations_or_history(self):
        class MutatingPolicy:
            def act(self, request):
                request.observation.state['eef_position_world_m'][0] = 999.
                request.observation.images.clear()
                request.history.clear()
                return [ACTION[:]]

        self.evaluate(lambda context: MutatingPolicy())
        path = self.root / 'runs/episode-0000/episode/events.jsonl'
        events = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        steps = [e for e in events if e['event'] == 'step']
        self.assertEqual([e['before']['eef_position_world_m'][0] for e in steps],
                         [i * .001 for i in range(5)])

    def test_feedback_boundary_discards_suffix_and_reports_executed_prefix(self):
        policies = []

        class PhasePolicy(HoldPolicy):
            def act(self, request):
                self.requests.append(request)
                return ([ACTION[:], [.1, 0, 0, 0, 0, 0, -1.]]
                        if request.max_actions >= 2 else [ACTION[:]])

        def factory(context):
            policy = PhasePolicy()
            policies.append(policy)
            return policy

        result = self.evaluate(factory)
        feedback = policies[0].requests[1].control_feedback['previous_decision']
        self.assertEqual((feedback['executed_steps'], feedback['discarded_actions']), (1, 1))
        self.assertEqual(result['episodes'][0]['policy_decisions'], 5)

    def test_inference_failure_is_unscored_stops_batch_and_closes_resources(self):
        policy = HoldPolicy()
        policy.act = lambda request: (_ for _ in ()).throw(APIError('offline fixture failure'))
        result = self.evaluate(lambda context: policy)
        self.assertEqual(result['status_counts'], {'api_error': 1, 'not_started': 4})
        self.assertIsNone(result['score'])
        self.assertEqual(result['batch_exit_code'], 2)
        self.assertTrue(policy.closed)
        self.assertTrue(all(e.closed for e in self.envs))
        with self.assertRaises(ExperimentError):
            MicroBench.run(self.root, policy_factory=lambda context: policy, replay=False)

    def test_done_without_success_and_initial_success_are_unscored(self):
        for index, options in enumerate(({'done_at': 1}, {'initial_success': True})):
            self.root = Path(self.temp.name) / f'unscored-{index}'
            self.env_options = options
            result = self.evaluate()
            self.assertIsNone(result['score'])
            self.assertEqual(result['scored'], 0)
            self.assertEqual(result['episodes'][0]['success'], None)
            self.assertEqual(result['batch_exit_code'], 2)

    def test_calibration_failure_prevents_factory_and_reports_preflight(self):
        self.calibration.return_value = {'status': 'calibration_failed', 'scored': False, 'task_success': None}
        factory = lambda context: self.fail('factory should not be called')
        result = self.evaluate(factory)
        self.assertEqual(result['scored'], 0)
        self.assertEqual(result['episodes'][0]['preflight_status']['status'], 'calibration_failed')
        self.assertEqual(result['batch_exit_code'], 2)

    def test_invalid_api_preflight_stops_batch_and_preserves_reason_and_paid_calls(self):
        from ds_libero.client import Client
        from test_experiment import reply
        bench = MicroBench(Config.from_dict({"protocol_version": "ds-libero-direct-v2",
                                             "max_steps": 5, "action_horizon": 3}))
        bench.prepare(self.root)
        with patch('ds_libero.client.resolve_api_key', return_value='fixture'), patch(
                'ds_libero.client.Client', side_effect=lambda cfg, store: Client(
                    cfg, store, api_key='fixture', send=lambda *args: reply('   '))):
            result = bench.run(self.root, replay=False)
        self.assertEqual(result['status_counts'], {'preflight_incomplete': 1, 'not_started': 4})
        self.assertEqual(result['batch_exit_code'], 2)
        self.assertEqual(result['scored'], 0)
        first = result['episodes'][0]
        self.assertEqual(first['preflight_status'], {'phase': 'api-check', 'status': 'invalid_output'})
        self.assertIn('api-check: model returned empty/non-text content', first['reason'])
        self.assertEqual(first['http_requests'], 0)
        self.assertEqual(first['campaign_http_requests'], 1)
        self.assertEqual(result['campaign_http_requests'], 1)
        self.assertFalse((self.root / 'runs/episode-0000/episode').exists())
        self.assertFalse((self.root / 'runs/episode-0001').exists())
        with self.assertRaisesRegex(ExperimentError, 'incomplete/error artifacts'):
            bench.run(self.root, replay=False)

    def test_recorded_trials_are_not_rerun_and_new_evaluate_cannot_overwrite(self):
        self.evaluate()
        calls = self.environment.call_count
        with patch('ds_libero.microbench._import_factory', side_effect=AssertionError('No policy reload')):
            resumed = MicroBench.run(self.root, replay=False)
        self.assertEqual(self.environment.call_count, calls)
        self.assertEqual(resumed['score'], 0.)
        with self.assertRaises(FileExistsError):
            self.evaluate()

    def test_selected_episode_preserves_all_five_in_denominator(self):
        self.bench.prepare(self.root, policy_id='test', revision='v1', adapter='test_microbench:create_hold')
        self.env_options = {'success_at': 1}
        result = MicroBench.run(self.root, policy_factory=create_hold, episode=2, replay=False)
        self.assertEqual((result['planned'], result['scored'], result['successes']), (5, 1, 1))
        self.assertIsNone(result['score'])
        self.assertEqual(result['coverage'], .2)
        self.assertEqual(result['planned_success_bounds'], [.2, 1.])
        for bad in (-1, 5, True):
            with self.assertRaises(ExperimentError):
                MicroBench.run(self.root, policy_factory=create_hold, episode=bad)

    def test_protocol_hash_excludes_model_but_distinguishes_horizons(self):
        base = MicroBench().spec
        builtin = MicroBench('configs/gpt61_sol_medium_220.json').spec
        self.assertEqual(digest(base), digest(builtin))
        self.assertNotEqual(digest(base), digest(MicroBench(max_steps=1024).spec))
        self.assertEqual(external_config().to_dict(), Config.load('configs/microbench_220.json').to_dict())

    def test_code_factory_and_metadata_drift_block_run(self):
        self.bench.prepare(self.root, policy_id='test', revision='v1', adapter='test_microbench:create_hold')
        MicroBench.run(self.root, policy_factory=create_hold, episode=0, replay=False)
        with patch('ds_libero.microbench.source_fingerprint', return_value='changed'):
            with self.assertRaises(ExperimentError):
                MicroBench.run(self.root, policy_factory=create_hold)
        with patch('ds_libero.microbench._factory_source', return_value='changed'):
            with self.assertRaises(ExperimentError):
                MicroBench.run(self.root, policy_factory=create_hold)
        plan = json.loads((self.root / 'plan.json').read_text(encoding='utf-8'))
        plan['microbench']['spec']['environment']['max_steps'] = 123
        plan['plan_sha256'] = digest({k: v for k, v in plan.items() if k != 'plan_sha256'})
        write_json(self.root / 'plan.json', plan)
        with self.assertRaises(ExperimentError):
            MicroBench.report(self.root)

    def test_factory_setup_failure_is_unscored_and_interruption_is_not_resumed(self):
        def bad_factory(context):
            raise InvalidAction('failure while initializing policy')

        result = self.evaluate(bad_factory)
        self.assertEqual(result['episodes'][0]['status'], 'runtime_error')
        self.assertEqual(result['scored'], 0)
        self.root = Path(self.temp.name) / 'interrupt'
        policy = HoldPolicy()
        policy.act = lambda request: (_ for _ in ()).throw(KeyboardInterrupt())

        def interrupted_factory(context):
            return policy

        result = self.evaluate(interrupted_factory)
        self.assertEqual(result['episodes'][0]['status'], 'interrupted')
        self.assertEqual(result['batch_exit_code'], 2)
        self.assertTrue(policy.closed)
        with self.assertRaisesRegex(ExperimentError, 'incomplete/error artifacts'):
            MicroBench.run(self.root, policy_factory=interrupted_factory, replay=False)

    def test_mixed_policy_artifacts_and_legacy_runner_are_rejected(self):
        from ds_libero.benchmark import run_plan_all, run_plan_episode
        self.evaluate()
        with self.assertRaisesRegex(ExperimentError, 'microbench run'):
            run_plan_all(self.root)
        with self.assertRaisesRegex(ExperimentError, 'microbench run'):
            run_plan_episode(self.root, 0)
        path = self.root / 'runs/episode-0000/episode/manifest.json'
        manifest = json.loads(path.read_text(encoding='utf-8'))
        manifest['microbench']['policy']['id'] = 'different-policy'
        write_json(path, manifest)
        with self.assertRaisesRegex(ExperimentError, 'frozen microbench policy'):
            MicroBench.report(self.root)

    def test_external_cli_runs_importable_factory_without_credentials_or_client(self):
        self.env_options = {'success_at': 1}
        self.assertEqual(main(['microbench', 'plan', '--out', str(self.root),
            '--policy', 'examples.noop_policy:create_policy', '--policy-id', 'noop', '--revision', 'v1']), 0)
        with patch('ds_libero.client.Client', side_effect=AssertionError('No HTTP client')), \
             patch('ds_libero.client.resolve_api_key', side_effect=AssertionError('No credential lookup')):
            self.assertEqual(main(['microbench', 'run', str(self.root), '--no-replay']), 0)
        self.assertEqual(MicroBench.report(self.root)['score'], 1.)

    def test_builtin_profile_uses_existing_client_and_all_three_phases(self):
        from test_experiment import FixedPolicy
        bench = MicroBench('configs/gpt61_sol_medium_220.json')
        self.env_options = {'success_at': 1}
        with patch('ds_libero.client.resolve_api_key', return_value='test-only'), \
             patch('ds_libero.client.Client') as client, \
             patch('ds_libero.client.ModelPolicy', side_effect=lambda *args: FixedPolicy([ACTION])):
            result = bench.evaluate(out=self.root, replay=False)
        self.assertEqual(result['score'], 1.)
        self.assertEqual(client.call_count, 10)
        self.assertEqual(self.environment.call_count, 15)
        for task in TASKS:
            campaign = self.root / f'runs/episode-{task.index:04d}'
            self.assertTrue((campaign / 'api-check/result.json').is_file())
        self.assertEqual(result['inference_accounting'], 'built_in_client')

    def test_builtin_preflight_invalid_output_never_becomes_task_score(self):
        class InvalidPolicy:
            def act(self, *args):
                raise InvalidAction('bad output')

        bench = MicroBench('configs/gpt61_sol_medium_220.json')
        with patch('ds_libero.client.resolve_api_key', return_value='test-only'), \
             patch('ds_libero.client.Client'), \
             patch('ds_libero.client.ModelPolicy', return_value=InvalidPolicy()):
            result = bench.evaluate(out=self.root, replay=False)
        self.assertEqual(result['scored'], 0)
        self.assertIsNone(result['score'])
        self.assertEqual(result['status_counts'], {'preflight_incomplete': 1, 'not_started': 4})
        self.assertEqual(result['batch_exit_code'], 2)

    def test_invalid_configuration_and_missing_identity_fail_before_outputs(self):
        for budget in (0, 2049, True):
            with self.assertRaises(ExperimentError):
                MicroBench(max_steps=budget)
        with self.assertRaises(ExperimentError):
            self.bench.prepare(self.root, policy_id='test', adapter='examples.noop_policy:create_policy')
        self.assertFalse(self.root.exists())
        with self.assertRaises(ExperimentError):
            MicroBench('configs/first_episode.json')
        with self.assertRaises(ExperimentError):
            Config.from_dict({**external_config().to_dict(), 'endpoint': 'https://invalid.example'})


if __name__ == '__main__':
    unittest.main()

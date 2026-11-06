import csv
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from ds_libero.artifacts import Store
from ds_libero.client import Client, ModelPolicy
from ds_libero.microbench import MicroBench
from ds_libero.report import build_report
from test_experiment import FakeEnv, MemoryStore, observation, reply, run_quiet
from test_microbench import StubEnvironment
from test_sam3 import FakeTool, config, tool_reply


ZERO = [0.0] * 7
MOVE = [.3, 0, -.1, 0, .2, 0, 1]


def actions(rows):
    return reply(json.dumps({'actions': rows}))


def input_observation(payload):
    return json.loads(payload['messages'][1]['content'][0]['text'])


class DimensionGuardTests(unittest.TestCase):
    def rollout(self, cfg, replies, *, store=None, success_at=None):
        store = store or MemoryStore()
        env, sent = FakeEnv(success_at=success_at), []
        tool = FakeTool(cfg.sam3)
        sequence = iter(replies)

        def send(endpoint, body, key, timeout):
            sent.append((env.steps, json.loads(body)))
            return next(sequence)

        client = Client(cfg, store, api_key='fixture', send=send)
        with patch('ds_libero.sam3_tools.Sam3Tool', return_value=tool), patch('test_experiment.observation',
                side_effect=lambda step=0: replace(observation(step), geometry={'cameras': {}})):
            policy = ModelPolicy(cfg, 'task', client, store)
            result = run_quiet(env, policy, store, cfg)
        return env, result, sent, store

    def test_real_failure_shapes_replace_whole_chunk_and_inform_next_decision_once(self):
        drawer = [[0, .5, 0, 0, 0, 0], [0, .25, .45, 0, 0, 0],
                  [0, 0, .45, -.05, .39, .11], [0, 0, .3, -.025, .19, .055]]
        book = [[.16, -.16, .03, 0, 0, -.6]] * 4
        cases = ((0, [MOVE, [0, .4, .5]]), (1, book), (2, drawer))
        for initial_tool_rounds, bad in cases:
            with self.subTest(tool_rounds=initial_tool_rounds), tempfile.TemporaryDirectory() as temp:
                sam = {'max_tool_rounds': 2} if initial_tool_rounds else {}
                cfg = config(max_steps=8, action_horizon=8, retry_incomplete_actions=True,
                             zero_on_dimension_error=True, sam3=sam, grasp_feedback=True, camera_geometry=True)
                turns = [tool_reply([(str(i), 'agentview', 'bowl')]) for i in range(initial_tool_rounds)]
                next_turns = [tool_reply([('a', 'agentview', 'book')]),
                              tool_reply([('b', 'agentview', 'bowl')])] if sam else []
                store = Store(Path(temp)/'episode', 'run', cfg)
                env, result, sent, _ = self.rollout(cfg, [*turns, actions(bad), *next_turns, actions([MOVE]),
                                                     *next_turns, actions([ZERO])], store=store, success_at=3)
                self.assertEqual(result['status'], 'success')
                self.assertEqual(env.actions, [ZERO, MOVE, ZERO])  # The last zero is a valid model output.
                self.assertEqual(sum(step == 0 for step, _ in sent), initial_tool_rounds + 1)
                for step, payload in sent:
                    obs = input_observation(payload)
                    self.assertEqual(obs['control_step'], step)
                    if step == 1:
                        feedback = obs['previous_action_dimension_error']
                        self.assertEqual(feedback['row_widths'], [len(row) for row in bad])
                        self.assertEqual((feedback['decision_step'], feedback['expected_width']), (0, 7))
                        self.assertEqual(feedback['replacement_action'], ZERO)
                        self.assertTrue(feedback['replacement_executed'])
                        self.assertEqual(obs['recent_history'][-1]['action'], ZERO)
                        self.assertEqual(obs['control_feedback']['previous_decision']['executed_actions'], [ZERO])
                    else:
                        self.assertNotIn('previous_action_dimension_error', obs)
                store.finish(result)
                summary = build_report(store.path, replay=False)
                self.assertEqual((summary['dimension_fallback_decisions'], summary['dimension_fallback_steps']), (1, 1))
                self.assertEqual((summary['valid_model_outputs'], summary['model_outputs']), (2, 3))
                self.assertEqual(summary['action_output_recovery']['retries_requested'], 0)

    def test_consecutive_bad_dimensions_consume_one_tick_each_up_to_step_limit(self):
        cfg = config(max_steps=3, action_horizon=3, zero_on_dimension_error=True, sam3={})
        env, result, sent, _ = self.rollout(cfg, [actions([[0]*6]), actions([[0]*8]), actions([])])
        self.assertEqual((result['status'], result['reason']), ('failure', 'control_step_limit'))
        self.assertEqual(env.actions, [ZERO, ZERO, ZERO])
        self.assertEqual([step for step, _ in sent], [0, 1, 2])
        for step, previous_width in ((1, 6), (2, 8)):
            feedback = input_observation(sent[step][1])['previous_action_dimension_error']
            self.assertEqual((feedback['decision_step'], feedback['row_widths']), (step-1, [previous_width]))

    def test_rank_row_budget_and_nested_arrays_are_shapes_but_bad_values_stay_rejected(self):
        cfg = config(max_steps=1, action_horizon=1, zero_on_dimension_error=True, sam3={})
        for bad in ([], 3, [[0]*7, [0]*7], [3], [[0, 0, [0], 0, 0, 0, 0]], [[{}, 0, 0, 0, 0, 0, 0]]):
            with self.subTest(shape=bad):
                env, _, sent, store = self.rollout(cfg, [actions(bad)])
                self.assertEqual(env.actions, [ZERO])
                self.assertEqual(len(sent), 1)
                decision, = [e for e in store.events if e['event']=='decision']
                self.assertFalse(decision['valid'])
                self.assertEqual(decision['action_source'], 'dimension_guard')
        for bad in (actions([[2, 0, 0, 0, 0, 0, 0]]), actions([[0, 0, 0, 0, 0, 0, .5]]),
                    actions([[True, 0, 0, 0, 0, 0, 0]]), actions([[float('nan'), 0, 0, 0, 0, 0, 0]]),
                    reply('{"actions":[],"actions":[]}'), reply('{"actions":[', finish='length')):
            env, result, _, _ = self.rollout(cfg, [bad])
            self.assertEqual((env.actions, result['status']), ([], 'invalid_output'))

    def test_preflight_proposal_is_labeled_and_handed_to_first_scored_request(self):
        cfg = config(max_steps=2, action_horizon=2, zero_on_dimension_error=True, sam3={})
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'five'
            bench = MicroBench(cfg)
            plan = bench.prepare(root)
            self.assertTrue(plan['microbench']['spec']['environment']['zero_on_dimension_error'])
            sent, envs = [], []
            replies = iter([actions([[0]*6]), actions([[0]*8]), actions([MOVE])])
            def environment(c):
                envs.append(StubEnvironment(c, success_at=2))
                return envs[-1]
            def send(endpoint, body, key, timeout):
                sent.append(input_observation(json.loads(body)))
                return next(replies)
            with patch('ds_libero.environment.LiberoEnvironment', side_effect=environment), \
                 patch('ds_libero.calibration.calibrate', return_value={'status':'pass','scored':False,'task_success':None}), \
                 patch('ds_libero.client.resolve_api_key', return_value='fixture'), \
                 patch('ds_libero.client.Client', side_effect=lambda c,s: Client(c,s,api_key='fixture',send=send)):
                report = bench.run(root, episode=0, replay=False)
            gate = json.loads((root/'runs/episode-0000/api-check/result.json').read_text())
            self.assertEqual((gate['status'], gate['model_output_valid'], gate['model_actions_executed']), ('pass', False, 0))
            self.assertEqual(gate['proposed_action'], [ZERO])
            self.assertEqual([env.actions for env in envs], [[], [], [ZERO, MOVE]])
            self.assertEqual(sent[1]['previous_action_dimension_error'], gate['action_dimension_feedback'])
            self.assertFalse(sent[1]['previous_action_dimension_error']['replacement_executed'])
            self.assertEqual(sent[2]['previous_action_dimension_error']['source_phase'], 'episode')
            self.assertTrue(sent[2]['previous_action_dimension_error']['replacement_executed'])
            self.assertEqual(report['episodes'][0]['action_validity_rate'], .5)
            self.assertEqual(report['episodes'][0]['dimension_fallback_steps'], 1)
            with (root/'scores.csv').open(encoding='utf-8',newline='') as f:
                self.assertEqual(next(csv.DictReader(f))['dimension_fallback_steps'], '1')


if __name__ == '__main__':
    unittest.main()

import copy
import contextlib
import io
import json
import math
import tempfile
import unittest
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
from PIL import Image

from ds_libero.artifacts import Store
from ds_libero.client import Client, HTTPResponse, ModelPolicy
from ds_libero.calibration import calibrate
from ds_libero.config import Config
from ds_libero.environment import LiberoEnvironment
from ds_libero.errors import ExperimentError, SimulationError
from ds_libero.geometry_contract import check_image_convention, quaternion_matrix, matrix_quaternion
from ds_libero.geometry_overlay import draw_geometry_overlay
from ds_libero.protocol import CAMERAS, request_payload, system_prompt
from ds_libero.report import build_report, build_campaign_report
from test_accounting import usage
from test_experiment import FakeEnv, run_quiet


def fixture(hand_q=(0., 0., 0., 1.), joints=(.04, -.04), verified=True):
    """Independent forward kinematics from pinned XML, including body/site offsets."""
    def rot(wxyz):
        q = np.array(wxyz, dtype=float); q /= np.linalg.norm(q)
        w, v = q[0], q[1:]
        x, y, z = v
        skew = np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
        return (w*w-v@v)*np.eye(3) + 2*np.outer(v,v) + 2*w*skew
    def pose(element):
        t = np.eye(4)
        t[:3,:3] = rot([float(x) for x in element.get('quat', '1 0 0 0').split()])
        t[:3,3] = [float(x) for x in element.get('pos', '0 0 0').split()]
        return t
    frames = {}
    joint_values = dict(zip(('finger_joint1', 'finger_joint2'), joints))
    def visit(element, parent):
        t = parent @ pose(element)
        joint = element.find('joint')
        if joint is not None:
            t[:3,3] += t[:3,:3] @ np.fromstring(joint.get('axis'), sep=' ') * joint_values[joint.get('name')]
        frames[element.get('name')] = t
        for child in element:
            if child.tag != 'joint':
                visit(child, t)
    hand = np.eye(4)
    hand[:3,:3] = rot([hand_q[3], *hand_q[:3]])
    hand[:3,3] = [.3, -.1, .5]
    visit(ET.parse(Path(__file__).parent/'fixtures/panda_frames.xml').getroot(), hand)
    site = frames['gripper0_grip_site']
    cam = frames['robot0_eye_in_hand']
    pad_names = ['gripper0_finger1_pad_collision', 'gripper0_finger2_pad_collision']
    names = []
    def geom_id(name):
        names.append(name)
        return pad_names.index(name)  # Fails if any scene geom is requested.
    sim = NS(model=NS(site_name2id=lambda n: ['gripper0_grip_site'].index(n),
                      body_name2id=lambda n: ['robot0_right_hand'].index(n),
                      geom_name2id=geom_id, camera_name2id=lambda n: list(CAMERAS).index(n),
                      cam_fovy=[45., 75.]),
             data=NS(site_xpos=[site[:3,3]], site_xmat=[site[:3,:3].ravel()],
                     body_xpos=[hand[:3,3]], body_xmat=[hand[:3,:3].ravel()],
                     geom_xpos=np.array([frames[n][:3,3] for n in pad_names]),
                     cam_xpos=[[1.,0.,1.5], cam[:3,3]], cam_xmat=[np.eye(3).ravel(), cam[:3,:3].ravel()], qpos=[0.], time=1.))
    sim.forward = lambda: None
    gripper = NS(important_sites={'grip_site':'gripper0_grip_site'},
                 important_geoms={'left_fingerpad':[pad_names[0]], 'right_fingerpad':[pad_names[1]]})
    robot = NS(gripper=gripper, controller=NS(eef_name='gripper0_grip_site'),
               robot_model=NS(eef_name='robot0_right_hand'), eef_site_id=0)
    adapter = LiberoEnvironment.__new__(LiberoEnvironment)
    adapter.cfg = Config.load('configs/direct_v2_geometry_high.json' if verified else 'configs/direct_v2_grasp_high.json')
    adapter.np, adapter.env = np, NS(sim=sim, robots=[robot])
    pixels = np.arange(256*256*3).reshape(256,256,3).astype('uint8')
    raw = {'robot0_eef_pos':site[:3,3].copy(), 'robot0_eef_quat':list(hand_q), 'robot0_gripper_qpos':list(joints),
           **{c+'_image':pixels.copy() for c in CAMERAS}}
    adapter.env.env = NS(_get_observations=lambda force_update: raw)
    return adapter, raw, frames, names


class GeometryContractTests(unittest.TestCase):
    def test_refresh_replaces_stale_cached_observations_without_stepping(self):
        env, raw, _, _ = fixture()
        calls = []
        env.env.sim.forward = lambda: calls.append('forward')
        def fresh(force_update):
            self.assertTrue(force_update)
            self.assertEqual(calls, ['forward'])
            calls.append('observations')
            return raw
        env.env.env._get_observations = fresh
        obs = env._observe({})  # A stale/empty cache is replaced, not mixed with live geometry.
        self.assertEqual(calls, ['forward','observations'])
        self.assertEqual(env.env.sim.data.time, 1.)
        self.assertIn('control_frame', obs.geometry)
        env.env.sim.forward = lambda: setattr(env.env.sim.data, 'time', 2.)
        env.env.env._get_observations = lambda force_update: raw
        with self.assertRaises(SimulationError): env._observe(raw)

    def test_xml_site_body_difference_and_pad_mapping_at_multiple_orientations(self):
        for q in ([0,0,0,1], [.5,.5,.5,.5], [1,0,0,0], [0,1,0,0], [0,0,1,0]):
            with self.subTest(q=q):
                env, raw, frames, reads = fixture(q, (.03, -.01))
                obs = env._observe(raw)
                site = frames['gripper0_grip_site']
                np.testing.assert_allclose(quaternion_matrix(obs.state['eef_quaternion_xyzw']), site[:3,:3], atol=1e-12)
                self.assertFalse(np.allclose(quaternion_matrix(q), site[:3,:3]))
                g = obs.geometry['robot_gripper_geometry']
                np.testing.assert_allclose(g['points_control_frame_m']['pad_midpoint'], [-.01,0,-.0036], atol=1e-12)
                np.testing.assert_allclose(g['jaw_axis_world_left_to_right'], site[:3,0], atol=1e-12)
                self.assertEqual(reads, ['gripper0_finger1_pad_collision', 'gripper0_finger2_pad_collision'])
                self.assertEqual(raw['robot0_eef_quat'], q)

    def test_wrist_camera_mount_and_projection_are_not_copied_from_droid(self):
        env, raw, _, _ = fixture([.5,.5,.5,.5])
        obs = env._observe(raw)
        c = obs.geometry['cameras']['robot0_eye_in_hand']
        np.testing.assert_allclose(c['position_control_frame_m'], [0,.05,-.097], atol=1e-12)
        np.testing.assert_allclose(c['R_control_from_camera_optical'], np.diag([-1,-1,1]), atol=1e-12)
        # Independent optical projection: pad midpoint is X=0, Y=.05, Z=.0934.
        focal = 128/math.tan(math.radians(75/2))
        uv = obs.geometry['robot_gripper_geometry']['pixels_uv']['robot0_eye_in_hand']['pad_midpoint']
        np.testing.assert_allclose(uv, [128, 128+focal*.05/.0934], atol=1e-10)

    def test_position_body_pose_pad_and_controller_mismatches_fail_before_policy(self):
        for kind in ('position', 'quaternion', 'pad', 'controller', 'site_id', 'camera'):
            with self.subTest(kind=kind):
                env, raw, _, _ = fixture()
                if kind == 'position': raw['robot0_eef_pos'][0] += .001
                if kind == 'quaternion': raw['robot0_eef_quat'] = [0,0,1,0]
                if kind == 'pad': env.env.sim.data.geom_xpos[0,2] += .001
                if kind == 'controller': env.env.robots[0].controller.eef_name = 'wrong_site'
                if kind == 'site_id': env.env.robots[0].eef_site_id = 9
                if kind == 'camera': env.env.sim.data.cam_xmat[0][0] = 2
                with self.assertRaises(SimulationError): env._observe(raw)

    def test_quaternion_sign_and_pi_rotations(self):
        for q in ([1,0,0,0], [0,1,0,0], [0,0,1,0], [.5,-.5,.5,-.5]):
            r = quaternion_matrix(q)
            np.testing.assert_allclose(quaternion_matrix(matrix_quaternion(r)), r, atol=1e-12)
        env, raw, _, _ = fixture([0,0,0,-1])
        env._observe(raw)  # Quaternion sign does not change a rotation.
        with self.assertRaises(ValueError): quaternion_matrix([0,0,0,0])

    def test_image_flip_is_once_and_legacy_state_remains_unchanged(self):
        check_image_convention('opengl')
        with self.assertRaises(ValueError): check_image_convention('opencv')
        for verified in (True, False):
            env, raw, _, _ = fixture(verified=verified)
            obs = env._observe(raw)
            np.testing.assert_array_equal(np.array(Image.open(io.BytesIO(obs.images['agentview']))), raw['agentview_image'][::-1])
            if not verified:
                self.assertEqual(obs.state['eef_quaternion_xyzw'], raw['robot0_eef_quat'])
                self.assertNotIn('control_frame', obs.geometry)

    def test_current_and_history_require_contract_and_profile_keeps_high_unlimited(self):
        env, raw, _, _ = fixture()
        obs = env._observe(raw)
        cfg = env.cfg
        payload = request_payload(cfg, 'task', obs, [], 3, [(0, obs)])
        self.assertEqual(payload['reasoning_effort'], 'high')
        self.assertNotIn('max_tokens', payload)
        self.assertIn('BOTH world position and orientation', system_prompt(cfg))
        self.assertEqual(Config.from_dict(cfg.to_dict()), cfg)
        with self.assertRaises(ExperimentError):
            Config.from_dict({**cfg.to_dict(), 'grasp_feedback':False})
        bad = replace(obs, geometry={})
        for current, history in ((bad, []), (obs, [(0, bad)])):
            with self.assertRaises(ExperimentError): request_payload(cfg, 'task', current, [], 3, history)

    def test_overlay_does_not_change_policy_image(self):
        env, raw, _, _ = fixture()
        obs = env._observe(raw)
        before = copy.deepcopy(obs)
        overlay = draw_geometry_overlay(obs, 'robot0_eye_in_hand')
        self.assertEqual(Image.open(io.BytesIO(overlay)).size, (512,546))
        self.assertEqual(obs, before)

    def test_calibration_exports_geometry_checks_and_overlays_even_if_motion_fails(self):
        adapter, raw, _, _ = fixture()
        obs = adapter._observe(raw)
        env = NS(reset=lambda: obs, step=lambda a: (obs, False), check_success=lambda: False,
                 contract={'geometry_schema': 'panda-grip-site-v1'})
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            store = Store(Path(temp)/'calibration', 'calibrate', adapter.cfg)
            with patch('ds_libero.client.Client', side_effect=AssertionError('No paid API during calibration')):
                result = calibrate(env, store)
            store.finish(result)
            summary = build_report(store.path, replay=False)
            self.assertEqual(summary['status'], 'calibration_failed')
            self.assertEqual(len(summary['geometry_contract_checks']), 26)
            self.assertEqual(len(list((store.path/'geometry_overlays').glob('*.png'))), 52)
            self.assertEqual(summary['http_requests'], 0)

    def test_new_profile_retains_task_and_campaign_api_cost_and_wall_latency(self):
        adapter, raw, _, _ = fixture()
        obs = adapter._observe(raw)
        class Clock:
            value = 0.
            def now(self): return self.value
            def advance(self, t): self.value += t
        clock = Clock()
        class Env(FakeEnv):
            def reset(self):
                super().reset(); clock.advance(5.)
                return obs
            def step(self, action):
                super().step(action); clock.advance(.25)
                return obs, False
        with tempfile.TemporaryDirectory() as temp:
            store = Store(Path(temp)/'episode', 'run', adapter.cfg)
            def send(endpoint, body, key, timeout):
                self.assertNotIn('max_tokens', json.loads(body))
                clock.advance(10.)
                return HTTPResponse(200, json.dumps({'model':'deepseek-flash', 'usage':usage(), 'choices':[
                    {'finish_reason':'stop','message':{'content':'{"actions":[[0,0,0,0,0,0,1]]}'}}]}).encode())
            with patch('time.perf_counter', side_effect=clock.now):
                policy = ModelPolicy(adapter.cfg, 'task', Client(adapter.cfg, store, api_key='dummy', send=send), store)
                result = run_quiet(Env(success_at=1), policy, store, adapter.cfg)
            store.finish(result)
            summary = build_report(store.path, replay=False)
            self.assertEqual(summary['task_execution_wall_seconds'], 10.25)
            self.assertEqual(summary['http_latency_seconds']['total'], 10.)
            self.assertEqual(summary['decision_latency_seconds']['total'], 10.)
            self.assertEqual(summary['reported_usage']['total_tokens'], 17611)
            self.assertGreater(summary['api_cost']['total_cost_range_cny'][0], 0)
            campaign = build_campaign_report(temp)
            self.assertEqual(campaign['reported_usage']['total_tokens'], 17611)
            self.assertEqual(campaign['api_wait_seconds'], 10.)


if __name__ == '__main__':
    unittest.main()

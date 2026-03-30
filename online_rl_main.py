"""
online_rl_main.py — 온라인 RL 제어 루프 + 평가 시스템
======================================================
3단계 리셋: SOFT_RECOVERY / WARM_RESET / HARD_RESET
평가: eval_interval마다 고정 시나리오 5개 순회 (learn OFF, greedy)
"""
import rclpy
import numpy as np
import math
import json
import collections
import os
import sys
import subprocess
import time as pytime

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import (
    OffboardControlMode, TrajectorySetpoint, VehicleCommand,
    SensorCombined, VehicleOdometry,
    VehicleThrustSetpoint, VehicleTorqueSetpoint, SensorGps
)
from nav_msgs.msg import Odometry as GroundTruthOdometry
from std_msgs.msg import String

import torch
from config import Config, sample_episode_scenario
from env.ukf_filter import DynamicsUKF, compute_nis_scaled, load_calibration, to_physical_u
from env.reward import calculate_reward
from rl.agent import OnlineSRRHUIFAgent


# ══════════════════════════════════════════════════════════════
#  Simulator Process Manager
# ══════════════════════════════════════════════════════════════
class SimProcessManager:
    def __init__(self, sim_script='run_sim.py', headless=True):
        self.sim_script = sim_script
        self.headless = headless
        self.process = None

    def start(self):
        cmd = [sys.executable, self.sim_script]
        if self.headless:
            cmd.append('--headless')
        self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"  [SimManager] Started PID={self.process.pid}")

    def stop(self):
        if self.process is None:
            return
        pid = self.process.pid
        try:
            self.process.terminate()
            self.process.wait(timeout=10)
            print(f"  [SimManager] Terminated PID={pid}")
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
            print(f"  [SimManager] Killed PID={pid}")
        self.process = None

    def restart(self):
        self.stop()
        pytime.sleep(5)
        self.start()


# ══════════════════════════════════════════════════════════════
#  Main Node
# ══════════════════════════════════════════════════════════════
class OnlineRLNode(Node):
    def __init__(self, cfg):
        super().__init__('online_rl_controller')
        self.cfg = cfg
        self.step_dt = 0.02

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST, depth=5)

        # ── Simulator ──
        self.sim_mgr = SimProcessManager('run_sim.py', cfg.headless)
        self.sim_mgr.start()
        pytime.sleep(10)

        # ── Publishers ──
        self.pub_offboard = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos)
        self.pub_traj = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos)
        self.pub_cmd = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', qos)
        self.pub_attack = self.create_publisher(String, '/attack_config', 10)
        self.pub_scenario = self.create_publisher(String, '/scenario_config', 10)
        self.pub_sim_ctrl = self.create_publisher(String, '/sim_control', 10)

        # ── Subscribers ──
        self.create_subscription(SensorGps, '/sim/sensor_gps', self._cb_gps, qos)
        self.create_subscription(SensorCombined, '/fmu/out/sensor_combined', self._cb_sensor, qos)
        self.create_subscription(VehicleOdometry, '/fmu/out/vehicle_odometry', self._cb_odometry, qos)
        self.create_subscription(VehicleThrustSetpoint, '/fmu/out/vehicle_thrust_setpoint', self._cb_thrust, qos)
        self.create_subscription(VehicleTorqueSetpoint, '/fmu/out/vehicle_torque_setpoint', self._cb_torque, qos)
        self.create_subscription(GroundTruthOdometry, '/gt/odometry', self._cb_gt, qos)

        # ── UKF + Agent ──
        self.calib = load_calibration('calibration.json')
        self.ukf = DynamicsUKF(dt=self.step_dt, calib=self.calib)
        self.agent = OnlineSRRHUIFAgent(cfg)
        self.window_buffer = collections.deque(maxlen=cfg.window_size)

        # ── Sensor state ──
        self.cur_accel = np.zeros(3); self.cur_gyro = np.zeros(3)
        self.cur_pos = np.zeros(3); self.cur_vel = np.zeros(3)
        self.cur_euler = np.zeros(3)
        self.cur_thrust = np.zeros(3); self.cur_torque = np.zeros(3)
        self.gt_pos = np.zeros(3); self.gt_vel = np.zeros(3)
        self.obs_gps_pos = np.zeros(3); self.obs_gps_vel = np.zeros(3)
        self.home_lat = None; self.home_lon = None; self.home_alt = None
        self.earth_radius = 6371000.0; self.gps_updated = False
        self.last_res = np.zeros(9); self.last_Pzz = np.eye(9)

        # ── Episode state ──
        self.flight_state = 'IDLE'
        self.episode = 0; self.scenario = None
        self.step_count = 0; self.tick_count = 0
        self.init_counter = 0; self.stable_counter = 0; self.theta = 0.0
        self.prev_state = None; self.prev_action = None
        self.episode_reward = 0.0; self.is_ukf_initialized = False
        self.attack_active_flag = False

        # ── Detection tracking (학습/평가 공통) ──
        self.first_hover_step = None
        self.hover_before_attack_count = 0

        # ── Evaluation mode ──
        self.eval_mode = False
        self.eval_scenario_idx = 0
        self.current_eval_results = []
        self.eval_history = []

        # ── Heartbeat ──
        self.last_gt_time = pytime.time()
        self.heartbeat_timeout = 8.0

        # ── Logging ──
        self.episode_losses = []
        self.train_start_time = pytime.time()
        self.hard_reset_count = 0

        self.timer = self.create_timer(self.step_dt, self._tick)
        self.get_logger().info(f'[INIT] dimS={cfg.dimS} | eval_interval={cfg.eval_interval} | max_ep={cfg.max_episodes}')

    # ══════════════════════════════════════════════════════════
    #  Sensor Callbacks
    # ══════════════════════════════════════════════════════════
    def _cb_gps(self, msg):
        if self.home_lat is None:
            self.home_lat = msg.latitude_deg; self.home_lon = msg.longitude_deg; self.home_alt = msg.altitude_msl_m
        lat_rad = math.radians(self.home_lat)
        self.obs_gps_pos[:] = [
            math.radians(msg.longitude_deg - self.home_lon) * self.earth_radius * math.cos(lat_rad),
            math.radians(msg.latitude_deg - self.home_lat) * self.earth_radius,
            msg.altitude_msl_m - self.home_alt]
        self.obs_gps_vel[:] = [msg.vel_e_m_s, msg.vel_n_m_s, -msg.vel_d_m_s]
        self.gps_updated = True

    def _cb_sensor(self, msg):
        self.cur_accel[:] = msg.accelerometer_m_s2[:3]; self.cur_gyro[:] = msg.gyro_rad[:3]

    def _cb_odometry(self, msg):
        self.cur_pos[:] = msg.position[:3]; self.cur_vel[:] = msg.velocity[:3]

    def _cb_thrust(self, msg): self.cur_thrust[:] = msg.xyz[:3]
    def _cb_torque(self, msg): self.cur_torque[:] = msg.xyz[:3]

    def _cb_gt(self, msg):
        self.gt_pos[:] = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
        self.gt_vel[:] = [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z]
        q = msg.pose.pose.orientation
        self.cur_euler[:] = self._quat_to_euler(q.w, q.x, q.y, q.z)
        self.last_gt_time = pytime.time()

    # ══════════════════════════════════════════════════════════
    #  Utilities
    # ══════════════════════════════════════════════════════════
    @staticmethod
    def _quat_to_euler(w, x, y, z):
        return [np.arctan2(2*(w*x+y*z), 1-2*(x**2+y**2)),
                np.arcsin(np.clip(2*(w*y-z*x), -1, 1)),
                np.arctan2(2*(w*z+x*y), 1-2*(y**2+z**2))]

    def _send_setpoint(self, x, y, z, yaw):
        msg = TrajectorySetpoint(); msg.position = [float(x), float(y), float(z)]
        msg.yaw = float(yaw); msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.pub_traj.publish(msg)

    def _vehicle_cmd(self, command, p1, p2=0.0):
        msg = VehicleCommand(); msg.command = command; msg.param1 = float(p1); msg.param2 = float(p2)
        msg.target_system = 1; msg.target_component = 1; msg.source_system = 1; msg.source_component = 1
        msg.from_external = True; msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.pub_cmd.publish(msg)

    def _publish_offboard(self):
        off = OffboardControlMode(); off.position = True
        off.timestamp = int(self.get_clock().now().nanoseconds / 1000); self.pub_offboard.publish(off)

    def _send_attack_cmd(self, active, attack_type='none', intensity=0.0):
        msg = String(); msg.data = json.dumps({'active': active, 'type': attack_type,
            'intensity': intensity, 'ramp_duration': self.cfg.attack_ramp_duration})
        self.pub_attack.publish(msg)

    def _send_scenario_cmd(self):
        msg = String(); msg.data = json.dumps({'disturbance_type': self.scenario['disturbance_type'],
            'wind_speed': self.scenario['wind_speed']})
        self.pub_scenario.publish(msg)

    def _send_sim_reset(self):
        msg = String(); msg.data = 'reset'; self.pub_sim_ctrl.publish(msg)

    def _check_heartbeat(self):
        return (pytime.time() - self.last_gt_time) < self.heartbeat_timeout

    # ══════════════════════════════════════════════════════════
    #  Flight Patterns
    # ══════════════════════════════════════════════════════════
    def _compute_setpoint(self):
        alt = -abs(self.cfg.flight_altitude); dt = self.step_dt
        t = self.tick_count * dt; R = self.cfg.flight_radius; w = self.cfg.flight_omega
        pattern = self.scenario['pattern'] if self.scenario else 'hover'

        if pattern == 'hover': return (0.0, 0.0, alt, 0.0)
        elif pattern == 'circle':
            x = R*np.cos(self.theta); y = R*np.sin(self.theta)
            yaw = self.theta + np.pi/2; self.theta += w*dt; return (x, y, alt, yaw)
        elif pattern == 'figure8':
            x = R*np.sin(w*t); y = (R/2)*np.sin(2*w*t)
            return (x, y, alt, np.arctan2(R*w*np.cos(2*w*t), R*w*np.cos(w*t)))
        elif pattern == 'waypoint':
            wps = np.array([[0,0],[5,0],[5,5],[-5,5],[-5,0],[0,0]]); seg = 4.0
            t_mod = t%(seg*(len(wps)-1)); idx = int(t_mod/seg); f = (t_mod%seg)/seg
            x = wps[idx][0]+(wps[idx+1][0]-wps[idx][0])*f; y = wps[idx][1]+(wps[idx+1][1]-wps[idx][1])*f
            return (float(x), float(y), alt, np.arctan2(wps[idx+1][1]-wps[idx][1], wps[idx+1][0]-wps[idx][0]))
        elif pattern == 'aggressive':
            spp = int(5.0/dt); phase = (self.tick_count//spp)%4; f = (self.tick_count%spp)/spp
            if phase == 0: return (0.0, 0.0, alt-3.0*np.sin(np.pi*f), 0.0)
            elif phase == 1: a = np.pi*f; return (4*np.cos(a), 4*np.sin(a), alt, a)
            elif phase == 2: return (4.0, 0.0, alt+2.0*np.sin(np.pi*f), np.pi)
            else: a = np.pi*(1-f); return (4*np.cos(a), 4*np.sin(a), alt, a)
        return (0.0, 0.0, alt, 0.0)

    # ══════════════════════════════════════════════════════════
    #  Episode State Reset
    # ══════════════════════════════════════════════════════════
    def _reset_episode_state(self):
        self.step_count = 0; self.tick_count = 0; self.stable_counter = 0; self.theta = 0.0
        self.prev_state = None; self.prev_action = None
        self.episode_reward = 0.0; self.episode_losses = []; self.attack_active_flag = False
        self.window_buffer.clear(); self.gps_updated = False
        self.first_hover_step = None; self.hover_before_attack_count = 0
        self.ukf = DynamicsUKF(dt=self.step_dt, calib=self.calib)
        self.is_ukf_initialized = False; self.last_res = np.zeros(9); self.last_Pzz = np.eye(9)

    # ══════════════════════════════════════════════════════════
    #  Episode Start (학습 / 평가 분기)
    # ══════════════════════════════════════════════════════════
    def _start_new_episode(self):
        if self.eval_mode:
            # ── 평가 모드: 고정 시나리오 (episode 카운터 안 올림) ──
            self.scenario = self.cfg.eval_scenarios[self.eval_scenario_idx]
            label = f'EVAL {self.eval_scenario_idx+1}/{len(self.cfg.eval_scenarios)}'
        else:
            # ── 학습 모드: 랜덤 시나리오 ──
            self.episode += 1
            if self.episode > self.cfg.max_episodes:
                self._finish_training(); return
            self.scenario = sample_episode_scenario(self.episode, self.cfg)
            label = f'TRAIN Ep {self.episode}/{self.cfg.max_episodes}'

        atk = self.scenario
        self.get_logger().info(
            f'\n{"="*60}\n  {label}\n'
            f'  Pattern: {atk["pattern"]} | Attack: {atk["attack_type"]} '
            f'(int={atk["attack_intensity"]:.3f}, start={atk["attack_start_step"]}) | '
            f'Wind: {atk.get("disturbance_type","none")} ({atk.get("wind_speed",0):.1f} m/s)\n{"="*60}')

        self._send_scenario_cmd()
        self._reset_episode_state()
        self.home_lat = None; self.init_counter = 0

    def _check_done(self, trajectory_sp):
        dist = math.hypot(self.cur_pos[0]-trajectory_sp[0], self.cur_pos[1]-trajectory_sp[1])
        if dist >= self.cfg.max_error: return True, 'crash_drift'
        if self.cur_pos[2] > self.cfg.min_altitude: return True, 'crash_altitude'
        if abs(self.cur_euler[0]) > 1.05 or abs(self.cur_euler[1]) > 1.05: return True, 'crash_flip'
        if self.step_count >= self.cfg.episode_max_steps: return True, 'timeout'
        return False, None

    def _finish_training(self):
        total = pytime.time() - self.train_start_time
        self.get_logger().info(f'\n{"#"*60}\n  Training Complete | {total:.0f}s ({total/60:.1f}min)\n'
            f'  Episodes: {self.episode-1} | Hard Resets: {self.hard_reset_count}\n{"#"*60}')
        self.agent.save(os.path.join(self.cfg.outdir, 'final_model.pt'))
        # 평가 히스토리 저장
        if self.eval_history:
            np.savez(os.path.join(self.cfg.outdir, 'eval_history.npz'),
                     eval_history=self.eval_history)
        self.sim_mgr.stop(); raise SystemExit("Training complete")

    def _trigger_hard_reset(self):
        self._send_attack_cmd(False); self._reset_episode_state()
        self.home_lat = None; self.init_counter = 0; self.flight_state = 'HARD_RESET'

    # ══════════════════════════════════════════════════════════
    #  Evaluation System
    # ══════════════════════════════════════════════════════════
    def _start_eval_round(self):
        """학습 → 평가 모드 전환"""
        self.eval_mode = True
        self.eval_scenario_idx = 0
        self.current_eval_results = []
        self.get_logger().info(
            f'\n  ╔═══ EVAL Round @ Ep {self.episode} '
            f'({len(self.cfg.eval_scenarios)} scenarios) ═══╗')

    def _record_eval_result(self, reason):
        """평가 에피소드 1개 결과 기록"""
        attack_start = self.scenario.get('attack_start_step', 0)

        det_delay = -1
        if self.first_hover_step is not None and attack_start > 0:
            det_delay = max(0, self.first_hover_step - attack_start)

        fa_rate = 0.0
        if attack_start > 0 and attack_start > 0:
            pre_attack_steps = min(self.step_count, attack_start)
            fa_rate = self.hover_before_attack_count / max(pre_attack_steps, 1)

        result = {
            'scenario_idx': self.eval_scenario_idx,
            'attack_type': self.scenario['attack_type'],
            'intensity': self.scenario['attack_intensity'],
            'pattern': self.scenario['pattern'],
            'survived': reason == 'timeout',
            'reward': self.episode_reward,
            'steps': self.step_count,
            'reward_rate': self.episode_reward / max(self.step_count, 1),
            'det_delay': det_delay,
            'false_alarm_rate': fa_rate,
            'reason': reason,
        }
        self.current_eval_results.append(result)

        surv = '✅' if result['survived'] else '❌'
        dd = f"{det_delay}" if det_delay >= 0 else 'N/A'
        self.get_logger().info(
            f'  ║ Eval {self.eval_scenario_idx+1}: {surv} {reason} | '
            f'R={self.episode_reward:.1f} | Steps={self.step_count} | '
            f'DetDelay={dd} | FA={fa_rate:.2f}')

    def _finish_eval_round(self):
        """평가 완료 → 결과 집계 → 학습 모드 복귀"""
        self.eval_mode = False
        results = self.current_eval_results

        survival_rate = np.mean([r['survived'] for r in results])
        mean_rr = np.mean([r['reward_rate'] for r in results])
        det_delays = [r['det_delay'] for r in results if r['det_delay'] >= 0]
        mean_dd = np.mean(det_delays) if det_delays else -1
        mean_fa = np.mean([r['false_alarm_rate'] for r in results])

        eval_summary = {
            'train_episode': self.episode,
            'survival_rate': float(survival_rate),
            'mean_reward_rate': float(mean_rr),
            'mean_det_delay': float(mean_dd),
            'mean_false_alarm_rate': float(mean_fa),
            'per_scenario': results,
        }
        self.eval_history.append(eval_summary)

        dd_str = f"{mean_dd:.1f}" if mean_dd >= 0 else "N/A"
        self.get_logger().info(
            f'  ╠═ Survival: {survival_rate:.0%} | '
            f'RewardRate: {mean_rr:.3f} | '
            f'DetDelay: {dd_str} | '
            f'FA: {mean_fa:.3f}\n'
            f'  ╚═══════════════════════════════════════════╝')

        # 파일 저장
        np.savez(os.path.join(self.cfg.outdir, 'eval_history.npz'),
                 eval_history=self.eval_history)

    # ══════════════════════════════════════════════════════════
    #  Main Tick (50Hz)
    # ══════════════════════════════════════════════════════════
    def _tick(self):
        # ── Heartbeat ──
        if self.flight_state not in ('IDLE', 'HARD_RESET'):
            if not self._check_heartbeat():
                self.get_logger().error('  💀 Heartbeat lost → HARD_RESET')
                self._trigger_hard_reset(); return

        if self.flight_state in ('SOFT_RECOVERY', 'TAKEOFF', 'STABILIZE', 'LEARNING'):
            self._publish_offboard()

        # ── IDLE ──
        if self.flight_state == 'IDLE':
            self._start_new_episode()
            self.flight_state = 'TAKEOFF'
            self.get_logger().info('  → TAKEOFF')

        # ── SOFT_RECOVERY ──
        elif self.flight_state == 'SOFT_RECOVERY':
            self._send_setpoint(0.0, 0.0, -abs(self.cfg.flight_altitude), 0.0)
            dist = np.linalg.norm(self.cur_pos[:2])
            alt_err = abs(self.cur_pos[2] + self.cfg.flight_altitude)
            if dist < 1.0 and alt_err < 0.5: self.stable_counter += 1
            else: self.stable_counter = 0
            if self.stable_counter >= int(self.cfg.warmup_seconds / self.step_dt):
                self._start_new_episode()
                self.flight_state = 'STABILIZE'; self.stable_counter = 0
                self.get_logger().info('  → STABILIZE (soft)')

        # ── WARM_RESET ──
        elif self.flight_state == 'WARM_RESET':
            self._send_setpoint(0.0, 0.0, -abs(self.cfg.flight_altitude), 0.0)
            self.init_counter += 1
            if self.init_counter == 1:
                self._send_sim_reset()
            if self.init_counter >= int(3.0 / self.step_dt):
                self._vehicle_cmd(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
                self._vehicle_cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
                self._start_new_episode()
                self.flight_state = 'TAKEOFF'

        # ── HARD_RESET ──
        elif self.flight_state == 'HARD_RESET':
            self.init_counter += 1
            if self.init_counter == 1:
                self.get_logger().warn('  [HARD] Restarting simulator...')
                self.sim_mgr.restart(); self.hard_reset_count += 1
                self.last_gt_time = pytime.time()
            if self.init_counter >= int(15.0 / self.step_dt):
                if self._check_heartbeat():
                    self._vehicle_cmd(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
                    self._vehicle_cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
                    self._start_new_episode()
                    self.flight_state = 'TAKEOFF'
                elif self.init_counter >= int(30.0 / self.step_dt):
                    self.get_logger().error('  [HARD] Retry...'); self.init_counter = 0

        # ── TAKEOFF ──
        elif self.flight_state == 'TAKEOFF':
            sp = (0.0, 0.0, -abs(self.cfg.flight_altitude), 0.0)
            self._send_setpoint(*sp); self.init_counter += 1
            if self.init_counter == int(2.0 / self.step_dt):
                self._vehicle_cmd(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
                self._vehicle_cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
            dist = np.linalg.norm(self.cur_pos - np.array(sp[:3]))
            if dist < 1.0: self.stable_counter += 1
            else: self.stable_counter = 0
            if self.stable_counter >= int(self.cfg.warmup_seconds / self.step_dt):
                self.flight_state = 'STABILIZE'; self.stable_counter = 0
                self.get_logger().info('  → STABILIZE')

        # ── STABILIZE ──
        elif self.flight_state == 'STABILIZE':
            self._send_setpoint(*self._compute_setpoint()); self.tick_count += 1
            self._run_ukf_step(); self.stable_counter += 1
            if self.stable_counter >= int(self.cfg.warmup_seconds / self.step_dt):
                self.window_buffer.clear(); self.step_count = 0
                self.flight_state = 'LEARNING'; self.get_logger().info('  → LEARNING')

        # ── LEARNING ──
        elif self.flight_state == 'LEARNING':
            trajectory_sp = self._compute_setpoint()
            if self.prev_action == 1:
                control_sp = (self.cur_pos[0], self.cur_pos[1], -abs(self.cfg.flight_altitude), 0.0)
            else: control_sp = trajectory_sp
            self._send_setpoint(*control_sp); self.tick_count += 1
            self._run_ukf_step()
            if self.gps_updated:
                self.gps_updated = False; self._rl_step_10hz(trajectory_sp)

    # ══════════════════════════════════════════════════════════
    #  UKF Step (50Hz)
    # ══════════════════════════════════════════════════════════
    def _run_ukf_step(self):
        gps_ned = [self.obs_gps_pos[1], self.obs_gps_pos[0], -self.obs_gps_pos[2]]
        vel_ned = [self.obs_gps_vel[1], self.obs_gps_vel[0], -self.obs_gps_vel[2]]
        z_9d = np.concatenate([gps_ned, vel_ned, self.cur_gyro])
        u_phys = to_physical_u(np.array([self.cur_thrust]), np.array([self.cur_torque]), self.calib)[0]
        if not self.is_ukf_initialized:
            self.ukf.x[0:3] = gps_ned; self.ukf.x[3:6] = self.cur_euler
            self.ukf.x[6:9] = vel_ned; self.ukf.x[9:12] = self.cur_gyro
            self.is_ukf_initialized = True
        self.last_res, self.last_Pzz = self.ukf.step(z_9d, u_phys)

    # ══════════════════════════════════════════════════════════
    #  10Hz RL Step (★ 학습/평가 분기)
    # ══════════════════════════════════════════════════════════
    def _rl_step_10hz(self, trajectory_sp):
        cfg = self.cfg

        _, nis_vel = compute_nis_scaled(self.last_res[3:6], self.last_Pzz[3:6, 3:6], 3.0)
        _, nis_gyr = compute_nis_scaled(self.last_res[6:9], self.last_Pzz[6:9, 6:9], 3.0)
        self.window_buffer.append([nis_vel, nis_gyr])

        if len(self.window_buffer) < cfg.window_size:
            self.step_count += 1; return

        state = np.array(self.window_buffer).flatten()
        done, term_reason = self._check_done(trajectory_sp)

        if done and term_reason in ('crash_drift', 'crash_altitude', 'crash_flip'):
            reward = -10.0
        else:
            reward, _, _ = calculate_reward(
                list(self.cur_pos), list(trajectory_sp[:3]),
                self.prev_action if self.prev_action is not None else 0)
        self.episode_reward += reward

        # ── Transition 저장 + 학습 (★ 평가 모드에서는 건너뜀) ──
        if self.prev_state is not None and self.prev_action is not None:
            if not self.eval_mode:
                self.agent.push(self.prev_state, self.prev_action, reward, state, done)
                loss = self.agent.learn()
                if loss > 0: self.episode_losses.append(loss)

        if done: self._end_episode(term_reason); return

        # ── Action 선택 (★ 평가: eps=0 greedy / 학습: eps-greedy) ──
        if self.eval_mode:
            action = self.agent.act(state, eps=0.0)
            eps = 0.0
        else:
            eps = self.agent.get_epsilon()
            action = self.agent.act(state, eps)

        # ── Detection tracking ──
        if action == 1:
            if self.first_hover_step is None:
                self.first_hover_step = self.step_count
            if not self.attack_active_flag:
                self.hover_before_attack_count += 1

        # ── Attack start (random timing) ──
        attack_step = self.scenario.get('attack_start_step', 0)
        if self.step_count == attack_step and self.scenario['attack_type'] != 'none':
            self._send_attack_cmd(True, self.scenario['attack_type'], self.scenario['attack_intensity'])
            self.attack_active_flag = True
            self.get_logger().warn(
                f'  🚨 Attack ON @ step {self.step_count}: {self.scenario["attack_type"]} '
                f'(int={self.scenario["attack_intensity"]:.3f})')

        # ── Debug log ──
        if self.step_count % cfg.log_interval == 0:
            sp = np.array(trajectory_sp[:3])
            ctrl_err = np.linalg.norm(self.cur_pos[:2] - sp[:2])
            gt_ned = np.array([self.gt_pos[1], self.gt_pos[0], -self.gt_pos[2]])
            gt_err = np.linalg.norm(gt_ned[:2] - sp[:2])
            alt = -self.cur_pos[2] if self.cur_pos[2] < 0 else 0.0
            atk = '🔴ATK' if self.attack_active_flag else '⚪NRM'
            act = 'HOVER' if action == 1 else 'TRACK'
            mode = 'EVAL' if self.eval_mode else 'TRAIN'
            self.get_logger().info(
                f'  [{self.step_count:3d}] {mode} {atk} {act} | ε={eps:.3f} | '
                f'NIS v={nis_vel:.3f} g={nis_gyr:.3f} | R={reward:+.2f} (Σ={self.episode_reward:.1f}) | '
                f'GT={gt_err:.2f}m ctrl={ctrl_err:.2f}m alt={alt:.1f}m')

        self.prev_state = state; self.prev_action = action; self.step_count += 1

    # ══════════════════════════════════════════════════════════
    #  Episode End — 3-tier reset + eval 분기
    # ══════════════════════════════════════════════════════════
    def _end_episode(self, reason):
        self._send_attack_cmd(False); self.attack_active_flag = False

        # ── 평가 모드: 결과 기록 + 다음 시나리오 or 평가 완료 ──
        if self.eval_mode:
            self._record_eval_result(reason)
            self.eval_scenario_idx += 1
            if self.eval_scenario_idx >= len(self.cfg.eval_scenarios):
                self._finish_eval_round()
                # 평가 완료 → 학습 모드 복귀, 다음 에피소드 시작
                self._reset_episode_state(); self.stable_counter = 0
                self.flight_state = 'SOFT_RECOVERY'
                return
            else:
                # 다음 평가 시나리오
                self._reset_episode_state(); self.stable_counter = 0
                self.flight_state = 'SOFT_RECOVERY'
                return

        # ── 학습 모드: 통계 기록 ──
        self.agent.end_episode(self.episode_reward, self.step_count)

        avg_loss = np.mean(self.episode_losses) if self.episode_losses else 0
        eps = self.agent.get_epsilon(); p_init = self.agent._compute_adaptive_p()
        emojis = {'crash_drift': '⚠️ DRIFT', 'crash_altitude': '💀 CRASH',
                  'crash_flip': '🔥 FLIP', 'timeout': '⏱️ TIMEOUT'}
        reset_map = {'crash_flip': 'HARD', 'crash_altitude': 'WARM',
                     'crash_drift': 'SOFT', 'timeout': 'SOFT'}

        self.get_logger().info(
            f'\n  ┌─ Ep {self.episode}: {emojis.get(reason, reason)} → {reset_map.get(reason, "?")} reset\n'
            f'  │ R={self.episode_reward:.1f} Steps={self.step_count} Loss={avg_loss:.4f}\n'
            f'  │ ε={eps:.3f} P={p_init:.5f}\n'
            f'  │ Atk: {self.scenario["attack_type"]}(int={self.scenario["attack_intensity"]:.3f}, '
            f'start={self.scenario["attack_start_step"]}) | {self.scenario["pattern"]} | '
            f'{self.scenario.get("disturbance_type","none")}\n  └─{"─"*50}')

        if self.episode % 50 == 0:
            self.agent.save(os.path.join(self.cfg.outdir, f'model_ep{self.episode}.pt'))

        # ── 평가 시점 체크 (학습 에피소드 끝난 직후) ──
        if self.episode % self.cfg.eval_interval == 0:
            self._start_eval_round()
            # eval_mode=True, 다음 _start_new_episode에서 평가 시나리오 사용됨

        # ── 3단계 리셋 분기 ──
        if reason == 'crash_flip':
            self._trigger_hard_reset()
        elif reason == 'crash_altitude':
            self._reset_episode_state(); self.home_lat = None; self.init_counter = 0
            self.flight_state = 'WARM_RESET'
        else:
            self._reset_episode_state(); self.stable_counter = 0
            self.flight_state = 'SOFT_RECOVERY'


def main():
    cfg = Config()
    if cfg.use_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    rclpy.init(); node = OnlineRLNode(cfg)
    try: rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit) as e: node.get_logger().info(f'Shutdown: {e}')
    finally: node.sim_mgr.stop(); node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()

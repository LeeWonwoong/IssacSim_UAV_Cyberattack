#!/usr/bin/env python3
"""
fly_and_record_v3.py (Realistic Raw Sensor Preprocessing 적용)
================================================================
* [Update]: 50Hz 처리 주기 적용
* [Update]: SensorGps 위경도 -> Local 미터(m) 선형 전처리 (UKF H행렬 단순화)
* [Update]: Optical Flow 자이로 회전 보상 역산 적용 (순수 선형속도 복원)
================================================================
"""

"""
[실행 가이드]
# ──────── 정상 비행 (캘리브레이션 / 검증용) ────────
$ python3 fly_and_record.py --pattern figure8 --ep-id 0 --steps 2500 --omega 0.8 --radius 6
$ python3 fly_and_record.py --pattern aggressive --ep-id 1 --steps 3000
$ python3 fly_and_record.py --pattern hover --ep-id 2 --steps 1500
$ python3 fly_and_record.py --pattern circle --ep-id 3 --steps 2500
$ python3 fly_and_record.py --pattern waypoint --ep-id 4 --steps 2500

# ──────── GPS 센서 공격 ────────
# FDI Constant (위치 +3m 편향, 기록 5초 후 10초간)
$ python3 fly_and_record.py --pattern circle --ep-id 10 --steps 2500 --sensor-attack fdi_constant --sensor-start 5.0 --sensor-duration 10.0 --sensor-intensity 3.0

# FDI Ramp (점진적 증가, 초당 0.5m)
$ python3 fly_and_record.py --pattern circle --ep-id 11 --steps 2500 --sensor-attack fdi_ramp --sensor-start 5.0 --sensor-duration 10.0 --sensor-intensity 0.5

# Jamming (GPS 노이즈 폭발)
$ python3 fly_and_record.py --pattern circle --ep-id 12 --steps 2500 --sensor-attack jamming --sensor-start 5.0 --sensor-duration 10.0 --sensor-intensity 2.0

# ──────── 액추에이터 하이재킹 ────────
# 전체 모터 비대칭 조작 (비틀거림)
$ python3 fly_and_record.py --pattern figure8 --ep-id 20 --steps 2500 --hijack-attack override_all --hijack-start 5.0 --hijack-duration 8.0 --hijack-intensity 0.15

# 추력 강제 고정
$ python3 fly_and_record.py --pattern circle --ep-id 21 --steps 2500 --hijack-attack override_thrust --hijack-start 5.0 --hijack-duration 5.0 --hijack-intensity 0.3

# 추력 바이어스 추가
$ python3 fly_and_record.py --pattern circle --ep-id 22 --steps 2500 --hijack-attack bias_thrust --hijack-start 5.0 --hijack-duration 8.0 --hijack-intensity 0.1

# ──────── 복합 (외란 + 공격) ────────
$ python3 fly_and_record.py --pattern circle --ep-id 30 --steps 2500 --disturbance-scenario wind_turbulence --sensor-attack fdi_constant --sensor-start 8.0 --sensor-duration 10.0 --sensor-intensity 2.0
"""


import rclpy
import numpy as np
import argparse
import os
import math
import json
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from px4_msgs.msg import (
    OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleStatus,
    VehicleAttitude, SensorCombined, VehicleOdometry, ActuatorMotors,
    VehicleThrustSetpoint, VehicleTorqueSetpoint, 
    VehicleAirData, DistanceSensor, SensorOpticalFlow, SensorGps
)
from nav_msgs.msg import Odometry as GroundTruthOdometry
from geometry_msgs.msg import Vector3Stamped
from std_msgs.msg import String


class FlyAndRecord(Node):
    def __init__(self, args):
        super().__init__('fly_and_record')
        self.args = args
        self.step_dt = args.step_dt
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST, depth=5
        )

        self.pub_offboard = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos)
        self.pub_traj = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos)
        self.pub_cmd = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', qos)
        self.pub_attack_cfg = self.create_publisher(String, '/attack_config', 10)

        self.create_subscription(VehicleStatus, '/fmu/out/vehicle_status', self._cb_status, qos)
        self.create_subscription(VehicleAttitude, '/fmu/out/vehicle_attitude', self._cb_attitude, qos)
        self.create_subscription(SensorCombined, '/fmu/out/sensor_combined', self._cb_sensor, qos)
        
        # 기체 제어를 위해 Odometry는 수신하되, 관측값(RL)으로는 쓰지 않음
        self.create_subscription(VehicleOdometry, '/fmu/out/vehicle_odometry', self._cb_odometry, qos)
        self.create_subscription(GroundTruthOdometry, '/gt/odometry', self._cb_gt, qos)
        
        self.create_subscription(VehicleThrustSetpoint, '/fmu/out/vehicle_thrust_setpoint', self._cb_thrust, qos)
        self.create_subscription(VehicleTorqueSetpoint, '/fmu/out/vehicle_torque_setpoint', self._cb_torque, qos)
        self.create_subscription(ActuatorMotors, '/fmu/out/actuator_motors', self._cb_motors, qos)
        self.create_subscription(Vector3Stamped, '/disturbance/force', self._cb_disturbance, 10)
        
        # ★ 다이렉트 수신 채널 연결 (Raw Sensors)
        self.create_subscription(SensorGps, '/sim/sensor_gps', self._cb_gps, qos)
        self.create_subscription(VehicleAirData, '/sim/sensor_baro', self._cb_baro, qos)
        self.create_subscription(DistanceSensor, '/sim/distance_sensor', self._cb_distance, qos)
        self.create_subscription(SensorOpticalFlow, '/sim/sensor_optical_flow', self._cb_flow, qos)

        # 상태 변수
        self.cur_accel = np.zeros(3); self.cur_gyro = np.zeros(3)
        self.cur_pos = np.zeros(3); self.cur_vel = np.zeros(3) # Control용
        self.cur_euler = np.zeros(3)
        self.gt_pos = np.zeros(3); self.gt_vel = np.zeros(3)
        self.gt_euler = np.zeros(3); self.gt_angvel = np.zeros(3)
        self.cur_thrust = np.zeros(3); self.cur_torque = np.zeros(3)
        self.cur_motors = np.zeros(4)
        self.cur_disturbance_enu = np.zeros(3)
        
        # 관측용 Raw 변수
        self.obs_gps_pos = np.zeros(3)
        self.obs_gps_vel = np.zeros(3)
        self.cur_baro_alt = 0.0          
        self.cur_dist_bottom = 0.0       
        self.cur_flow_vel = np.zeros(2)  

        # ★ Baro 영점 조절용 변수 추가
        self.is_baro_calibrated = False
        self.baro_offset = 0.0


        # GPS Home 초기화 변수
        self.home_lat = None
        self.home_lon = None
        self.home_alt = None
        self.earth_radius = 6371000.0

        N = args.steps
        self.buf_accel = np.zeros((N, 3)); self.buf_gyro = np.zeros((N, 3))
        self.buf_pos = np.zeros((N, 3)); self.buf_vel = np.zeros((N, 3)) # Control Log
        self.buf_euler = np.zeros((N, 3))
        self.buf_gt_pos = np.zeros((N, 3)); self.buf_gt_vel = np.zeros((N, 3))
        self.buf_gt_euler = np.zeros((N, 3)); self.buf_gt_angvel = np.zeros((N, 3))
        self.buf_thrust = np.zeros((N, 3)); self.buf_torque = np.zeros((N, 3))
        self.buf_motors = np.zeros((N, 4)); self.buf_setpoint = np.zeros((N, 4))
        self.buf_disturbance_enu = np.zeros((N, 3)); self.buf_disturbance_ned = np.zeros((N, 3))
        self.buf_disturbance_label = np.zeros(N, dtype=np.int32)
        self.buf_sensor_label = np.zeros(N, dtype=np.int32)
        self.buf_hijack_label = np.zeros(N, dtype=np.int32)
        
        # RL 관측용 버퍼 (UKF 주입용)
        self.buf_obs_gps_pos = np.zeros((N, 3))
        self.buf_obs_gps_vel = np.zeros((N, 3))
        self.buf_baro_alt = np.zeros(N)
        self.buf_dist_bottom = np.zeros(N)
        self.buf_flow_vel = np.zeros((N, 2))

        self.flight_state = 'INIT'
        self.init_counter = 0; self.stable_counter = 0
        self.step = 0; self.record_step = 0; self.done = False; self.theta = 0.0
        self._prev_sensor_atk = False; self._prev_hijack_atk = False

        os.makedirs(args.output_dir, exist_ok=True)
        self.timer = self.create_timer(self.step_dt, self._tick)
        hz = 1.0 / self.step_dt
        self.get_logger().info(f'[시작] pattern={args.pattern}, steps={args.steps}, dt={self.step_dt}s ({hz:.0f}Hz)')

    def _cb_status(self, msg): pass
    def _cb_attitude(self, msg): self.cur_euler[:] = self._quat_to_euler(msg.q[0], msg.q[1], msg.q[2], msg.q[3])
    def _cb_sensor(self, msg): self.cur_accel[:] = msg.accelerometer_m_s2[:3]; self.cur_gyro[:] = msg.gyro_rad[:3]
    def _cb_odometry(self, msg): self.cur_pos[:] = msg.position[:3]; self.cur_vel[:] = msg.velocity[:3]
    def _cb_thrust(self, msg): self.cur_thrust[:] = msg.xyz[:3]
    def _cb_torque(self, msg): self.cur_torque[:] = msg.xyz[:3]
    def _cb_motors(self, msg):
        n_motors = min(4, len(msg.control))
        self.cur_motors[:n_motors] = msg.control[:n_motors]
    def _cb_disturbance(self, msg): self.cur_disturbance_enu[:] = [msg.vector.x, msg.vector.y, msg.vector.z]

    def _cb_gt(self, msg):
        # Isaac Sim ENU: position.x=E, position.y=N, position.z=U
        self.gt_pos[:] = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]  # ENU
        self.gt_vel[:] = [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z]  # ENU
        self.gt_angvel[:] = [msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z]  # ENU
        q = msg.pose.pose.orientation
        self.gt_euler[:] = self._quat_to_euler(q.w, q.x, q.y, q.z)

    # ==========================================================
    # ★ 관측치 전처리 콜백 (UKF 선형화를 위한 복원)
    # ==========================================================
    def _cb_gps(self, msg):
        # 1. 첫 수신 좌표를 Home으로 설정
        if self.home_lat is None:
            self.home_lat = msg.latitude_deg
            self.home_lon = msg.longitude_deg
            self.home_alt = msg.altitude_msl_m
            self.get_logger().info(f'📍 GPS Home 설정: {self.home_lat:.4f}, {self.home_lon:.4f}')

        # 2. LLA -> Local ENU(미터) 변환
        lat_rad = math.radians(self.home_lat)
        x_m = math.radians(msg.longitude_deg - self.home_lon) * self.earth_radius * math.cos(lat_rad)
        y_m = math.radians(msg.latitude_deg - self.home_lat) * self.earth_radius
        z_m = msg.altitude_msl_m - self.home_alt

        self.obs_gps_pos[:] = [x_m, y_m, z_m]
        
        # 속도는 시뮬레이터와 동일하게 ENU 배열로 맞춤 (px4 msg는 NED)
        self.obs_gps_vel[:] = [msg.vel_e_m_s, msg.vel_n_m_s, -msg.vel_d_m_s]

    def _cb_baro(self, msg):
        # 기압계 고도를 UKF가 사용하는 NED 좌표계(음수 고도)로 일단 변환
        raw_baro_ned = -msg.baro_alt_meter
        
        # ★ 초기 영점 캘리브레이션 (GPS의 cur_pos[2]와 고도를 일치시킴)
        if not self.is_baro_calibrated:
            # INIT 상태에서 Odometry 통신이 안정화될 때까지 잠시 대기 (5 ticks)
            if self.init_counter > 5:
                self.baro_offset = raw_baro_ned - self.cur_pos[2]
                self.is_baro_calibrated = True
                self.get_logger().info(f'✅ Barometer Calibrated: Offset={self.baro_offset:.3f}m')
            else:
                self.cur_baro_alt = raw_baro_ned
                return
                
        # 캘리브레이션이 끝난 후에는 항상 Offset을 빼서 GPS Z축과 동기화
        self.cur_baro_alt = raw_baro_ned - self.baro_offset

    def _cb_distance(self, msg):
        self.cur_dist_bottom = msg.current_distance

    def _cb_flow(self, msg):
        dt = msg.integration_timespan_us / 1e6 
        dist = msg.distance_m if msg.distance_available else self.cur_dist_bottom
        
        if dt > 0 and dist > 0.1:
            # 생성 시 동봉된 자이로 값 사용 
            w_x = msg.delta_angle[0] / dt
            w_y = msg.delta_angle[1] / dt
            
            flow_rate_x = msg.pixel_flow[0] / dt
            flow_rate_y = msg.pixel_flow[1] / dt
            
            # 자이로(회전) 성분을 빼서 순수 선형 속도로 복원
            vx = (flow_rate_x + w_y) * dist
            vy = (flow_rate_y - w_x) * dist
            
            self.cur_flow_vel[:] = [vx, vy]
        else:
            self.cur_flow_vel[:] = [0.0, 0.0]

    def _quat_to_euler(self, w, x, y, z):
        roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x**2 + y**2))
        pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
        yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))
        return [roll, pitch, yaw]

    @staticmethod
    def _enu_to_ned(v_enu): return np.array([v_enu[1], v_enu[0], -v_enu[2]])

    def _tick(self):
        if self.done: return
        off = OffboardControlMode()
        off.position = True
        off.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.pub_offboard.publish(off)

        if self.flight_state == 'INIT':
            sp = (0.0, 0.0, -abs(self.args.altitude), 0.0)
            self._send_setpoint(*sp)
            self.init_counter += 1
            if self.init_counter == int(1.0 / self.step_dt):
                self.get_logger().info('✅ Offboard 활성화 및 이륙 (중앙 이륙)')
                self._vehicle_cmd(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
                self._vehicle_cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
                self.flight_state = 'TAKEOFF'

        elif self.flight_state == 'TAKEOFF':
            sp = (0.0, 0.0, -abs(self.args.altitude), 0.0)
            self._send_setpoint(*sp)
            target_pos = np.array(sp[0:3])
            dist = np.linalg.norm(self.cur_pos - target_pos)
            if dist < 1.0: self.stable_counter += 1
            else: self.stable_counter = 0
            if self.stable_counter >= int(3.0 / self.step_dt):
                self.get_logger().info('✅ 이륙 고도 및 호버링 안정화 완료! 궤도 기동(예열) 시작!')
                self.flight_state = 'WARMUP'
                self.step = 0

        elif self.flight_state == 'WARMUP':
            sp = self._compute_setpoint()
            self._send_setpoint(*sp)
            self.step += 1
            if self.step >= int(3.0 / self.step_dt):
                self.get_logger().info('✅ 기동 예열 완료! 깨끗한 관측 데이터 기록 시작 (20Hz)!')
                self.flight_state = 'RECORDING'
                self.record_step = 0

        elif self.flight_state == 'RECORDING':
            if self.record_step >= self.args.steps:
                # 공격 종료 신호 전송
                if self._prev_sensor_atk:
                    self._send_attack_config('sensor', False, self.args.sensor_attack, self.args.sensor_intensity)
                if self._prev_hijack_atk:
                    self._send_attack_config('hijack', False, self.args.hijack_attack, self.args.hijack_intensity)
                self._finish()
                return
            sp = self._compute_setpoint()
            self._send_setpoint(*sp)
            
            # ★ 공격 타이밍 판별 (기록 시작 이후 상대 시간 기준)
            t_rel = self.record_step * self.step_dt
            
            cur_sensor_atk = False
            if self.args.sensor_attack != 'none':
                if self.args.sensor_start <= t_rel < (self.args.sensor_start + self.args.sensor_duration):
                    cur_sensor_atk = True
            if cur_sensor_atk != self._prev_sensor_atk:
                self._send_attack_config('sensor', cur_sensor_atk, self.args.sensor_attack, self.args.sensor_intensity)
                self._prev_sensor_atk = cur_sensor_atk

            cur_hijack_atk = False
            if self.args.hijack_attack != 'none':
                if self.args.hijack_start <= t_rel < (self.args.hijack_start + self.args.hijack_duration):
                    cur_hijack_atk = True
            if cur_hijack_atk != self._prev_hijack_atk:
                self._send_attack_config('hijack', cur_hijack_atk, self.args.hijack_attack, self.args.hijack_intensity)
                self._prev_hijack_atk = cur_hijack_atk

            self._record(sp, cur_sensor_atk, cur_hijack_atk)
            
            if self.record_step % (5 * int(1.0 / self.step_dt)) == 0:
                self.get_logger().info(
                    f'  [기록 중: {self.record_step}/{self.args.steps}] t={t_rel:.1f}s '
                    f'Sensor:{cur_sensor_atk} Hijack:{cur_hijack_atk} '
                    f'GPS=({self.obs_gps_pos[0]:.1f}, {self.obs_gps_pos[1]:.1f}) '
                    f'Flow=({self.cur_flow_vel[0]:.2f}, {self.cur_flow_vel[1]:.2f})')
            self.step += 1
            self.record_step += 1

    def _send_attack_config(self, target, active, atk_type, intensity):
        """px4.py에 공격 ON/OFF 명령 전송"""
        msg = String()
        payload = {'target': target, 'active': active, 'type': atk_type, 'intensity': intensity}
        msg.data = json.dumps(payload)
        self.pub_attack_cfg.publish(msg)
        status = "ON" if active else "OFF"
        self.get_logger().warn(f'🚨 [{target.upper()} ATTACK {status}] Type: {atk_type}, Intensity: {intensity}')

    def _compute_setpoint(self):
        alt = -abs(self.args.altitude)
        dt = self.step_dt; t = self.step * dt; R = self.args.radius; w = self.args.omega
        if self.args.pattern == 'hover': return (0.0, 0.0, alt, 0.0)
        elif self.args.pattern == 'circle':
            x = R * np.cos(self.theta); y = R * np.sin(self.theta); yaw = self.theta + np.pi / 2; self.theta += w * dt
            return (x, y, alt, yaw)
        elif self.args.pattern == 'figure8':
            x = R * np.sin(w * t); y = (R / 2) * np.sin(2 * w * t)
            vx = R * w * np.cos(w * t); vy = R * w * np.cos(2 * w * t)
            return (x, y, alt, np.arctan2(vy, vx))
        elif self.args.pattern == 'waypoint':
            wps = np.array([[0, 0], [5, 0], [5, 5], [-5, 5], [-5, 0], [0, 0]])
            seg_time = 4.0
            total_time = seg_time * (len(wps) - 1)
            t_mod = t % total_time
            idx = int(t_mod / seg_time)
            f = (t_mod % seg_time) / seg_time
            wp_curr = wps[idx]
            wp_next = wps[idx + 1]
            x = wp_curr[0] + (wp_next[0] - wp_curr[0]) * f
            y = wp_curr[1] + (wp_next[1] - wp_curr[1]) * f
            yaw = np.arctan2(wp_next[1] - wp_curr[1], wp_next[0] - wp_curr[0])
            return (float(x), float(y), alt, yaw)

        elif self.args.pattern == 'aggressive':
            steps_per_phase = int(5.0 / dt)
            phase = (self.step // steps_per_phase) % 4
            f = (self.step % steps_per_phase) / steps_per_phase
            if phase == 0:
                z = alt - 3.0 * np.sin(np.pi * f)
                return (0.0, 0.0, z, 0.0)
            elif phase == 1:
                angle = np.pi * f
                return (4*np.cos(angle), 4*np.sin(angle), alt, angle)
            elif phase == 2:
                z = alt + 2.0 * np.sin(np.pi * f)
                return (4.0, 0.0, z, np.pi)
            else:
                angle = np.pi * (1 - f)
                return (4*np.cos(angle), 4*np.sin(angle), alt, angle)

        return (0.0, 0.0, alt, 0.0)


    def _record(self, sp, is_sensor_atk=False, is_hijack_atk=False):
        k = self.record_step
        self.buf_accel[k] = self.cur_accel; self.buf_gyro[k] = self.cur_gyro
        self.buf_pos[k] = self.cur_pos; self.buf_vel[k] = self.cur_vel
        self.buf_euler[k] = self.cur_euler
        self.buf_gt_pos[k] = self.gt_pos; self.buf_gt_vel[k] = self.gt_vel
        self.buf_gt_euler[k] = self.gt_euler; self.buf_gt_angvel[k] = self.gt_angvel
        self.buf_thrust[k] = self.cur_thrust; self.buf_torque[k] = self.cur_torque
        self.buf_motors[k] = self.cur_motors; self.buf_setpoint[k] = list(sp)
        self.buf_disturbance_enu[k] = self.cur_disturbance_enu
        self.buf_disturbance_ned[k] = self._enu_to_ned(self.cur_disturbance_enu)
        
        # ★ RL UKF 필터 주입용 관측치 저장
        self.buf_obs_gps_pos[k] = self.obs_gps_pos
        self.buf_obs_gps_vel[k] = self.obs_gps_vel
        self.buf_baro_alt[k] = self.cur_baro_alt
        self.buf_dist_bottom[k] = self.cur_dist_bottom
        self.buf_flow_vel[k] = self.cur_flow_vel

        # ★ 공격/외란 라벨 기록
        self.buf_sensor_label[k] = 1 if is_sensor_atk else 0
        self.buf_hijack_label[k] = 1 if is_hijack_atk else 0
        if np.linalg.norm(self.cur_disturbance_enu) > 0.1: self.buf_disturbance_label[k] = 1

    def _finish(self):
        self.done = True
        n = self.record_step
        
        # 파일명에 공격/외란 태그 추가
        d_tag = f"_{self.args.disturbance_scenario}" if self.args.disturbance_scenario != 'none' else ""
        s_tag = f"_S-{self.args.sensor_attack}" if self.args.sensor_attack != 'none' else ""
        h_tag = f"_H-{self.args.hijack_attack}" if self.args.hijack_attack != 'none' else ""
        fname = os.path.join(self.args.output_dir, f'ep{self.args.ep_id:04d}_{self.args.pattern}{d_tag}{s_tag}{h_tag}.npz')

        np.savez_compressed(fname,
            accelerometer=self.buf_accel[:n], gyro=self.buf_gyro[:n],
            position=self.buf_pos[:n], velocity=self.buf_vel[:n], euler=self.buf_euler[:n],
            gt_pos=self.buf_gt_pos[:n], gt_vel=self.buf_gt_vel[:n],
            gt_euler=self.buf_gt_euler[:n], gt_angvel=self.buf_gt_angvel[:n],
            thrust=self.buf_thrust[:n], torque=self.buf_torque[:n], motors=self.buf_motors[:n],
            disturbance_force=self.buf_disturbance_enu[:n], disturbance_force_ned=self.buf_disturbance_ned[:n],
            disturbance_label=self.buf_disturbance_label[:n], disturbance_scenario=self.args.disturbance_scenario,
            setpoint=self.buf_setpoint[:n], pattern=self.args.pattern,
            episode_id=self.args.ep_id, dt=self.step_dt, steps=n,
            # Raw 전처리 관측값
            obs_gps_pos=self.buf_obs_gps_pos[:n], obs_gps_vel=self.buf_obs_gps_vel[:n],
            baro_alt=self.buf_baro_alt[:n], dist_bottom=self.buf_dist_bottom[:n], flow_vel=self.buf_flow_vel[:n],
            # ★ 공격 라벨 (실시간 기록)
            label_sensor=self.buf_sensor_label[:n], label_hijack=self.buf_hijack_label[:n]
        )

        self.get_logger().info(f'💾 저장 완료: {fname} ({n} steps)')
        self._vehicle_cmd(VehicleCommand.VEHICLE_CMD_NAV_LAND, 0.0)
        self.get_logger().info('🛬 착륙 명령 전송. 2초 후 자동 종료합니다.')
        import time, sys
        time.sleep(2.0)
        os._exit(0)

    def _send_setpoint(self, x, y, z, yaw):
        msg = TrajectorySetpoint()
        msg.position = [float(x), float(y), float(z)]; msg.yaw = float(yaw)
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.pub_traj.publish(msg)

    def _vehicle_cmd(self, command, p1, p2=0.0):
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = float(p1); msg.param2 = float(p2)
        msg.target_system = 1; msg.target_component = 1
        msg.source_system = 1; msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.pub_cmd.publish(msg)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pattern', default='circle', choices=['hover', 'circle', 'figure8', 'waypoint', 'aggressive', 'manual'])
    parser.add_argument('--ep-id', type=int, default=0)
    parser.add_argument('--steps', type=int, default=2500) # 50Hz 기준 50초 비행
    parser.add_argument('--step-dt', type=float, default=0.02, help='샘플링 주기 (0.02 = 50Hz, UKF 주입용)')
    parser.add_argument('--altitude', type=float, default=5.0)
    parser.add_argument('--radius', type=float, default=5.0)
    parser.add_argument('--omega', type=float, default=0.5)
    parser.add_argument('--output-dir', default='data_raw')
    parser.add_argument('--disturbance-scenario', default='none',
        choices=['none', 'wind_constant', 'wind_gust', 'wind_turbulence',
                 'impact_projectile', 'impact_obstacle', 'impact_falling'])
    
    # ★ 센서 공격 인자
    parser.add_argument('--sensor-attack', default='none', choices=['none', 'fdi_constant', 'fdi_ramp', 'jamming'])
    parser.add_argument('--sensor-start', type=float, default=5.0, help='공격 시작 시간 (기록 시작 기준, 초)')
    parser.add_argument('--sensor-duration', type=float, default=3.0, help='공격 지속 시간 (초)')
    parser.add_argument('--sensor-intensity', type=float, default=2.0, help='공격 강도 (m)')
    
    # ★ 액추에이터 하이재킹 인자
    parser.add_argument('--hijack-attack', default='none', 
        choices=['none', 'bias_thrust', 'override_thrust', 'override_roll', 'override_yaw', 'override_all'])
    parser.add_argument('--hijack-start', type=float, default=5.0)
    parser.add_argument('--hijack-duration', type=float, default=3.0)
    parser.add_argument('--hijack-intensity', type=float, default=0.15)
    args = parser.parse_args()

    rclpy.init()
    node = FlyAndRecord(args)
    try: rclpy.spin(node)
    except KeyboardInterrupt:
        if not node.done: node._finish()
    finally:
        node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': main()
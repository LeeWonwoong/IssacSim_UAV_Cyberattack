#!/usr/bin/env python
"""
1_px4_single_vehicle_disturbance.py (Realistic Raw Sensor Simulation)
====================================================================
Isaac Sim + Pegasus + PX4 with 물리 외란 (바람 + 실제 충돌)
* [Update]: GPS (10Hz, LLA 변환), Flow/Baro (50Hz, 자이로 회전 보상 적용)
====================================================================
"""
"""
[실행 가이드]
# 1. 기본 (외란 없음)
$ python3 px4.py

# 2. 돌풍 외란
$ python3 px4.py --scenario wind_gust --wind-speed 7.0 --gust-start 10.0 --gust-duration 3.0

# 3. 지속 난기류
$ python3 px4.py --scenario wind_turbulence --wind-speed 5.0 --turbulence-intensity 0.5

# 4. 투사체 충돌
$ python3 px4.py --scenario impact_projectile --proj-count 3 --proj-start 15.0 --proj-interval 2.0

# ※ 공격은 fly_and_record에서 지휘하므로 px4.py에 공격 인자는 없음
"""

import carb
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import argparse, time, math, json
import numpy as np
import omni.timeline, omni.usd
from omni.isaac.core.world import World
from omni.isaac.core.prims import RigidPrimView
from pxr import UsdGeom, UsdPhysics, PhysxSchema, Gf, Sdf

from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.backends.px4_mavlink_backend import (
    PX4MavlinkBackend, PX4MavlinkBackendConfig)
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface

from scipy.spatial.transform import Rotation
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Vector3Stamped
from std_msgs.msg import String

# ★ 다이렉트 퍼블리시용 센서 메시지 임포트 (VehicleGpsPosition 추가)
from px4_msgs.msg import VehicleAirData, DistanceSensor, SensorOpticalFlow, SensorGps

# ======================================================================
#  PhysX 충돌체 & 바람 모델 (기존 동일)
# ======================================================================
def spawn_rigid_sphere(stage, path, position, radius=0.05, mass=0.3, color=(1.0, 0.2, 0.2), velocity=None, restitution=0.5):
    xform = UsdGeom.Xform.Define(stage, path)
    xform.AddTranslateOp().Set(Gf.Vec3d(*position))
    sphere_path = f"{path}/sphere"
    sphere = UsdGeom.Sphere.Define(stage, sphere_path)
    sphere.GetRadiusAttr().Set(radius)
    sphere.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
    root_prim = stage.GetPrimAtPath(path)
    UsdPhysics.RigidBodyAPI.Apply(root_prim)
    physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(root_prim)
    physx_rb.CreateEnableCCDAttr().Set(True)
    if velocity is not None:
        rb_api = UsdPhysics.RigidBodyAPI(root_prim)
        rb_api.CreateVelocityAttr().Set(Gf.Vec3f(*velocity))
    sphere_prim = stage.GetPrimAtPath(sphere_path)
    UsdPhysics.CollisionAPI.Apply(sphere_prim)
    mass_api = UsdPhysics.MassAPI.Apply(root_prim)
    mass_api.CreateMassAttr().Set(mass)
    mat_path = f"{path}/physics_material"
    from pxr import UsdShade
    UsdShade.Material.Define(stage, mat_path)
    mat_prim = stage.GetPrimAtPath(mat_path)
    mat_api = UsdPhysics.MaterialAPI.Apply(mat_prim)
    mat_api.CreateRestitutionAttr().Set(restitution)
    mat_api.CreateStaticFrictionAttr().Set(0.5)
    mat_api.CreateDynamicFrictionAttr().Set(0.3)
    binding_api = UsdShade.MaterialBindingAPI.Apply(sphere_prim)
    binding_api.Bind(UsdShade.Material(mat_prim), UsdShade.Tokens.weakerThanDescendants, "physics")
    return root_prim

def spawn_rigid_box(stage, path, position, size=(0.15, 0.15, 2.0), color=(0.6, 0.4, 0.2), is_static=True):
    xform = UsdGeom.Xform.Define(stage, path)
    xform.AddTranslateOp().Set(Gf.Vec3d(*position))
    xform.AddScaleOp().Set(Gf.Vec3d(*size))
    cube_path = f"{path}/cube"
    cube = UsdGeom.Cube.Define(stage, cube_path)
    cube.GetSizeAttr().Set(1.0)
    cube.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
    cube_prim = stage.GetPrimAtPath(cube_path)
    UsdPhysics.CollisionAPI.Apply(cube_prim)
    root_prim = stage.GetPrimAtPath(path)
    UsdPhysics.RigidBodyAPI.Apply(root_prim)
    if is_static:
        UsdPhysics.RigidBodyAPI(root_prim).CreateKinematicEnabledAttr().Set(True)
    return root_prim

def remove_prim(stage, path):
    prim = stage.GetPrimAtPath(path)
    if prim.IsValid():
        stage.RemovePrim(path)

class WindModel:
    def __init__(self, scenario='none', params=None):
        self.scenario = scenario
        p = params or {}
        self.rng = np.random.default_rng(int(time.time()) % 2**32)
        self.A, self.Cd, self.rho = 0.04, 1.28, 1.225
        self.ws = p.get('wind_speed', 5.0)
        self.wd = np.deg2rad(p.get('wind_dir', 0.0))
        self.gs = p.get('gust_start', 10.0)
        self.gd = p.get('gust_duration', 3.0)
        self.ti = p.get('turbulence_intensity', 0.5)
        self.tb = p.get('turbulence_bandwidth', 2.0)
        self._ts = np.zeros(3)

    def _drag(self, w):
        v = np.linalg.norm(w)
        return np.zeros(3) if v < 1e-6 else 0.5*self.rho*v**2*self.Cd*self.A*(w/v)

    def get_force(self, t, dt=0.004):
        d = np.array([np.cos(self.wd), np.sin(self.wd), 0.0])
        if self.scenario == 'wind_constant': return self._drag(self.ws * d)
        elif self.scenario == 'wind_gust':
            if t < self.gs or t > self.gs + self.gd: return np.zeros(3)
            V = (self.ws/2)*(1 - np.cos(np.pi*(t-self.gs)/self.gd))
            return self._drag(V * d)
        elif self.scenario == 'wind_turbulence':
            a = np.exp(-self.tb * dt)
            self._ts = a*self._ts + (1-a)*self.ti*self.ws*self.rng.standard_normal(3)
            return self._drag(self.ws * d + self._ts)
        return np.zeros(3)

class CollisionManager:
    def __init__(self, stage, scenario='none', params=None):
        self.stage = stage; self.scenario = scenario; p = params or {}
        self.rng = np.random.default_rng(42)
        self.spawn_log = []
        self.proj_mass = p.get('proj_mass', 0.3); self.proj_radius = p.get('proj_radius', 0.05)
        self.proj_speed = p.get('proj_speed', 10.0); self.proj_count = p.get('proj_count', 1)
        self.proj_start = p.get('proj_start', 15.0); self.proj_interval = p.get('proj_interval', 2.0)
        self.obs_count = p.get('obs_count', 3); self.obs_height = p.get('obs_height', 8.0)
        self.impact_active = False; self.impact_active_until = 0.0; self.IMPACT_SIGNAL_DURATION = 2.0
        self._sched = []
        if scenario in ('impact_projectile', 'impact_falling'):
            for i in range(self.proj_count):
                self._sched.append({'time': self.proj_start + i * self.proj_interval, 'fired': False, 'index': i})

    def setup_static_obstacles(self, flight_pattern='circle', flight_radius=5.0, flight_alt=5.0):
        if self.scenario != 'impact_obstacle': return
        self.obs_positions = []
        for i in range(self.obs_count):
            angle = 2*np.pi*i/self.obs_count + np.pi/self.obs_count
            r = flight_radius * 0.95 if flight_pattern in ('circle', 'figure8') else self.rng.uniform(2.0, 8.0)
            x, y = r*np.cos(angle), r*np.sin(angle); z = self.obs_height / 2.0
            path = f"/World/obstacle_{i:02d}"
            spawn_rigid_box(self.stage, path, [x, y, z], size=(0.15, 0.15, self.obs_height), color=(0.6, 0.4, 0.2), is_static=True)
            self.spawn_log.append((0.0, path, 'static_obstacle'))
            self.obs_positions.append(np.array([x, y, z])) 

    def update(self, sim_time, drone_pos_enu):
        for e in self._sched:
            if e['fired'] or sim_time < e['time']: continue
            e['fired'] = True; idx = e['index']; path = f"/World/projectile_{idx:03d}"
            if self.scenario == 'impact_projectile':
                angle = self.rng.uniform(0, 2*np.pi); dist = self.rng.uniform(3.0, 5.0); dz = self.rng.uniform(-1.0, 1.0)
                sp = [drone_pos_enu[0] + dist*np.cos(angle), drone_pos_enu[1] + dist*np.sin(angle), drone_pos_enu[2] + dz]
                target_pos = np.array(drone_pos_enu); target_pos[2] += 0.4 
                direction = target_pos - np.array(sp); direction /= np.linalg.norm(direction) + 1e-6
                vel = (self.proj_speed * direction).tolist()
                spawn_rigid_sphere(self.stage, path, sp, radius=self.proj_radius, mass=self.proj_mass, color=(1, 0.2, 0.2), velocity=vel, restitution=0.6)
            elif self.scenario == 'impact_falling':
                xy = self.rng.standard_normal(2) * 0.3; h = self.rng.uniform(2.0, 4.0)
                sp = [drone_pos_enu[0]+xy[0], drone_pos_enu[1]+xy[1], drone_pos_enu[2]+h]
                m = self.proj_mass * self.rng.uniform(0.5, 2.0); r = self.proj_radius * (m/self.proj_mass)**(1/3)
                spawn_rigid_sphere(self.stage, path, sp, radius=r, mass=m, color=(0.4, 0.3, 0.2), velocity=[float(xy[0]*0.5), float(xy[1]*0.5), 0.0], restitution=0.3)
            self.spawn_log.append((sim_time, path, self.scenario))
            self.impact_active = True; self.impact_active_until = sim_time + self.IMPACT_SIGNAL_DURATION

        if self.scenario == 'impact_obstacle' and hasattr(self, 'obs_positions'):
            p_drone = np.array(drone_pos_enu)
            for op in self.obs_positions:
                if np.linalg.norm(p_drone[:2] - op[:2]) < 0.29:
                    self.impact_active = True; self.impact_active_until = sim_time + 0.4

        if sim_time > self.impact_active_until: self.impact_active = False

    def cleanup(self, sim_time, max_age=10.0):
        for t, path, ptype in self.spawn_log:
            if ptype != 'static_obstacle' and sim_time - t > max_age: remove_prim(self.stage, path)

    def get_impact_force_signal(self): return np.array([1.0, 0.0, 0.0]) if self.impact_active else np.zeros(3)

# ======================================================================
#  메인 앱
# ======================================================================
class PegasusApp:
    def __init__(self, args):
        self.args = args
        self.sim_time = 0.0
        self.physics_dt = 1.0 / 250.0

        # ★ 제어 타이머 (10Hz, 50Hz 비동기 발행용)
        self.last_gps_time = 0.0
        self.last_obs_time = 0.0
        
        # --- 사이버 공격 상태 변수 ---
        self.sensor_attack_active = False
        self.sensor_atk_type = 'none'
        self.sensor_intensity = 0.0
        self.sensor_ramp_accum = np.zeros(3)
        
        self.hijack_attack_active = False
        self.hijack_atk_type = 'none'
        self.hijack_intensity = 0.0
        
        # GPS 생성을 위한 지구 물리 상수 및 임의의 Home 설정 (취리히)
        self.home_lat = 47.397742
        self.home_lon = 8.545594
        self.home_alt = 488.0  # MSL
        self.earth_radius = 6371000.0

        rclpy.init()
        self.ros_node = Node('ground_truth_publisher')
        self.gt_pub = self.ros_node.create_publisher(Odometry, '/gt/odometry', 10)
        self.dist_pub = self.ros_node.create_publisher(Vector3Stamped, '/disturbance/force', 10)
        
        # 공격 명령 수신 토픽
        self.attack_sub = self.ros_node.create_subscription(String, '/attack_config', self._cb_attack_config, 10)

        # 다이렉트 센서 토픽 
        self.pub_gps = self.ros_node.create_publisher(SensorGps, '/sim/sensor_gps', 10)
        self.pub_baro = self.ros_node.create_publisher(VehicleAirData, '/sim/sensor_baro', 10)
        self.pub_dist = self.ros_node.create_publisher(DistanceSensor, '/sim/distance_sensor', 10)
        self.pub_flow = self.ros_node.create_publisher(SensorOpticalFlow, '/sim/sensor_optical_flow', 10)

        self.timeline = omni.timeline.get_timeline_interface()
        self.pg = PegasusInterface()
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world
        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Flat Plane"])

        config_multirotor = MultirotorConfig()
        
        mavlink_config = PX4MavlinkBackendConfig({
            "vehicle_id": 0, "px4_autolaunch": True,
            "px4_dir": self.pg.px4_path,
            "px4_vehicle_model": self.pg.px4_default_airframe})
        config_multirotor.backends = [PX4MavlinkBackend(mavlink_config)]

        self.vehicle = Multirotor(
            "/World/quadrotor", ROBOTS['Iris'], 0,
            [0.0, 0.0, 0.07], Rotation.from_euler("XYZ", [0,0,0], degrees=True).as_quat(),
            config=config_multirotor)

        # ★ 액추에이터 하이재킹 가로채기 적용
 
        self.world.reset()
        self.stage = omni.usd.get_context().get_stage()

        is_wind = args.scenario.startswith('wind')
        self.wind = WindModel(args.scenario if is_wind else 'none', {k: getattr(args, k.replace('-','_'), None) for k in ['wind_speed','wind_dir','gust_start','gust_duration','turbulence_intensity']})

        is_impact = args.scenario.startswith('impact')
        self.collision = CollisionManager(self.stage, args.scenario if is_impact else 'none', {k: getattr(args, k.replace('-','_'), None) for k in ['proj_mass','proj_radius','proj_speed','proj_count', 'proj_start','proj_interval','obs_count','obs_height']})
        if args.scenario == 'impact_obstacle': self.collision.setup_static_obstacles(args.flight_pattern, args.flight_radius, args.flight_alt)

        self.body_view = None
        self._setup_wind_body()
        self.stop_sim = False

    def _cb_attack_config(self, msg):
        """fly_and_record로부터 공격 ON/OFF 명령 수신"""
        try:
            cfg = json.loads(msg.data)
            if cfg['target'] == 'sensor':
                self.sensor_attack_active = cfg['active']
                self.sensor_atk_type = cfg['type']
                self.sensor_intensity = float(cfg['intensity'])
                if not self.sensor_attack_active: self.sensor_ramp_accum = np.zeros(3)
                carb.log_warn(f"[HACKER] Sensor Attack: {cfg}")
            elif cfg['target'] == 'hijack':
                self.hijack_attack_active = cfg['active']
                self.hijack_atk_type = cfg['type']
                self.hijack_intensity = float(cfg['intensity'])
                carb.log_warn(f"[HACKER] Hijack Attack: {cfg}")
        except Exception as e:
            carb.log_error(f"Failed to parse attack config: {e}")

    
    def _setup_wind_body(self):
        for path in ["/World/quadrotor/body", "/World/quadrotor"]:
            prim = self.stage.GetPrimAtPath(path)
            if prim.IsValid() and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                try:
                    self.body_view = RigidPrimView(prim_paths_expr=path, name="wind_body")
                    self.world.scene.add(self.body_view)
                    self.body_view.initialize()
                except Exception: pass
                break

    def run(self):
        self.timeline.play()

        while simulation_app.is_running() and not self.stop_sim:

            # 1. 바람 외란 계산
            wf = self.wind.get_force(self.sim_time, self.physics_dt)

            attack_force = np.zeros(3)
            attack_torque = np.zeros(3)
            
            if self.hijack_attack_active:
                # 강도 증폭: 0.15 입력을 -> 15.0의 거대한 물리적 힘/토크로 변환
                mag = self.hijack_intensity * 100 
                
                if self.hijack_atk_type == 'override_thrust':
                    attack_force[2] = -mag  # 모터 꺼짐 모사 (강제로 밑으로 끌어내림)
                elif self.hijack_atk_type == 'override_roll':
                    attack_torque[0] = mag * 0.8  # 좌우로 강하게 뒤집기
                elif self.hijack_atk_type == 'override_yaw':
                    attack_torque[2] = mag * 0.8  # 팽이처럼 회전시키기
                elif 'override' in self.hijack_atk_type or self.hijack_atk_type == 'override_all':
                    attack_torque[0] = mag * 0.4  # Roll 회전
                    attack_torque[1] = -mag * 0.4 # Pitch 뒤집기
                    attack_force[2] = -mag * 0.5  # 고도 상실

            # 2. 바람(외란)과 해킹 물리력 합산
            total_force = wf + attack_force
            
            if self.body_view:
                try:
                    # Isaac Sim API: 힘(Force)과 회전력(Torque)을 기체 중심에 직접 타격
                    forces = np.array([total_force], dtype=np.float32)
                    torques = np.array([attack_torque], dtype=np.float32)
                    self.body_view.apply_forces_and_torques_at_pos(forces=forces, torques=torques, is_global=True)
                except Exception:
                    # 구버전 API 호환용 (힘만 적용)
                    if np.linalg.norm(total_force) > 1e-6:
                        self.body_view.apply_forces(np.array([total_force], dtype=np.float32), is_global=True)

            dp = self.vehicle.state.position
            drone_pos = [float(dp[0]), float(dp[1]), float(dp[2])]
            self.collision.update(self.sim_time, drone_pos)
            if int(self.sim_time*10) % 100 == 0: self.collision.cleanup(self.sim_time)

            self.world.step(render=True)
            self.sim_time += self.physics_dt

            # ==========================================================
            # 250Hz - Ground Truth Odometry 기록용 (평가용으로 유지)
            # ==========================================================
            state = self.vehicle.state
            msg = Odometry()
            msg.header.stamp = self.ros_node.get_clock().now().to_msg()
            msg.header.frame_id = "world"
            msg.pose.pose.position.x = float(state.position[0])
            msg.pose.pose.position.y = float(state.position[1])
            msg.pose.pose.position.z = float(state.position[2])
            msg.pose.pose.orientation.x = float(state.attitude[0])
            msg.pose.pose.orientation.y = float(state.attitude[1])
            msg.pose.pose.orientation.z = float(state.attitude[2])
            msg.pose.pose.orientation.w = float(state.attitude[3])
            msg.twist.twist.linear.x = float(state.linear_velocity[0])
            msg.twist.twist.linear.y = float(state.linear_velocity[1])
            msg.twist.twist.linear.z = float(state.linear_velocity[2])
            msg.twist.twist.angular.x = float(state.angular_velocity[0])
            msg.twist.twist.angular.y = float(state.angular_velocity[1])
            msg.twist.twist.angular.z = float(state.angular_velocity[2])
            self.gt_pub.publish(msg)

            timestamp_us = int(self.sim_time * 1e6)

            # ==========================================================
            # ★ 50Hz Update: Optical Flow, Baro, Distance 센서 
            # ==========================================================
            if self.sim_time - self.last_obs_time >= 0.02:
                dt_obs = self.sim_time - self.last_obs_time
                current_z = float(drone_pos[2])
                
                # 1. 기압계 (VehicleAirData) - 고도 변화에 느린 편향(Drift) 추가
                msg_baro = VehicleAirData()
                msg_baro.timestamp = timestamp_us
                baro_drift = math.sin(self.sim_time * 0.05) * 0.5 
                msg_baro.baro_alt_meter = float(self.home_alt + current_z + baro_drift + np.random.normal(0, 0.05))
                self.pub_baro.publish(msg_baro)

                # 2. LiDAR 거리계 
                msg_dist = DistanceSensor()
                msg_dist.timestamp = timestamp_us
                safe_distance = max(0.01, float(current_z + np.random.normal(0, 0.02)))
                msg_dist.current_distance = safe_distance
                self.pub_dist.publish(msg_dist)

                # 3. Optical Flow (★ 선속도 Body 변환 + 각속도 물리 모사 + 자이로 동봉)
                msg_flow = SensorOpticalFlow()
                msg_flow.timestamp = timestamp_us
                dt_obs = 0.02 
                msg_flow.integration_timespan_us = int(dt_obs * 1e6) 
                msg_flow.quality = 255
                msg_flow.distance_m = safe_distance
                msg_flow.distance_available = True
                
                if safe_distance > 0.1: 
                    # 1) World 선속도 및 각속도 추출 (ENU 좌표계)
                    v_world = np.array([state.linear_velocity[0], state.linear_velocity[1], state.linear_velocity[2]])
                    w_world = np.array([state.angular_velocity[0], state.angular_velocity[1], state.angular_velocity[2]])
                    
                    # 2) World -> Body(FLU) 역회전 변환
                    rot = Rotation.from_quat([state.attitude[0], state.attitude[1], state.attitude[2], state.attitude[3]])
                    v_body_flu = rot.inv().apply(v_world)
                    w_body_flu = rot.inv().apply(w_world)
                    
                    # 3) FLU -> FRD (항공 표준: Front, Right, Down) 변환
                    v_x, v_y = v_body_flu[0], -v_body_flu[1]
                    w_x, w_y = w_body_flu[0], -w_body_flu[1]
                    
                    # 4) 센서 노이즈 주입
                    v_x += np.random.normal(0, 0.05)
                    v_y += np.random.normal(0, 0.05)
                    w_x += np.random.normal(0, 0.01)
                    w_y += np.random.normal(0, 0.01)
                    
                    # 5) 픽셀 이동량 계산 (선속도 이동량 + 각속도에 의한 착시 이동량)
                    flow_rate_x = (v_x / safe_distance) - w_y
                    flow_rate_y = (v_y / safe_distance) + w_x
                    
                    # ★ 수정: physics_dt(0.004)가 아닌 dt_obs(0.02)를 곱해 누적 이동량을 구합니다.
                    msg_flow.pixel_flow[0] = float(flow_rate_x * dt_obs)
                    msg_flow.pixel_flow[1] = float(flow_rate_y * dt_obs)
                    
                    msg_flow.delta_angle[0] = float(w_x * dt_obs)
                    msg_flow.delta_angle[1] = float(w_y * dt_obs)
                    msg_flow.delta_angle[2] = 0.0
                    msg_flow.delta_angle_available = True

                else:
                    msg_flow.pixel_flow[0] = 0.0
                    msg_flow.pixel_flow[1] = 0.0
                    msg_flow.delta_angle[0] = 0.0
                    msg_flow.delta_angle[1] = 0.0
                    msg_flow.delta_angle[2] = 0.0
                    msg_flow.delta_angle_available = False
                    
                self.pub_flow.publish(msg_flow)
                    
                self.last_obs_time = self.sim_time

                # ★ 10Hz Update: GPS (SensorGps)
    # ==========================================================
            if self.sim_time - self.last_gps_time >= 0.1:
                msg_gps = SensorGps()
                msg_gps.timestamp = timestamp_us
    
                gps_noise_n = np.random.normal(0, 0.3)
                gps_noise_e = np.random.normal(0, 0.3)
                gps_noise_alt = np.random.normal(0, 0.6)
    
                lat_rad = math.radians(self.home_lat)
                lat_offset_deg = math.degrees((float(dp[1]) + gps_noise_n) / self.earth_radius)
                lon_offset_deg = math.degrees((float(dp[0]) + gps_noise_e) / (self.earth_radius * math.cos(lat_rad)))
    
                # latitude_deg / longitude_deg / altitude_msl_m (float64/32) 사용
                msg_gps.latitude_deg = float(self.home_lat + lat_offset_deg)
                msg_gps.longitude_deg = float(self.home_lon + lon_offset_deg)
                msg_gps.altitude_msl_m = float(self.home_alt + float(dp[2]) + gps_noise_alt)
    
                # 속도 데이터
                msg_gps.vel_n_m_s = float(state.linear_velocity[1] + np.random.normal(0, 0.1))
                msg_gps.vel_e_m_s = float(state.linear_velocity[0] + np.random.normal(0, 0.1))
                msg_gps.vel_d_m_s = float(-state.linear_velocity[2] + np.random.normal(0, 0.1))
                msg_gps.vel_m_s = math.sqrt(msg_gps.vel_n_m_s**2 + msg_gps.vel_e_m_s**2 + msg_gps.vel_d_m_s**2)
    
                msg_gps.eph = 0.5 
                msg_gps.epv = 0.8 
                msg_gps.satellites_used = 12
                msg_gps.fix_type = 3 # 3D fix
    
                # ★ GPS 센서 공격 주입 (퍼블리시 직전 값 변조)
                if self.sensor_attack_active:
                    m_to_deg = 1.0 / 111111.0
                    if self.sensor_atk_type == 'fdi_constant':
                        msg_gps.latitude_deg += self.sensor_intensity * m_to_deg
                        msg_gps.longitude_deg += self.sensor_intensity * m_to_deg
                        msg_gps.vel_n_m_s += self.sensor_intensity * 0.1
                        msg_gps.vel_e_m_s += self.sensor_intensity * 0.1
                    elif self.sensor_atk_type == 'fdi_ramp':
                        self.sensor_ramp_accum += self.sensor_intensity * self.physics_dt
                        msg_gps.latitude_deg += self.sensor_ramp_accum[0] * m_to_deg
                        msg_gps.longitude_deg += self.sensor_ramp_accum[1] * m_to_deg
                        msg_gps.vel_n_m_s += self.sensor_intensity
                        msg_gps.vel_e_m_s += self.sensor_intensity
                    elif self.sensor_atk_type == 'jamming':
                        msg_gps.latitude_deg += np.random.normal(0, self.sensor_intensity) * m_to_deg
                        msg_gps.longitude_deg += np.random.normal(0, self.sensor_intensity) * m_to_deg
                        msg_gps.vel_n_m_s += np.random.normal(0, self.sensor_intensity * 0.5)
                        msg_gps.vel_e_m_s += np.random.normal(0, self.sensor_intensity * 0.5)

                self.pub_gps.publish(msg_gps)
                self.last_gps_time = self.sim_time

            # ==========================================================

            dm = Vector3Stamped()
            dm.header.stamp = msg.header.stamp
            dm.header.frame_id = "world_enu"
            impact_signal = self.collision.get_impact_force_signal() if self.collision else np.zeros(3)
            combined = wf + impact_signal
            dm.vector.x, dm.vector.y, dm.vector.z = float(combined[0]), float(combined[1]), float(combined[2])
            self.dist_pub.publish(dm)

            rclpy.spin_once(self.ros_node, timeout_sec=0)

        carb.log_warn("PegasusApp closing.")
        self.timeline.stop()
        simulation_app.close()
        rclpy.shutdown()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', default='none', choices=['none','wind_constant','wind_gust','wind_turbulence', 'impact_projectile','impact_obstacle','impact_falling'])
    parser.add_argument('--wind-speed', type=float, default=5.0)
    parser.add_argument('--wind-dir', type=float, default=0.0)
    parser.add_argument('--gust-start', type=float, default=10.0)
    parser.add_argument('--gust-duration', type=float, default=3.0)
    parser.add_argument('--turbulence-intensity', type=float, default=0.5)
    parser.add_argument('--proj-mass', type=float, default=0.3)
    parser.add_argument('--proj-radius', type=float, default=0.05)
    parser.add_argument('--proj-speed', type=float, default=10.0)
    parser.add_argument('--proj-count', type=int, default=1)
    parser.add_argument('--proj-start', type=float, default=15.0)
    parser.add_argument('--proj-interval', type=float, default=2.0)
    parser.add_argument('--obs-count', type=int, default=3)
    parser.add_argument('--obs-height', type=float, default=8.0)
    parser.add_argument('--flight-pattern', default='circle')
    parser.add_argument('--flight-radius', type=float, default=5.0)
    parser.add_argument('--flight-alt', type=float, default=5.0)
    args = parser.parse_args()
    PegasusApp(args).run()

if __name__ == "__main__":
    main()
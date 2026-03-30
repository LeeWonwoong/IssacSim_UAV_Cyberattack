"""
config.py — 드론 자율 복원 제어 통합 설정
===========================================
RL, 드론 물리, 보상, 공격/외란 풀, 커리큘럼, 시나리오 샘플러
모든 하이퍼파라미터의 중앙 통제소.
"""
import os
import random
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Config:
    # ══════════════════════════════════════════════════════════
    #  시스템
    # ══════════════════════════════════════════════════════════
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    outdir: str = "./results"
    headless: bool = True           # True: 렌더링 OFF → GPU 학습 집중
    use_tf32: bool = True           # TF32 가속 (FP32 연산을 TF32 정밀도로)
    use_compile: bool = True       # torch.compile (실시간 환경에서 충돌 시 False)

    # ══════════════════════════════════════════════════════════
    #  에피소드 구조
    # ══════════════════════════════════════════════════════════
    max_episodes: int = 500
    episode_max_steps: int = 500    # 10Hz × 50초
    warmup_seconds: float = 3.0     # 이륙 후 안정화 대기 (UKF 워밍업 겸용)
    attack_start_range: Tuple[int, int] = (30, 80)  # 공격 시작 스텝 범위 (3~8초)
    attack_ramp_duration: float = 0.1  # 공격 강도 ramp 시간 (초)
    
    # ── 디버깅 ──
    log_interval: int = 10          # N 스텝마다 상세 로그 출력 (10Hz 기준 1초)

    # ══════════════════════════════════════════════════════════
    #  드론 물리 (보상용)
    # ══════════════════════════════════════════════════════════
    natural_lag: float = 0.6        # (m) 정상 비행 위상 지연 (페널티 면제)
    max_error: float = 2.0          # (m) 궤도 이탈 판정 한계선
    min_altitude: float = -0.5      # (m) 추락 판정 고도 (NED z 기준)
    flight_altitude: float = 5.0    # (m) 기본 비행 고도
    flight_radius: float = 5.0      # (m) 원 궤도 반지름
    flight_omega: float = 0.5       # (rad/s) 원 궤도 각속도

    # ══════════════════════════════════════════════════════════
    #  비행 패턴 풀
    # ══════════════════════════════════════════════════════════
    flight_patterns: List[str] = field(default_factory=lambda: [
        'hover', 'circle', 'figure8', 'waypoint', 'aggressive'
    ])

    # ══════════════════════════════════════════════════════════
    #  액추에이터 공격 풀 (Additive — PhysX 외력/토크 주입)
    # ══════════════════════════════════════════════════════════
    attack_enabled: bool = True
    attack_types: List[str] = field(default_factory=lambda: [
        'loe_thrust',       # 추력 저하 → 아래로 끌어내림
        'loe_roll',         # Roll 축 토크 불균형
        'loe_pitch',        # Pitch 축 토크 불균형
        'loe_yaw',          # Yaw 축 토크 불균형
        'loe_combined',     # Roll + Pitch + 추력 복합 결함
    ])
    prob_no_attack: float = 0.15    # 15% 확률로 공격 없는 에피소드

    # ══════════════════════════════════════════════════════════
    #  커리큘럼 학습 (하한 고정, 상한만 점진 확장)
    #  ───────────────────────────────────────────────────────
    #  loe_combined 기준 임계값: intensity=0.13 생존, 0.14 즉사
    #  → 상한을 0.13으로 제한하여 학습 가능한 범위 보장
    #  → 하한은 항상 0.05 유지하여 약한 공격 탐지 능력도 학습
    # ══════════════════════════════════════════════════════════
    curriculum_enabled: bool = True
    curriculum_fixed_min: float = 0.05      # 항상 고정된 하한 (약한 공격도 계속 겪음)
    curriculum_start_max: float = 0.08      # 초반 상한 (확실히 생존 가능)
    curriculum_end_max: float = 0.13        # 후반 상한 (즉사 직전, 호버링으로 간신히 버팀)
    curriculum_warmup_episodes: int = 50    # 이 에피소드까지 초반 상한
    curriculum_full_episodes: int = 300     # 이 에피소드부터 후반 상한 (사이는 선형 보간)

    # ══════════════════════════════════════════════════════════
    #  환경 외란 풀
    # ══════════════════════════════════════════════════════════
    disturbance_enabled: bool = True
    disturbance_types: List[str] = field(default_factory=lambda: [
        'none', 'wind_constant', 'wind_gust', 'wind_turbulence'
    ])
    wind_speed_range: Tuple[float, float] = (2.0, 8.0)

    # ══════════════════════════════════════════════════════════
    #  RL 하이퍼파라미터
    # ══════════════════════════════════════════════════════════
    window_size: int = 4
    dimS: int = 8                   # window_size × 2 (vel_nis, gyr_nis)
    num_actions: int = 2            # 0=궤도 추종, 1=강제 호버링
    gamma: float = 0.94
    batch_size: int = 128
    buffer_size: int = 10000

    # ── 탐험 (Epsilon-Greedy) ──
    eps_start: float = 0.99
    eps_end: float = 0.001
    eps_decay_steps: int = 5000

    # ══════════════════════════════════════════════════════════
    #  D3QN 네트워크 구조
    # ══════════════════════════════════════════════════════════
    shared_layers: List[int] = field(default_factory=lambda: [16, 16])
    value_layers: List[int] = field(default_factory=lambda: [4])
    advantage_layers: List[int] = field(default_factory=lambda: [4])

    # ══════════════════════════════════════════════════════════
    #  SRRHUIF-ND (필터 뇌)
    # ══════════════════════════════════════════════════════════
    use_spas: bool = True           # Sigma Point Action Selection
    tau_srrhuif: float = 0.005      # Soft update 비율
    N_horizon: int = 5              # Receding Horizon 윈도우 크기
    #   매 10Hz 스텝마다 배치를 1개 샘플링 → deque(maxlen=N)에 추가
    #   N개가 모이면 h=0(비정보적 초기조건)~h=N-1까지 순차 필터 업데이트
    #   이후 매 스텝: 가장 오래된 배치 탈락, 새 배치 진입 → 슬라이딩 윈도우
    q_std: float = 3e-3
    r_std: float = 1.8
    alpha: float = 0.8
    beta: float = 2.0
    kappa: float = 0.0

    # ── Adaptive P (Time Decay × Reward Feedback) ──
    p_init_min: float = 0.001
    p_init_max: float = 0.025
    adaptive_window: int = 25
    adaptive_p_decay_steps: int = 100   # 이 에피소드 간격마다 상한 감쇠
    adaptive_p_decay_factor: float = 0.5  # 감쇠 비율 (StepLR 차용)
    adaptive_p_min_ceiling: float = 0.3   # 보상률 ceiling 클램핑 하한

    # ══════════════════════════════════════════════════════════
    #  평가 (고정 시나리오 — SRRHUIF vs Adam 공정 비교용)
    # ══════════════════════════════════════════════════════════
    eval_interval: int = 20             # N 학습 에피소드마다 평가 실행
    eval_scenarios: List[dict] = field(default_factory=lambda: [
        # 1. 기준선 — 무공격
        {'pattern': 'circle', 'attack_type': 'none',
         'attack_intensity': 0.0, 'attack_start_step': 0,
         'disturbance_type': 'none', 'wind_speed': 0.0},
        # 2. 약한 공격 — 탐지 능력 테스트
        {'pattern': 'circle', 'attack_type': 'loe_combined',
         'attack_intensity': 0.06, 'attack_start_step': 50,
         'disturbance_type': 'none', 'wind_speed': 0.0},
        # 3. 중간 공격 — 탐지 + 생존
        {'pattern': 'figure8', 'attack_type': 'loe_combined',
         'attack_intensity': 0.10, 'attack_start_step': 50,
         'disturbance_type': 'none', 'wind_speed': 0.0},
        # 4. 강한 공격 — 생존 한계
        {'pattern': 'hover', 'attack_type': 'loe_combined',
         'attack_intensity': 0.13, 'attack_start_step': 50,
         'disturbance_type': 'none', 'wind_speed': 0.0},
        # 5. 공격 + 외란 — 최악 조건
        {'pattern': 'circle', 'attack_type': 'loe_combined',
         'attack_intensity': 0.10, 'attack_start_step': 50,
         'disturbance_type': 'wind_turbulence', 'wind_speed': 5.0},
    ])

    def __post_init__(self):
        self.r_inv_sqrt = 1.0 / self.r_std
        self.r_inv = 1.0 / (self.r_std ** 2)
        self.dimS = self.window_size * 2
        os.makedirs(self.outdir, exist_ok=True)


# ══════════════════════════════════════════════════════════════
#  커리큘럼 스케줄러
# ══════════════════════════════════════════════════════════════
def get_curriculum_intensity(episode: int, cfg: Config) -> Tuple[float, float]:
    """
    에피소드 번호 → 공격 강도 범위 (하한 고정, 상한만 선형 확장)

    초반 (Ep ≤ 50):  [0.05, 0.08]  — 확실히 생존, 탐지 기초 학습
    중반 (50~300):    [0.05, 0.08→0.13]  — 상한이 점진적으로 올라감
    후반 (Ep ≥ 300):  [0.05, 0.13]  — 약한 공격~즉사 직전까지 전 범위

    Returns:
        (lo, hi): 공격 강도 샘플링 범위
    """
    if not cfg.curriculum_enabled:
        return (cfg.curriculum_fixed_min, cfg.curriculum_end_max)

    if episode <= cfg.curriculum_warmup_episodes:
        progress = 0.0
    elif episode >= cfg.curriculum_full_episodes:
        progress = 1.0
    else:
        progress = (episode - cfg.curriculum_warmup_episodes) / \
                   (cfg.curriculum_full_episodes - cfg.curriculum_warmup_episodes)

    # 하한: 항상 고정
    lo = cfg.curriculum_fixed_min

    # 상한: 선형 보간 (start_max → end_max)
    hi = cfg.curriculum_start_max + progress * \
         (cfg.curriculum_end_max - cfg.curriculum_start_max)

    return (lo, hi)


# ══════════════════════════════════════════════════════════════
#  시나리오 샘플러
# ══════════════════════════════════════════════════════════════
def sample_episode_scenario(episode: int, cfg: Config) -> dict:
    """매 에피소드 시작 시 호출 → 패턴/공격/외란/공격시점 랜덤 결정"""
    scenario = {
        'pattern': random.choice(cfg.flight_patterns),
        'attack_type': 'none',
        'attack_intensity': 0.0,
        'attack_start_step': 0,         # 공격 시작 10Hz 스텝
        'disturbance_type': 'none',
        'wind_speed': 0.0,
    }

    # 공격 결정 (커리큘럼 적용)
    if cfg.attack_enabled and random.random() > cfg.prob_no_attack:
        scenario['attack_type'] = random.choice(cfg.attack_types)
        lo, hi = get_curriculum_intensity(episode, cfg)
        scenario['attack_intensity'] = random.uniform(lo, hi)
        # ★ 공격 시작 시점 랜덤 (범위 내)
        scenario['attack_start_step'] = random.randint(*cfg.attack_start_range)

    # 외란 결정
    if cfg.disturbance_enabled:
        scenario['disturbance_type'] = random.choice(cfg.disturbance_types)
        if scenario['disturbance_type'] != 'none':
            scenario['wind_speed'] = random.uniform(*cfg.wind_speed_range)

    return scenario


# ══════════════════════════════════════════════════════════════
#  공격 Ramp 유틸리티
# ══════════════════════════════════════════════════════════════
def compute_attack_ramp(t_since_attack: float, target_intensity: float,
                        ramp_duration: float = 0.1) -> float:
    """공격 시작 후 ramp_duration 동안 선형으로 강도 증가"""
    if ramp_duration <= 0 or t_since_attack >= ramp_duration:
        return target_intensity
    return target_intensity * (t_since_attack / ramp_duration)


# ══════════════════════════════════════════════════════════════
#  공격 힘/토크 변환
# ══════════════════════════════════════════════════════════════
def compute_attack_forces(attack_type: str,
                          intensity: float) -> Tuple[np.ndarray, np.ndarray]:
    """공격 유형 + 정규화 강도 → PhysX 힘(N)/토크(Nm) 벡터"""
    force = np.zeros(3)
    torque = np.zeros(3)
    mag = intensity * 100  # 정규화 강도 → 물리 단위

    if attack_type == 'loe_thrust':
        force[2] = -mag                     # 추력 상실 → Z축 하방
    elif attack_type == 'loe_roll':
        torque[0] = mag * 0.8              # Roll 불균형
    elif attack_type == 'loe_pitch':
        torque[1] = -mag * 0.8             # Pitch 불균형
    elif attack_type == 'loe_yaw':
        torque[2] = mag * 0.8              # Yaw 회전
    elif attack_type == 'loe_combined':
        torque[0] = mag * 0.4             # Roll
        torque[1] = -mag * 0.4            # Pitch
        force[2] = -mag * 0.5             # 고도 상실

    return force, torque

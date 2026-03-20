#!/usr/bin/env python3
"""
calibrate_today_rls.py (Prior-based Recursive Least Squares)
====================================================================
[작동 원리]
1. Offline으로 구한 'calibration_sysld.json'을 초기 지식(Prior)으로 로드합니다.
2. 이륙 직후 수행한 WARMUP 비행 데이터(ep_today_warmup.npz)를 1스텝씩 순회하며
   RLS(망각 계수 lambda=0.99)를 통해 파라미터를 당일 환경에 맞게 미세 조정합니다.
3. 파라미터 오염(Poisoning)을 막기 위해 Base 값의 ±20% 이내로만 변동을 허용(Clamp)합니다.
4. 최종 결과를 'calibration_today.json'으로 저장하여 본 비행(Freeze)에 사용합니다.
====================================================================
"""

import numpy as np
import json, os, argparse
from scipy.signal import butter, filtfilt

DRONE = {
    'mass': 1.5,
    'g':    9.81,
    'Ixx':  0.02912,
    'Iyy':  0.02912,
    'Izz':  0.0552,
}

def apply_zero_phase_filter(data, fs=50):
    nyq = 0.5 * fs
    b, a = butter(4, min(8, 0.8 * nyq) / nyq, btype='low')
    return filtfilt(b, a, data, axis=0)

def rls_update(theta, P, phi, y, lam=0.99):
    """
    단일 스텝 RLS 업데이트 함수
    :param theta: 현재 파라미터 벡터 (N x 1)
    :param P: 현재 공분산 행렬 (N x N)
    :param phi: 입력 레그레서 벡터 (N x 1)
    :param y: 센서 측정값 (Scalar)
    :param lam: 망각 계수 (Forgetting factor, 0.99면 최근 데이터에 가중치)
    """
    phi = phi.reshape(-1, 1)
    denom = lam + phi.T @ P @ phi
    K = (P @ phi) / denom
    theta_new = theta + K @ (y - phi.T @ theta)
    P_new = (P - K @ phi.T @ P) / lam
    return theta_new, P_new

def run_rls_fine_tuning(base_calib, warmup_file, lam=0.99, cov_init=0.1):
    print(f"[*] Base Calibration 로드 완료. (초기 Covariance: {cov_init}, Lambda: {lam})")
    print(f"[*] WARMUP 데이터 분석 중: {warmup_file}")
    
    data = np.load(warmup_file, allow_pickle=True)
    dt = float(data['dt'])
    
    # 데이터 전처리 (노이즈 필터링)
    acc = apply_zero_phase_filter(data['accelerometer'])
    gyro = apply_zero_phase_filter(data['gyro'])
    vel = apply_zero_phase_filter(data['velocity'])
    ang_acc = apply_zero_phase_filter(np.gradient(gyro, dt, axis=0))
    thrust = data['thrust']
    torque = data['torque']
    
    m, g = DRONE['mass'], DRONE['g']
    I = np.array([DRONE['Ixx'], DRONE['Iyy'], DRONE['Izz']])
    
    # 1. 초기 파라미터(Theta) 및 공분산(P) 세팅 (Prior Knowledge)
    # Z축 (Thrust & Z-Drag)
    theta_z = np.array([[base_calib['C_thrust']], [base_calib['drag'][2]]])
    P_z = np.eye(2) * cov_init
    
    # X, Y Drag
    theta_x = np.array([[base_calib['drag'][0]]])
    theta_y = np.array([[base_calib['drag'][1]]])
    P_x, P_y = np.eye(1) * cov_init, np.eye(1) * cov_init
    
    # Torque
    theta_tx = np.array([[base_calib['C_torque_xy']]])
    theta_ty = np.array([[base_calib['C_torque_xy']]])
    theta_tz = np.array([[base_calib['C_torque_z']]])
    P_tx, P_ty, P_tz = np.eye(1) * cov_init, np.eye(1) * cov_init, np.eye(1) * cov_init

    # 2. RLS 루프 (1스텝씩 업데이트)
    N = len(acc)
    for k in range(N):
        # --- [1] Z축 업데이트 (Thrust & Z-Drag) ---
        f_net_ned = m * acc[k]
        y_z = f_net_ned[2] - m * g
        phi_z = np.array([-thrust[k, 2], -vel[k, 2]])
        theta_z, P_z = rls_update(theta_z, P_z, phi_z, y_z, lam)
        
        # --- [2] X, Y Drag 업데이트 ---
        y_x, y_y = f_net_ned[0], f_net_ned[1]
        phi_x, phi_y = np.array([-vel[k, 0]]), np.array([-vel[k, 1]])
        theta_x, P_x = rls_update(theta_x, P_x, phi_x, y_x, lam)
        theta_y, P_y = rls_update(theta_y, P_y, phi_y, y_y, lam)
        
        # --- [3] Torque 업데이트 ---
        p, q, r = gyro[k]
        p_dot, q_dot, r_dot = ang_acc[k]
        tau_net = np.array([
            I[0]*p_dot + (I[2]-I[1])*q*r,
            I[1]*q_dot + (I[0]-I[2])*p*r,
            I[2]*r_dot + (I[1]-I[0])*p*q
        ])
        theta_tx, P_tx = rls_update(theta_tx, P_tx, np.array([torque[k, 0]]), tau_net[0], lam)
        theta_ty, P_ty = rls_update(theta_ty, P_ty, np.array([torque[k, 1]]), tau_net[1], lam)
        theta_tz, P_tz = rls_update(theta_tz, P_tz, np.array([torque[k, 2]]), tau_net[2], lam)

    # 3. 안전장치 (Clamping): Base 값의 ±20% 이내로 제한
    def clamp(val, base_val, margin=0.2):
        return float(np.clip(val, base_val * (1 - margin), base_val * (1 + margin)))

    final_calib = {
        'drone': DRONE,
        'C_thrust': clamp(theta_z[0, 0], base_calib['C_thrust']),
        'C_torque_xy': clamp((theta_tx[0, 0] + theta_ty[0, 0]) / 2, base_calib['C_torque_xy']),
        'C_torque_z': clamp(theta_tz[0, 0], base_calib['C_torque_z']),
        'drag': [
            clamp(theta_x[0, 0], base_calib['drag'][0]),
            clamp(theta_y[0, 0], base_calib['drag'][1]),
            clamp(abs(theta_z[1, 0]), base_calib['drag'][2])
        ]
    }

    # 변화량 리포트
    print("\n[RLS Fine-tuning 결과 (Base -> Today)]")
    print(f" - C_thrust   : {base_calib['C_thrust']:.6f} -> {final_calib['C_thrust']:.6f}")
    print(f" - C_torque_xy: {base_calib['C_torque_xy']:.6f} -> {final_calib['C_torque_xy']:.6f}")
    print(f" - Drag_x     : {base_calib['drag'][0]:.6f} -> {final_calib['drag'][0]:.6f}")
    print(f" - Drag_z     : {base_calib['drag'][2]:.6f} -> {final_calib['drag'][2]:.6f}")

    return final_calib

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default='calibration_sysld.json', help='Offline 기본 캘리브레이션 파일')
    parser.add_argument('--warmup', default='data_raw/ep_today_warmup.npz', help='오늘의 예열 비행 데이터')
    parser.add_argument('--output', default='calibration_today.json', help='저장될 오늘의 캘리브레이션 파일')
    args = parser.parse_args()

    if not os.path.exists(args.base):
        print(f"[!] {args.base} 파일이 없습니다. 먼저 오프라인 LS를 실행하세요.")
        return
    if not os.path.exists(args.warmup):
        print(f"[!] {args.warmup} 파일이 없습니다. 현장 예열 비행 데이터를 준비하세요.")
        return

    base_calib = json.load(open(args.base))
    today_calib = run_rls_fine_tuning(base_calib, args.warmup)

    with open(args.output, 'w') as f:
        json.setdefault('indent', 4)
        json.dump(today_calib, f, indent=4)
    print(f"\n[*] '{args.output}' 저장 완료! 이제 본 비행 방어 시스템에 이 파일을 적용하세요.")

if __name__ == '__main__':
    main()
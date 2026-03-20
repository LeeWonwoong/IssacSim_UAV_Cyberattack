import numpy as np
import json
import glob
import os
from scipy.signal import butter, filtfilt, savgol_filter

# 논문용 물리 파라미터 설정

DRONE = {
    'mass': 1.5,
    'g':    9.81,
    'Ixx':  0.02912,
    'Iyy':  0.02912,
    'Izz':  0.0552,
}
def apply_zero_phase_filter(data, fs=50):
    """지연 없는 영위상 필터링 적용 (IEEE 표준)"""
    nyq = 0.5 * fs
    b, a = butter(4, min(8, 0.8 * nyq) / nyq, btype='low')
    return filtfilt(b, a, data, axis=0)

def run_sysid_ols(files):
    m, g = DRONE['mass'], DRONE['g']
    I = np.array([DRONE['Ixx'], DRONE['Iyy'], DRONE['Izz']])
    
    Y_z, Phi_z = [], []      # Thrust & Z-Drag
    Y_x, Phi_x = [], []      # X Drag
    Y_y, Phi_y = [], []      # Y Drag
    Y_tq, Phi_tq = [], []    # Torque

    for fpath in files:
        data = np.load(fpath, allow_pickle=True)
        dt = float(data['dt'])
        fs = 1.0 / dt
        
        gyro_f = apply_zero_phase_filter(data['gyro'], fs=fs)
        ang_accel = savgol_filter(gyro_f, window_length=11, polyorder=3, deriv=1, delta=dt, axis=0)
        
        euler = data['euler']
        accel_body = data['accelerometer']
        vel_ned = data['velocity']
        thrust_in = data['thrust'][:, 2] # PX4 normalized thrust Z
        torque_in = data['torque']

        for k in range(int(len(euler)*0.1), int(len(euler)*0.9)):
            phi, th, psi = euler[k]
            cp, sp, ct, st, cps, sps = np.cos(phi), np.sin(phi), np.cos(th), np.sin(th), np.cos(psi), np.sin(psi)
            
            # Body to NED Rotation Matrix
            R = np.array([
                [ct*cps,  sp*st*cps - cp*sps,  cp*st*cps + sp*sps],
                [ct*sps,  sp*st*sps + cp*cps,  cp*st*sps - sp*cps],
                [-st,     sp*ct,               cp*ct             ]
            ])
            
            a_ned = R @ accel_body[k]
            
            # [수정됨] IMU 가속도는 이미 중력이 반영된 비력(Specific force)이므로 g를 더할 필요 없음
            Y_z.append(m * a_ned[2])
            Phi_z.append([thrust_in[k], -vel_ned[k, 2]])
            
            Y_tq.append(I * ang_accel[k])
            Phi_tq.append(torque_in[k])

    # 1. Thrust & Z-Drag 도출
    res_z = np.linalg.lstsq(np.array(Phi_z), np.array(Y_z), rcond=None)[0]
    c_thrust, c_drag_z = float(res_z[0]), float(res_z[1])

    # 2. X/Y Drag 도출
    for fpath in files:
        data = np.load(fpath, allow_pickle=True)
        for k in range(int(len(data['euler'])*0.1), int(len(data['euler'])*0.9)):
            phi, th, psi = data['euler'][k]
            cp, sp, ct, st, cps, sps = np.cos(phi), np.sin(phi), np.cos(th), np.sin(th), np.cos(psi), np.sin(psi)
            R = np.array([
                [ct*cps,  sp*st*cps - cp*sps,  cp*st*cps + sp*sps],
                [ct*sps,  sp*st*sps + cp*cps,  cp*st*sps - sp*cps],
                [-st,     sp*ct,               cp*ct             ]
            ])
            
            # 3D 추력 벡터 계산
            f_thr_ned = R @ np.array([0, 0, c_thrust * data['thrust'][k, 2]])
            
            # 알짜힘 = 측정된 전체 힘 - 추력
            a_ned = R @ data['accelerometer'][k]
            f_net_ned = m * a_ned - f_thr_ned
            
            Y_x.append(f_net_ned[0])
            Phi_x.append([-data['velocity'][k, 0]])
            
            Y_y.append(f_net_ned[1])
            Phi_y.append([-data['velocity'][k, 1]])
    
    res_x = np.linalg.lstsq(np.array(Phi_x), np.array(Y_x), rcond=None)[0]
    res_y = np.linalg.lstsq(np.array(Phi_y), np.array(Y_y), rcond=None)[0]
    c_drag_x, c_drag_y = float(res_x[0]), float(res_y[0])

    # 3. Torque 상수 도출
    Phi_tq_arr, Y_tq_arr = np.array(Phi_tq), np.array(Y_tq)
    c_torques = [float(np.linalg.lstsq(Phi_tq_arr[:, i:i+1], Y_tq_arr[:, i], rcond=None)[0][0]) for i in range(3)]

    return c_thrust, c_torques, [c_drag_x, c_drag_y, abs(c_drag_z)]

def main():
    files = sorted(glob.glob('data_raw/ep*.npz'))
    if not files:
        print("[!] data_raw 폴더에 데이터가 없습니다.")
        return

    c_thr, c_tqs, drags = run_sysid_ols(files)
    
    calib = {
        'C_thrust': c_thr,
        'C_torque_xy': (c_tqs[0] + c_tqs[1]) / 2.0,
        'C_torque_z': c_tqs[2],
        'drag': drags,
        'drone': DRONE,
        'note': "SysId via OLS with Zero-phase filtering (Gravity bug fixed)"
    }
    
    with open('calibration.json', 'w') as f:
        json.dump(calib, f, indent=2)
        
    print(f"[*] 시스템 식별(캘리브레이션) 완료!")
    print(f"    - C_thrust: {c_thr:.4f} (정상: 20~50)")
    print(f"    - C_torque: XY={calib['C_torque_xy']:.4f}, Z={c_tqs[2]:.4f}")
    print(f"    - Drag:     X={drags[0]:.4f}, Y={drags[1]:.4f}, Z={drags[2]:.4f} (정상: 양수 0.05~0.5)")

if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
postprocess_Integrated_final.py (No-DOB SOTA Anomaly Detector + Batch Mode)
====================================================================
[주요 업데이트 내역]
1. 모달/필터 튜닝: DOB 완전 제거 (12 States), 순수 Dynamics 기반 탐지기로 전환
2. Q-Tuning & Clipping: 위치 Q=1.0(부채 초기화), 오일러 Q=1e-5(고집 극대화), 0.8rad 안전 클리핑 적용
3. 관측값 재구성: [Pos, Vel, Gyr] 3채널 NIS (상수 C=3.0)
4. 시각화 개선: 오일러 각도(Attitude) 추종 모니터링 플롯 추가
5. Batch Mode 지원: 전체 RL 데이터셋(swrl_final_dataset.npz) 자동 병합
====================================================================
"""

import numpy as np
import os, json, argparse, glob
import matplotlib.pyplot as plt

# ======================================================================
#  공용 유틸리티
# ======================================================================
def load_calibration(path='calibration.json'):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[!] {path} 파일이 없습니다. calibrate_sysld.py를 먼저 실행하세요.")
    with open(path) as f: return json.load(f)

def to_physical_u(thrust, torque, calib):
    N = len(thrust)
    u = np.zeros((N, 4))
    u[:, 0] = abs(thrust[:, 2]) * calib['C_thrust']
    u[:, 1] = torque[:, 0] * calib['C_torque_xy']
    u[:, 2] = torque[:, 1] * calib['C_torque_xy']
    u[:, 3] = torque[:, 2] * calib['C_torque_z']
    return u

def get_contiguous_intervals(mask):
    if mask is None or len(mask) == 0: return []
    arr = np.concatenate(([0], mask, [0]))
    diff = np.diff(arr)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1
    return list(zip(starts, ends))

def compute_nis_scaled(r_sub, Pzz_sub, nz):
    try:
        nis_raw = r_sub @ np.linalg.solve(Pzz_sub, r_sub) / nz
    except np.linalg.LinAlgError:
        nis_raw = 0.0
    
    log_nis = np.log1p(nis_raw)
    nis_scaled = log_nis / (log_nis + 1.0)
    return nis_raw, nis_scaled


# ======================================================================
#  1. Dynamics-Centric UKF (12 states - No DOB SOTA)
# ======================================================================
class DynamicsUKF:
    def __init__(self, dt=0.02, calib=None):
        self.nx, self.nz, self.dt = 12, 12, dt
        d = calib['drone']
        self.m, self.g = d['mass'], d['g']
        self.I = [d['Ixx'], d['Iyy'], d['Izz']]
        self.drag = np.array(calib['drag'])
        n, lam = self.nx, 0.5**2 * self.nx - self.nx
        self.lam = lam
        self.Wm, self.Wc = np.full(2*n+1, 1.0/(2*(n+lam))), np.full(2*n+1, 1.0/(2*(n+lam)))
        self.Wm[0], self.Wc[0] = lam/(n+lam), lam/(n+lam) + (1 - 0.5**2 + 2.0)
        
        # ★ [핵심 튜닝] 위치 부채 초기화(1.0) + 오일러 고집 극대화(1e-5)
        self.Q = np.diag([1e-3]*3 + [1e-4]*3 + [3e-2]*3 + [1e-3]*3)
        self.R = np.diag([1.0]*3 + [1.0]*3 + [0.5] + [0.5]*2 + [1.0]*3)
        self.x, self.P = np.zeros(12), np.eye(12) * 0.1

    def _f(self, x, u):
        s = x.copy(); sdt = 0.005; n_sub = int(self.dt/sdt)
        for _ in range(n_sub):
            phi, th, psi = s[3:6]; vx, vy, vz = s[6:9]; p, q, r = s[9:12]
            
            # ★ [안전 장치] 짐벌 락(특이점) 및 채터링 폭발을 막기 위한 0.8rad(45도) 클리핑
            limit = 0.8 
            phi = np.clip(phi, -limit, limit)
            th = np.clip(th, -limit, limit)
            s[3:6] = [phi, th, psi]
            
            cp, sp, ct, st, tt, cps, sps = np.cos(phi), np.sin(phi), np.cos(th), np.sin(th), np.tan(th), np.cos(psi), np.sin(psi)
            
            vbx = cps*ct*vx + sps*ct*vy - st*vz
            vby = (cps*st*sp-sps*cp)*vx + (sps*st*sp+cps*cp)*vy + ct*sp*vz
            vbz = (cps*st*cp+sps*sp)*vx + (sps*st*cp-cps*sp)*vy + ct*cp*vz
            fd = np.array([-self.drag[0]*vbx, -self.drag[1]*vby, -self.drag[2]*vbz])
            
            f_thrust_body = np.array([0, 0, -u[0]])
            f_total_body = f_thrust_body + fd
            R_mat = np.array([
                [ct*cps, sp*st*cps-cp*sps, cp*st*cps+sp*sps],
                [ct*sps, sp*st*sps+cp*cps, cp*st*sps-sp*cps],
                [-st, sp*ct, cp*ct]
            ])
            f_total_ned = R_mat @ f_total_body
            accel_ned = (f_total_ned / self.m) + np.array([0, 0, self.g])
            
            s[0:3] += np.array([vx, vy, vz])*sdt
            s[3:6] += np.array([p + sp*tt*q + cp*tt*r, cp*q - sp*r, sp/(ct+1e-10)*q + cp/(ct+1e-10)*r])*sdt
            s[6:9] += accel_ned * sdt
            s[9] += ((self.I[1]-self.I[2])/self.I[0]*q*r + u[1]/self.I[0])*sdt
            s[10] += ((self.I[2]-self.I[0])/self.I[1]*p*r + u[2]/self.I[1])*sdt
            s[11] += ((self.I[0]-self.I[1])/self.I[2]*p*q + u[3]/self.I[2])*sdt
            
        return s

    def _h(self, x):
        pos, vel, gyro = x[0:3], x[6:9], x[9:12]
        cp, sp, ct, st = np.cos(x[3]), np.sin(x[3]), np.cos(x[4]), np.sin(x[4])
        cps, sps = np.cos(x[5]), np.sin(x[5])
        R_w_b = np.array([
            [ct*cps, ct*sps, -st],
            [sp*st*cps - cp*sps, sp*st*sps + cp*cps, sp*ct],
            [cp*st*cps + sp*sps, cp*st*sps - sp*cps, cp*ct]
        ])
        v_body = R_w_b @ vel
        return np.array([pos[0], pos[1], pos[2], vel[0], vel[1], vel[2], pos[2], v_body[0], v_body[1], gyro[0], gyro[1], gyro[2]])

    def step(self, z, u):
        n = self.nx
        self.P = 0.5 * (self.P + self.P.T)
        try:
            S_root = np.linalg.cholesky((n + self.lam) * self.P + 1e-6*np.eye(n))
        except np.linalg.LinAlgError:
            self.P += np.eye(n) * 1e-3
            S_root = np.linalg.cholesky((n + self.lam) * self.P + 1e-6*np.eye(n))
            
        pts = np.vstack([self.x, self.x + S_root.T, self.x - S_root.T])
        pts_f = np.array([self._f(p, u) for p in pts])
        x_bar = self.Wm @ pts_f
        P_bar = self.Q + sum(self.Wc[i] * np.outer(pts_f[i]-x_bar, pts_f[i]-x_bar) for i in range(2*n+1))
        
        z_pts = np.array([self._h(p) for p in pts_f])
        z_bar = self.Wm @ z_pts
        Pzz = self.R + sum(self.Wc[i] * np.outer(z_pts[i]-z_bar, z_pts[i]-z_bar) for i in range(2*n+1))
        Pxz = sum(self.Wc[i] * np.outer(pts_f[i]-x_bar, z_pts[i]-z_bar) for i in range(2*n+1))
        
        K = Pxz @ np.linalg.inv(Pzz)
        res = z - z_bar
        self.x = x_bar + K @ res
        self.P = P_bar - K @ Pzz @ K.T
        
        self.P = 0.5 * (self.P + self.P.T)
        if np.isnan(self.P).any() or np.isinf(self.P).any():
            self.P = np.eye(n) * 0.1
            
        matrix_to_decompose = (n + self.lam) * self.P + 1e-6 * np.eye(n)
        matrix_to_decompose = 0.5 * (matrix_to_decompose + matrix_to_decompose.T)
        try:
            S_root = np.linalg.cholesky(matrix_to_decompose)
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(matrix_to_decompose)
            eigvals = np.maximum(eigvals, 1e-8)
            matrix_pd = eigvecs @ np.diag(eigvals) @ eigvecs.T
            matrix_pd = 0.5 * (matrix_pd + matrix_pd.T) + 1e-6 * np.eye(n)
            try:
                S_root = np.linalg.cholesky(matrix_pd)
                self.P = (matrix_pd - 1e-6 * np.eye(n)) / (n + self.lam)
            except np.linalg.LinAlgError:
                self.P = np.eye(n) * 0.1
                S_root = np.linalg.cholesky((n + self.lam) * self.P)

        return res, Pzz


# ======================================================================
#  2. Plot Utilities 
# ======================================================================
def draw_label_spans(ax, label_s, label_h, label_d):
    for s, e in get_contiguous_intervals(label_s): 
        ax.axvspan(s, e, color='red', alpha=0.15, label='Sensor Attack' if s==get_contiguous_intervals(label_s)[0][0] else "")
    for s, e in get_contiguous_intervals(label_h): 
        ax.axvspan(s, e, color='deepskyblue', alpha=0.25, label='Hijack Attack' if s==get_contiguous_intervals(label_h)[0][0] else "")
    for s, e in get_contiguous_intervals(label_d): 
        ax.axvspan(s, e, color='limegreen', alpha=0.15, label='Disturbance' if s==get_contiguous_intervals(label_d)[0][0] else "")

def plot_position_2d(setpoint, meas, est, label_s, label_h, label_d, title, save_path):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(title, fontsize=16, fontweight='bold')
    axis_labels = ['X (East) Position', 'Y (North) Position', 'Z (Up) Position']
    for i in range(3):
        if setpoint is not None: axes[i].plot(setpoint[:, i], 'g--', lw=1.8, alpha=0.8, label='Setpoint (Ref)' if i==0 else "")
        axes[i].plot(meas[:, i], color='gray', lw=1.2, alpha=0.6, label='Measured (GPS)' if i==0 else "")
        axes[i].plot(est[:, i], color='orange', lw=1.5, label='Estimated (Dyn UKF)' if i==0 else "")
        draw_label_spans(axes[i], label_s, label_h, label_d)
        axes[i].set_ylabel(axis_labels[i]); axes[i].grid(True, alpha=0.3)
        if i == 0: axes[i].legend(loc='upper right')
    axes[2].set_xlabel('Steps')
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()

def plot_3d_trajectory(setpoint, meas_pos, est_pos_d, title, save_path):
    fig = plt.figure(figsize=(10, 8))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    ax = fig.add_subplot(111, projection='3d')
    if setpoint is not None: ax.plot(setpoint[:,0], setpoint[:,1], setpoint[:,2], 'g--', lw=1.8, label='Setpoint / Ground Truth', alpha=0.7)
    ax.plot(meas_pos[:,0], meas_pos[:,1], meas_pos[:,2], color='gray', lw=1.0, label='Measured (GPS)', alpha=0.5)
    ax.plot(est_pos_d[:,0], est_pos_d[:,1], est_pos_d[:,2], color='orange', lw=1.5, label='Estimated (Dyn UKF)')
    ax.set_xlabel('E'); ax.set_ylabel('N'); ax.set_zlabel('U'); ax.legend()
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()

def plot_rmse_monitoring(val_pos, val_vel, val_gyr, label_s, label_h, label_d, title, save_path):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(title, fontsize=15, fontweight='bold')
    axes[0].plot(val_pos, color='#1f77b4', lw=1.2, label='Dynamics Pos RMSE')
    draw_label_spans(axes[0], label_s, label_h, label_d); axes[0].set_ylabel('RMSE (m)'); axes[0].grid(True, alpha=0.3); axes[0].legend(loc='upper right')
    axes[1].plot(val_vel, color='#ff7f0e', lw=1.2, label='Dynamics Vel RMSE')
    draw_label_spans(axes[1], label_s, label_h, label_d); axes[1].set_ylabel('RMSE (m/s)'); axes[1].grid(True, alpha=0.3); axes[1].legend(loc='upper right')
    axes[2].plot(val_gyr, color='#2ca02c', lw=1.2, label='Dynamics Gyr RMSE')
    draw_label_spans(axes[2], label_s, label_h, label_d); axes[2].set_ylabel('RMSE (rad/s)'); axes[2].set_xlabel('Steps'); axes[2].grid(True, alpha=0.3); axes[2].legend(loc='upper right')
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()

def plot_detailed_dynamic(meas, est, res, label_s, label_h, label_d, names, title, save_path):
    n_ch = len(names); cols = 3; rows = 2 * int(np.ceil(n_ch / 3.0))
    fig, axes = plt.subplots(rows, cols, figsize=(18, 3.5 * rows))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    for i in range(n_ch):
        r_t, c = divmod(i, 3); r_r = r_t + int(rows/2)
        axes[r_t, c].plot(meas[:, i], 'k--', alpha=0.5, label='Measured'); axes[r_t, c].plot(est[:, i], 'orange', lw=1.0, label='Estimated')
        draw_label_spans(axes[r_t, c], label_s, label_h, label_d); axes[r_t, c].set_title(names[i] + " Tracking"); axes[r_t, c].grid(True, alpha=0.2)
        if i == 0: axes[r_t, c].legend(loc='upper right')
        axes[r_r, c].plot(res[:, i], 'r-', lw=0.8); draw_label_spans(axes[r_r, c], label_s, label_h, label_d)
        axes[r_r, c].set_title(names[i] + " Residual"); axes[r_r, c].grid(True, alpha=0.2)
    plt.tight_layout(); plt.savefig(save_path, dpi=120); plt.close()

def plot_rmse_vs_nis(rmse_pos, rmse_vel, rmse_gyr, nis_pos, nis_vel, nis_gyr, label_s, label_h, label_d, title, save_path):
    fig, axes = plt.subplots(3, 2, figsize=(16, 10), sharex='col')
    fig.suptitle(title, fontsize=15, fontweight='bold')
    axes[0,0].plot(rmse_pos, color='#1f77b4', lw=1.0); axes[0,0].set_ylabel('RMSE (m)'); draw_label_spans(axes[0,0], label_s, label_h, label_d); axes[0,0].set_title('Dyn Pos RMSE'); axes[0,0].grid(True, alpha=0.3)
    axes[1,0].plot(rmse_vel, color='#ff7f0e', lw=1.0); axes[1,0].set_ylabel('RMSE (m/s)'); draw_label_spans(axes[1,0], label_s, label_h, label_d); axes[1,0].set_title('Dyn Vel RMSE'); axes[1,0].grid(True, alpha=0.3)
    axes[2,0].plot(rmse_gyr, color='#2ca02c', lw=1.0); axes[2,0].set_ylabel('RMSE (rad/s)'); draw_label_spans(axes[2,0], label_s, label_h, label_d); axes[2,0].set_title('Dyn Gyr RMSE'); axes[2,0].set_xlabel('Steps'); axes[2,0].grid(True, alpha=0.3)
    axes[0,1].plot(nis_pos, color='#1f77b4', lw=1.0); axes[0,1].set_ylabel('NIS (0~1)'); draw_label_spans(axes[0,1], label_s, label_h, label_d); axes[0,1].set_title('Dyn Pos NIS'); axes[0,1].set_ylim(-0.05, 1.05); axes[0,1].grid(True, alpha=0.3)
    axes[1,1].plot(nis_vel, color='#ff7f0e', lw=1.0); axes[1,1].set_ylabel('NIS (0~1)'); draw_label_spans(axes[1,1], label_s, label_h, label_d); axes[1,1].set_title('Dyn Vel NIS'); axes[1,1].set_ylim(-0.05, 1.05); axes[1,1].grid(True, alpha=0.3)
    axes[2,1].plot(nis_gyr, color='#2ca02c', lw=1.0); axes[2,1].set_ylabel('NIS (0~1)'); draw_label_spans(axes[2,1], label_s, label_h, label_d); axes[2,1].set_title('Dyn Gyr NIS'); axes[2,1].set_ylim(-0.05, 1.05); axes[2,1].set_xlabel('Steps'); axes[2,1].grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()

def plot_raw_vs_scaled_nis(raw_pos, raw_vel, raw_gyr, scaled_pos, scaled_vel, scaled_gyr, label_s, label_h, label_d, title, save_path):
    fig, axes = plt.subplots(3, 2, figsize=(16, 10), sharex='col')
    fig.suptitle(title, fontsize=15, fontweight='bold')
    axes[0,0].plot(raw_pos, color='#1f77b4', lw=1.0); axes[0,0].set_ylabel('Raw NIS'); draw_label_spans(axes[0,0], label_s, label_h, label_d); axes[0,0].set_title('Dyn Pos RAW NIS'); axes[0,0].grid(True, alpha=0.3)
    axes[1,0].plot(raw_vel, color='#ff7f0e', lw=1.0); axes[1,0].set_ylabel('Raw NIS'); draw_label_spans(axes[1,0], label_s, label_h, label_d); axes[1,0].set_title('Dyn Vel RAW NIS'); axes[1,0].grid(True, alpha=0.3)
    axes[2,0].plot(raw_gyr, color='#2ca02c', lw=1.0); axes[2,0].set_ylabel('Raw NIS'); draw_label_spans(axes[2,0], label_s, label_h, label_d); axes[2,0].set_title('Dyn Gyr RAW NIS'); axes[2,0].set_xlabel('Steps'); axes[2,0].grid(True, alpha=0.3)
    axes[0,1].plot(scaled_pos, color='#1f77b4', lw=1.0); axes[0,1].set_ylabel('Scaled NIS (0~1)'); draw_label_spans(axes[0,1], label_s, label_h, label_d); axes[0,1].set_title('Dyn Pos SCALED NIS'); axes[0,1].set_ylim(-0.05, 1.05); axes[0,1].grid(True, alpha=0.3)
    axes[1,1].plot(scaled_vel, color='#ff7f0e', lw=1.0); axes[1,1].set_ylabel('Scaled NIS (0~1)'); draw_label_spans(axes[1,1], label_s, label_h, label_d); axes[1,1].set_title('Dyn Vel SCALED NIS'); axes[1,1].set_ylim(-0.05, 1.05); axes[1,1].grid(True, alpha=0.3)
    axes[2,1].plot(scaled_gyr, color='#2ca02c', lw=1.0); axes[2,1].set_ylabel('Scaled NIS (0~1)'); draw_label_spans(axes[2,1], label_s, label_h, label_d); axes[2,1].set_title('Dyn Gyr SCALED NIS'); axes[2,1].set_ylim(-0.05, 1.05); axes[2,1].set_xlabel('Steps'); axes[2,1].grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()

# ★ [신규 플롯] 오일러 각도(자세) 추종 디버깅 플롯
def plot_euler_comparison(gt_euler, est_euler, lbl_s, lbl_h, lbl_d, title, save_path):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(title, fontsize=16, fontweight='bold')
    labels = ['Roll (phi) [rad]', 'Pitch (theta) [rad]', 'Yaw (psi) [rad]']
    for i in range(3):
        axes[i].plot(gt_euler[:, i], 'k--', lw=1.5, alpha=0.8, label='Ground Truth (Actual)')
        axes[i].plot(est_euler[:, i], color='orange', lw=1.5, alpha=0.9, label='Estimated (Dyn UKF)')
        draw_label_spans(axes[i], lbl_s, lbl_h, lbl_d)
        axes[i].set_ylabel(labels[i])
        axes[i].grid(True, alpha=0.3)
        if i == 0: axes[i].legend(loc='upper right', fontsize=10)
    axes[2].set_xlabel('Steps')
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()

def generate_all_plots(res, lbl_s, lbl_h, lbl_d, p, out_dir):
    idx = res['rl_indices']
    plot_position_2d(res['setpoint_pos'], res['meas_pos'], res['est_pos_d_enu'], lbl_s, lbl_h, lbl_d, f"2D Position Tracking ({p})", os.path.join(out_dir, 'pos_tracking_2d.png'))
    plot_3d_trajectory(res['setpoint_pos'], res['meas_pos'], res['est_pos_d_enu'], f"3D Trajectory Dynamics Only ({p})", os.path.join(out_dir, 'trajectory_3d.png'))
    plot_rmse_monitoring(res['rmse_d_pos'], res['rmse_d_vel'], res['rmse_d_gyr'], lbl_s, lbl_h, lbl_d, f"RMSE Monitoring 50Hz ({p})", os.path.join(out_dir, 'rmse_50hz.png'))
    plot_rmse_monitoring(res['rl_rmse_d_pos'], res['rl_rmse_d_vel'], res['rl_rmse_d_gyr'], lbl_s[idx], lbl_h[idx], lbl_d[idx], f"RMSE Monitoring 10Hz GPS-fresh ({p})", os.path.join(out_dir, 'rmse_10hz.png'))
    plot_rmse_vs_nis(res['rmse_d_pos'], res['rmse_d_vel'], res['rmse_d_gyr'], res['nis_d_pos'], res['nis_d_vel'], res['nis_d_gyr'], lbl_s, lbl_h, lbl_d, f"RMSE vs NIS 50Hz ({p})", os.path.join(out_dir, 'rmse_vs_nis_50hz.png'))
    plot_rmse_vs_nis(res['rl_rmse_d_pos'], res['rl_rmse_d_vel'], res['rl_rmse_d_gyr'], res['rl_nis_d_pos'], res['rl_nis_d_vel'], res['rl_nis_d_gyr'], lbl_s[idx], lbl_h[idx], lbl_d[idx], f"RMSE vs NIS 10Hz GPS-fresh ({p})", os.path.join(out_dir, 'rmse_vs_nis_10hz.png'))
    plot_raw_vs_scaled_nis(res['rl_nis_raw_d_pos'], res['rl_nis_raw_d_vel'], res['rl_nis_raw_d_gyr'], res['rl_nis_d_pos'], res['rl_nis_d_vel'], res['rl_nis_d_gyr'], lbl_s[idx], lbl_h[idx], lbl_d[idx], f"RAW vs SCALED NIS 10Hz ({p})", os.path.join(out_dir, 'nis_raw_vs_scaled_10hz.png'))
    plot_euler_comparison(res['gt_euler'], res['est_euler'], lbl_s, lbl_h, lbl_d, f"Euler Angle (Attitude) Monitoring ({p})", os.path.join(out_dir, 'euler_tracking.png'))
    m_d_posvel = res['z_12d'][:, 0:6]; r_d_posvel = res['res_12ch'][:, 0:6]; e_d_posvel = m_d_posvel - r_d_posvel
    plot_detailed_dynamic(m_d_posvel, e_d_posvel, r_d_posvel, lbl_s, lbl_h, lbl_d, ['N (Pos)', 'E (Pos)', 'D (Pos)', 'Vn (Vel)', 'Ve (Vel)', 'Vd (Vel)'], "Dynamics-centric Detail (Pos & Vel)", os.path.join(out_dir, 'dyn_detail_posvel.png'))
    m_d_gyr = res['z_12d'][:, 9:12]; r_d_gyr = res['res_12ch'][:, 9:12]; e_d_gyr = m_d_gyr - r_d_gyr
    plot_detailed_dynamic(m_d_gyr, e_d_gyr, r_d_gyr, lbl_s, lbl_h, lbl_d, ['p (Roll rate)', 'q (Pitch rate)', 'r (Yaw rate)'], "Dynamics-centric Detail (Gyro)", os.path.join(out_dir, 'dyn_detail_gyr.png'))


# ======================================================================
#  3. Core Loop
# ======================================================================
def process_episode(data, calib):
    dt = float(data['dt']); N = int(data['steps'])
    
    gps_pos_enu = data['obs_gps_pos']
    gps_vel_enu = data['obs_gps_vel']
    gps_pos_ned = np.column_stack([gps_pos_enu[:, 1], gps_pos_enu[:, 0], -gps_pos_enu[:, 2]])
    gps_vel_ned = np.column_stack([gps_vel_enu[:, 1], gps_vel_enu[:, 0], -gps_vel_enu[:, 2]])
    
    flow_vel = data['flow_vel']
    baro_d = data['baro_alt'].reshape(-1, 1)

    z_9d = np.hstack([gps_pos_ned, gps_vel_ned, baro_d, flow_vel])
    imu = np.hstack([data['accelerometer'], data['gyro']])
    u_phys = to_physical_u(data['thrust'], data['torque'], calib)
    euler_ned = data['euler']
    z_12d = np.hstack([z_9d, imu[:, 3:6]])

    ukf_dyn = DynamicsUKF(dt=dt, calib=calib)
    ukf_dyn.x[0:3] = z_9d[0, 0:3]     
    ukf_dyn.x[3:6] = euler_ned[0]     
    ukf_dyn.x[6:9] = z_9d[0, 3:6]     
    ukf_dyn.x[9:12] = imu[0, 3:6]     

    if 'setpoint' in data:
        sp_pos = data['setpoint'][:, 0:3].copy(); sp_pos[:, 2] = -sp_pos[:, 2] 
    elif 'ref_pos' in data:
        sp_pos = data['ref_pos'][:, 0:3].copy(); sp_pos[:, 2] = -sp_pos[:, 2]  
    else:
        sp_pos = data.get('gt_pos', None)

    res = { 
        'rmse_d_pos': np.zeros(N), 'rmse_d_vel': np.zeros(N), 'rmse_d_gyr': np.zeros(N), 
        'nis_raw_d_pos': np.zeros(N), 'nis_raw_d_vel': np.zeros(N), 'nis_raw_d_gyr': np.zeros(N), 
        'nis_d_pos': np.zeros(N), 'nis_d_vel': np.zeros(N), 'nis_d_gyr': np.zeros(N),
        'est_pos_d_enu': np.zeros((N, 3)), 'res_12ch': np.zeros((N, 12)),
        'est_euler': np.zeros((N, 3)),
        'setpoint_pos': sp_pos
    }

    rl_rmse_d_pos_list = []; rl_rmse_d_vel_list = []; rl_rmse_d_gyr_list = []
    rl_nis_raw_d_pos_list = []; rl_nis_raw_d_vel_list = []; rl_nis_raw_d_gyr_list = [] 
    rl_nis_d_pos_list = []; rl_nis_d_vel_list = []; rl_nis_d_gyr_list = []
    rl_indices = []

    for k in range(N):
        rd, Pzz_d = ukf_dyn.step(z_12d[k], u_phys[k])
        
        res['rmse_d_pos'][k] = np.sqrt(np.mean(rd[0:3]**2))
        res['nis_raw_d_pos'][k], res['nis_d_pos'][k] = compute_nis_scaled(rd[0:3], Pzz_d[0:3, 0:3], 3.0)
        res['rmse_d_vel'][k] = np.sqrt(np.mean(rd[3:6]**2))
        res['nis_raw_d_vel'][k], res['nis_d_vel'][k] = compute_nis_scaled(rd[3:6], Pzz_d[3:6, 3:6], 3.0)
        res['rmse_d_gyr'][k] = np.sqrt(np.mean(rd[9:12]**2))
        res['nis_raw_d_gyr'][k], res['nis_d_gyr'][k] = compute_nis_scaled(rd[9:12], Pzz_d[9:12, 9:12], 3.0)
        
        res['est_pos_d_enu'][k] = [ukf_dyn.x[1], ukf_dyn.x[0], -ukf_dyn.x[2]]
        res['est_euler'][k] = ukf_dyn.x[3:6].copy()
        res['res_12ch'][k] = rd

        if k == 0 or np.any(gps_pos_ned[k] != gps_pos_ned[k-1]):
            rl_rmse_d_pos_list.append(res['rmse_d_pos'][k])
            rl_rmse_d_vel_list.append(res['rmse_d_vel'][k])
            rl_rmse_d_gyr_list.append(res['rmse_d_gyr'][k])
            rl_nis_raw_d_pos_list.append(res['nis_raw_d_pos'][k])
            rl_nis_raw_d_vel_list.append(res['nis_raw_d_vel'][k])
            rl_nis_raw_d_gyr_list.append(res['nis_raw_d_gyr'][k])
            rl_nis_d_pos_list.append(res['nis_d_pos'][k])
            rl_nis_d_vel_list.append(res['nis_d_vel'][k])
            rl_nis_d_gyr_list.append(res['nis_d_gyr'][k])
            rl_indices.append(k)

    res['rl_rmse_d_pos'] = np.array(rl_rmse_d_pos_list); res['rl_rmse_d_vel'] = np.array(rl_rmse_d_vel_list); res['rl_rmse_d_gyr'] = np.array(rl_rmse_d_gyr_list)
    res['rl_nis_raw_d_pos'] = np.array(rl_nis_raw_d_pos_list); res['rl_nis_raw_d_vel'] = np.array(rl_nis_raw_d_vel_list); res['rl_nis_raw_d_gyr'] = np.array(rl_nis_raw_d_gyr_list)
    res['rl_nis_d_pos'] = np.array(rl_nis_d_pos_list); res['rl_nis_d_vel'] = np.array(rl_nis_d_vel_list); res['rl_nis_d_gyr'] = np.array(rl_nis_d_gyr_list)
    res['rl_indices'] = np.array(rl_indices)

    res['gt_pos'] = data.get('gt_pos', None)
    res['gt_euler'] = euler_ned
    res['meas_pos'] = gps_pos_enu
    res['z_12d'] = z_12d
    
    lbl_s = data.get('label_sensor', np.zeros(N, dtype=np.int32))
    lbl_h = data.get('label_hijack', np.zeros(N, dtype=np.int32))
    lbl_d = data.get('disturbance_label', np.zeros(N, dtype=np.int32))
    
    return res, lbl_s, lbl_h, lbl_d

# ======================================================================
#  4. 실행 모드
# ======================================================================
def mode_single(args):
    data = np.load(args.input, allow_pickle=True)
    calib = load_calibration(args.calib)
    base = os.path.splitext(os.path.basename(args.input))[0]
    out_dir = f"results_{base}"; os.makedirs(out_dir, exist_ok=True)

    print(f"[*] Processing Single: {args.input}")
    res, lbl_s, lbl_h, lbl_d = process_episode(data, calib)
    p = str(data.get('pattern', 'unknown'))

    generate_all_plots(res, lbl_s, lbl_h, lbl_d, p, out_dir)

    idx = res['rl_indices']
    rl_obs_rmse = np.column_stack([res['rl_rmse_d_pos'], res['rl_rmse_d_vel'], res['rl_rmse_d_gyr']])
    rl_obs_nis = np.column_stack([res['rl_nis_d_pos'], res['rl_nis_d_vel'], res['rl_nis_d_gyr']])
    np.savez_compressed(os.path.join(out_dir, "rl_dataset.npz"), 
                        rl_obs=rl_obs_rmse,       
                        rl_obs_nis=rl_obs_nis,    
                        label_sensor=lbl_s[idx], label_hijack=lbl_h[idx], label_dist=lbl_d[idx])
    
    print(f"[*] Success. Files saved in {out_dir}/")

def mode_batch(args):
    calib = load_calibration(args.calib)
    batch_out_dir = "batch_plots"
    batch_data_dir = "batch_processed_data" 
    os.makedirs(batch_out_dir, exist_ok=True)
    os.makedirs(batch_data_dir, exist_ok=True)
    
    search_pattern = os.path.join(args.input_dir, '*.npz')
    files = sorted(glob.glob(search_pattern))
    
    if not files:
        print(f"\n[!] '{args.input_dir}' 폴더에 처리할 .npz 파일이 없습니다.")
        return

    all_obs_rmse, all_obs_nis = [], []
    all_lbl_s, all_lbl_h, all_lbl_d = [], [], []
    total_steps = 0

    print(f"\n[*] Batch Mode: '{args.input_dir}' 폴더 내 총 {len(files)}개 파일을 순차적으로 실행합니다.")

    for count, fp in enumerate(files):
        file_basename = os.path.splitext(os.path.basename(fp))[0]
        print(f"  [{count+1}/{len(files)}] 처리 중: {file_basename}")
        
        try:
            data = np.load(fp, allow_pickle=True)
            p = str(data.get('pattern', file_basename))
            
            res, lbl_s, lbl_h, lbl_d = process_episode(data, calib)
            
            ep_dir = os.path.join(batch_out_dir, file_basename)
            os.makedirs(ep_dir, exist_ok=True)
            
            generate_all_plots(res, lbl_s, lbl_h, lbl_d, p, ep_dir)
            
            idx = res['rl_indices']
            obs_rmse = np.column_stack([res['rl_rmse_d_pos'], res['rl_rmse_d_vel'], res['rl_rmse_d_gyr']])
            obs_nis = np.column_stack([res['rl_nis_d_pos'], res['rl_nis_d_vel'], res['rl_nis_d_gyr']])
            
            lbl_s_10hz = lbl_s[idx]
            lbl_h_10hz = lbl_h[idx]
            lbl_d_10hz = lbl_d[idx]

            indiv_filepath = os.path.join(batch_data_dir, f"{file_basename}_rl.npz")
            np.savez_compressed(indiv_filepath, 
                                rl_obs=obs_rmse,       
                                rl_obs_nis=obs_nis,    
                                label_sensor=lbl_s_10hz, 
                                label_hijack=lbl_h_10hz, 
                                label_dist=lbl_d_10hz)
            
            all_obs_rmse.append(obs_rmse)
            all_obs_nis.append(obs_nis)
            all_lbl_s.append(lbl_s_10hz); all_lbl_h.append(lbl_h_10hz); all_lbl_d.append(lbl_d_10hz)
            total_steps += len(idx)
            
        except Exception as e:
            print(f"  [!] {file_basename} 처리 중 오류 발생: {e}")
            continue

    if total_steps == 0:
        print("\n[!] 통합할 데이터가 없습니다.")
        return

    final_obs_rmse = np.vstack(all_obs_rmse)
    final_obs_nis = np.vstack(all_obs_nis)
    final_lbl_s = np.concatenate(all_lbl_s)
    final_lbl_h = np.concatenate(all_lbl_h)
    final_lbl_d = np.concatenate(all_lbl_d)

    out_path = f"{args.output}.npz"
    np.savez_compressed(out_path, 
                        rl_obs=final_obs_rmse, 
                        rl_obs_nis=final_obs_nis, 
                        label_sensor=final_lbl_s, 
                        label_hijack=final_lbl_h, 
                        label_dist=final_lbl_d)
    
    print(f"\n[*] 자동 Batch 처리 완료!")
    print(f"  1) 개별 에피소드 플롯 ➡️ '{batch_out_dir}/' 폴더 저장 완료")
    print(f"  2) 개별 에피소드 데이터 ➡️ '{batch_data_dir}/' 폴더 저장 완료")
    print(f"  3) 통합 학습용 데이터({total_steps} 스텝) ➡️ '{out_path}' 저장 완료")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='single', choices=['single', 'batch'])
    parser.add_argument('--input', help='단일 처리할 .npz 파일 경로 (Single 모드용)')
    parser.add_argument('--calib', default='calibration.json')
    parser.add_argument('--input-dir', default='data_raw', help='배치 처리할 데이터 폴더 (Batch 모드용)')
    parser.add_argument('--output', default='swrl_final_dataset', help='생성될 통합 파일 이름')
    
    args = parser.parse_args()
    if args.mode == 'single': 
        if not args.input:
            print("[!] Single 모드에서는 --input 으로 파일 경로를 지정해야 합니다.")
            return
        mode_single(args)
    elif args.mode == 'batch': 
        mode_batch(args)

if __name__ == '__main__': main()
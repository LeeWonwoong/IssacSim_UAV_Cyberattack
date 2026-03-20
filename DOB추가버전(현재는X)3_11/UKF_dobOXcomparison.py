#!/usr/bin/env python3
"""
postprocess_Integrated_final.py (Ablation Study: 5-Split Plots)
====================================================================
[주요 업데이트 내역]
1. 플롯 5종 분리: No-DOB 단독, DOB 단독, NIS 직접 비교, Ext Force 단독
2. [신규] 오일러 각도(Attitude) 오염 분석 플롯 추가 (ablation_5_euler_corruption.png)
   - 은닉 상태인 Roll/Pitch 가 공격 후 어떻게 꼬여서 발산하는지 증명
3. 시인성 개선: 비교 플롯에서 회색 선 대신 강렬한 빨강(No-DOB)과 파랑(DOB) 사용
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
#  1-A. SOTA Baseline UKF (No DOB, 12 states)
# ======================================================================
class DynamicsUKF_NoDOB:
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
        
        self.Q = np.diag([1e-3]*3 + [1e-4]*3 + [3e-2]*3 + [1e-3]*3)
        self.R = np.diag([1.0]*3 + [1.0]*3 + [0.5] + [0.5]*2 + [1.0]*3)
        self.x, self.P = np.zeros(12), np.eye(12) * 0.1

    def _f(self, x, u):
        s = x.copy(); sdt = 0.005; n_sub = int(self.dt/sdt)
        for _ in range(n_sub):
            phi, th, psi = s[3:6]; vx, vy, vz = s[6:9]; p, q, r = s[9:12]
            
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
        try: S_root = np.linalg.cholesky((n + self.lam) * self.P + 1e-6*np.eye(n))
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
        matrix_to_decompose = (n + self.lam) * self.P + 1e-6 * np.eye(n)
        matrix_to_decompose = 0.5 * (matrix_to_decompose + matrix_to_decompose.T)
        try: np.linalg.cholesky(matrix_to_decompose)
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(matrix_to_decompose)
            eigvals = np.maximum(eigvals, 1e-8)
            matrix_pd = eigvecs @ np.diag(eigvals) @ eigvecs.T
            self.P = (matrix_pd - 1e-6 * np.eye(n)) / (n + self.lam)
            
        return res, Pzz

# ======================================================================
#  1-B. Proposed UKF (Dynamics + FOGM DOB, 15 states)
# ======================================================================
class DynamicsUKF_DOB:
    def __init__(self, dt=0.02, calib=None):
        self.nx, self.nz, self.dt = 15, 12, dt
        d = calib['drone']
        self.m, self.g = d['mass'], d['g']
        self.I = [d['Ixx'], d['Iyy'], d['Izz']]
        self.drag = np.array(calib['drag'])
        n, lam = self.nx, 0.5**2 * self.nx - self.nx
        self.lam = lam
        self.Wm, self.Wc = np.full(2*n+1, 1.0/(2*(n+lam))), np.full(2*n+1, 1.0/(2*(n+lam)))
        self.Wm[0], self.Wc[0] = lam/(n+lam), lam/(n+lam) + (1 - 0.5**2 + 2.0)
        
        self.tau_ext = 4.0 
        self.Q = np.diag([1e-3]*3 + [1e-4]*3 + [1e-2]*3 + [1e-3]*3 + [5e-2]*3)
        self.R = np.diag([1.0]*3 + [1.0]*3 + [0.5] + [0.5]*2 + [1.0]*3)
        self.x, self.P = np.zeros(15), np.eye(15) * 0.1

    def _f(self, x, u):
        s = x.copy(); sdt = 0.005; n_sub = int(self.dt/sdt)
        for _ in range(n_sub):
            phi, th, psi = s[3:6]; vx, vy, vz = s[6:9]; p, q, r = s[9:12]
            f_ext_ned = s[12:15]
            limit = 1.5 
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
            R = np.array([
                [ct*cps, sp*st*cps-cp*sps, cp*st*cps+sp*sps],
                [ct*sps, sp*st*sps+cp*cps, cp*st*sps-sp*cps],
                [-st, sp*ct, cp*ct]
            ])
            f_total_ned = R @ f_total_body
            accel_ned = (f_total_ned / self.m) + np.array([0, 0, self.g]) + (f_ext_ned / self.m)
            
            s[0:3] += np.array([vx, vy, vz])*sdt
            s[3:6] += np.array([p + sp*tt*q + cp*tt*r, cp*q - sp*r, sp/(ct+1e-10)*q + cp/(ct+1e-10)*r])*sdt
            s[6:9] += accel_ned * sdt
            s[9] += ((self.I[1]-self.I[2])/self.I[0]*q*r + u[1]/self.I[0])*sdt
            s[10] += ((self.I[2]-self.I[0])/self.I[1]*p*r + u[2]/self.I[1])*sdt
            s[11] += ((self.I[0]-self.I[1])/self.I[2]*p*q + u[3]/self.I[2])*sdt
            
            s[12:15] -= (s[12:15] / self.tau_ext) * sdt
            
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
        try: S_root = np.linalg.cholesky((n + self.lam) * self.P + 1e-6*np.eye(n))
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
        matrix_to_decompose = (n + self.lam) * self.P + 1e-6 * np.eye(n)
        matrix_to_decompose = 0.5 * (matrix_to_decompose + matrix_to_decompose.T)
        try: np.linalg.cholesky(matrix_to_decompose)
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(matrix_to_decompose)
            eigvals = np.maximum(eigvals, 1e-8)
            matrix_pd = eigvecs @ np.diag(eigvals) @ eigvecs.T
            self.P = (matrix_pd - 1e-6 * np.eye(n)) / (n + self.lam)
            
        return res, Pzz


# ======================================================================
#  2. Plot Utilities (4-Split + Euler Debugging Visualizations)
# ======================================================================
def draw_label_spans(ax, label_s, label_h, label_d):
    for s, e in get_contiguous_intervals(label_s): 
        ax.axvspan(s, e, color='red', alpha=0.15, label='Sensor Attack' if s==get_contiguous_intervals(label_s)[0][0] else "")
    for s, e in get_contiguous_intervals(label_h): 
        ax.axvspan(s, e, color='deepskyblue', alpha=0.25, label='Hijack Attack' if s==get_contiguous_intervals(label_h)[0][0] else "")
    for s, e in get_contiguous_intervals(label_d): 
        ax.axvspan(s, e, color='limegreen', alpha=0.15, label='Disturbance' if s==get_contiguous_intervals(label_d)[0][0] else "")

def plot_single_rmse_nis(res, lbl_s, lbl_h, lbl_d, title, save_path, is_dob=True):
    fig, axes = plt.subplots(3, 2, figsize=(16, 10), sharex='col')
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    color_rmse = '#1f77b4' if is_dob else 'dimgray'
    color_nis = '#ff7f0e' if is_dob else 'indianred'

    axes[0,0].plot(res['rmse_d_pos'], color=color_rmse, lw=1.5); axes[0,0].set_ylabel('RMSE (m)'); axes[0,0].set_title('Position RMSE'); axes[0,0].grid(True, alpha=0.3)
    axes[1,0].plot(res['rmse_d_vel'], color=color_rmse, lw=1.5); axes[1,0].set_ylabel('RMSE (m/s)'); axes[1,0].set_title('Velocity RMSE'); axes[1,0].grid(True, alpha=0.3)
    axes[2,0].plot(res['rmse_d_gyr'], color=color_rmse, lw=1.5); axes[2,0].set_ylabel('RMSE (rad/s)'); axes[2,0].set_title('Gyro RMSE'); axes[2,0].set_xlabel('Steps'); axes[2,0].grid(True, alpha=0.3)

    axes[0,1].plot(res['nis_d_pos'], color=color_nis, lw=1.5); axes[0,1].set_ylabel('NIS (0~1)'); axes[0,1].set_title('Position Scaled NIS'); axes[0,1].set_ylim(-0.05, 1.05); axes[0,1].grid(True, alpha=0.3)
    axes[1,1].plot(res['nis_d_vel'], color=color_nis, lw=1.5); axes[1,1].set_ylabel('NIS (0~1)'); axes[1,1].set_title('Velocity Scaled NIS'); axes[1,1].set_ylim(-0.05, 1.05); axes[1,1].grid(True, alpha=0.3)
    axes[2,1].plot(res['nis_d_gyr'], color=color_nis, lw=1.5); axes[2,1].set_ylabel('NIS (0~1)'); axes[2,1].set_title('Gyro Scaled NIS'); axes[2,1].set_ylim(-0.05, 1.05); axes[2,1].set_xlabel('Steps'); axes[2,1].grid(True, alpha=0.3)

    for ax in axes.flat: draw_label_spans(ax, lbl_s, lbl_h, lbl_d)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()

def plot_compare_nis_only(res_nodob, res_dob, lbl_s, lbl_h, lbl_d, title, save_path):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"NIS Comparison: No-DOB vs Proposed DOB\n({title})", fontsize=16, fontweight='bold')
    
    c_nodob = 'crimson'      
    c_dob = 'dodgerblue'     

    axes[0].plot(res_nodob['nis_d_pos'], color=c_nodob, linestyle='-', alpha=0.7, lw=1.5, label='No-DOB (Baseline)')
    axes[0].plot(res_dob['nis_d_pos'], color=c_dob, lw=2.0, label='Proposed DOB')
    axes[0].set_ylabel('Pos NIS (0~1)'); axes[0].set_ylim(-0.05, 1.05); axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper right', fontsize=11)

    axes[1].plot(res_nodob['nis_d_vel'], color=c_nodob, linestyle='-', alpha=0.7, lw=1.5, label='No-DOB (Baseline)')
    axes[1].plot(res_dob['nis_d_vel'], color=c_dob, lw=2.0, label='Proposed DOB')
    axes[1].set_ylabel('Vel NIS (0~1)'); axes[1].set_ylim(-0.05, 1.05); axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right', fontsize=11)

    axes[2].plot(res_nodob['nis_d_gyr'], color=c_nodob, linestyle='-', alpha=0.7, lw=1.5, label='No-DOB (Baseline)')
    axes[2].plot(res_dob['nis_d_gyr'], color=c_dob, lw=2.0, label='Proposed DOB')
    axes[2].set_ylabel('Gyr NIS (0~1)'); axes[2].set_ylim(-0.05, 1.05); axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc='upper right', fontsize=11)
    axes[2].set_xlabel('Steps')

    for ax in axes: draw_label_spans(ax, lbl_s, lbl_h, lbl_d)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()

def plot_ext_force_dob(res_dob, lbl_s, lbl_h, lbl_d, title, save_path):
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"DOB Estimated External Force\n({title})", fontsize=16, fontweight='bold')
    
    labels = ['Ext Force N (X axis)', 'Ext Force E (Y axis)', 'Ext Force D (Z axis)']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i in range(3):
        axes[i].plot(res_dob['ext_force'][:, i], color=colors[i], lw=1.8, label=f'Estimated {labels[i]}')
        draw_label_spans(axes[i], lbl_s, lbl_h, lbl_d)
        axes[i].set_ylabel('Force (N)')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(loc='upper right')
    axes[2].set_xlabel('Steps')
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()

# ★ [신규 플롯] 오일러 각도(자세) 오염 디버깅 플롯
def plot_euler_comparison(gt_euler, est_euler_nodob, est_euler_dob, lbl_s, lbl_h, lbl_d, title, save_path):
    """실제 기체 자세(GT)와 No-DOB, DOB 필터가 추정한 자세(Estimated Attitude)를 겹쳐서 비교"""
    gt_euler[:, 2] = np.unwrap(gt_euler[:, 2])
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"Euler Angle (Attitude) Corruption Analysis\n({title})", fontsize=16, fontweight='bold')

    labels = ['Roll (phi) [rad]', 'Pitch (theta) [rad]', 'Yaw (psi) [rad]']
    
    for i in range(3):
        # 1. Ground Truth (검은색 점선)
        axes[i].plot(gt_euler[:, i], color='black', linestyle='--', lw=1.5, alpha=0.8, label='Ground Truth (Actual)')
        # 2. No-DOB 추정치 (빨간색) - 발산하는 모습 확인용
        axes[i].plot(est_euler_nodob[:, i], color='crimson', lw=1.5, alpha=0.8, label='No-DOB (Corrupted)')
        # 3. DOB 추정치 (파란색) - 방어해 낸 모습 확인용
        axes[i].plot(est_euler_dob[:, i], color='dodgerblue', lw=1.5, alpha=0.9, label='Proposed DOB (Protected)')
        
        draw_label_spans(axes[i], lbl_s, lbl_h, lbl_d)
        axes[i].set_ylabel(labels[i])
        axes[i].grid(True, alpha=0.3)
        if i == 0:
            axes[i].legend(loc='upper right', fontsize=10)

    axes[2].set_xlabel('Steps')
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()


# ======================================================================
#  3. Core Loop (Dual Execution)
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
    euler_ned = data['euler'] # [★ Ground Truth Euler 추출]
    z_12d = np.hstack([z_9d, imu[:, 3:6]])

    ukf_nodob = DynamicsUKF_NoDOB(dt=dt, calib=calib)
    ukf_dob = DynamicsUKF_DOB(dt=dt, calib=calib)
    
    ukf_nodob.x[0:3] = z_9d[0, 0:3]; ukf_nodob.x[3:6] = euler_ned[0]     
    ukf_nodob.x[6:9] = z_9d[0, 3:6]; ukf_nodob.x[9:12] = imu[0, 3:6]     
    ukf_dob.x[0:3] = z_9d[0, 0:3]; ukf_dob.x[3:6] = euler_ned[0]     
    ukf_dob.x[6:9] = z_9d[0, 3:6]; ukf_dob.x[9:12] = imu[0, 3:6]     

    def init_res_dict():
        return { 
            'rmse_d_pos': np.zeros(N), 'rmse_d_vel': np.zeros(N), 'rmse_d_gyr': np.zeros(N), 
            'nis_d_pos': np.zeros(N), 'nis_d_vel': np.zeros(N), 'nis_d_gyr': np.zeros(N),
            'est_euler': np.zeros((N, 3)) # [★ 은닉 상태 오일러 각도 저장용 변수 추가]
        }
    
    res_nodob = init_res_dict()
    res_dob = init_res_dict()
    res_dob['ext_force'] = np.zeros((N, 3))

    rl_indices = []

    for k in range(N):
        rd_n, Pzz_n = ukf_nodob.step(z_12d[k], u_phys[k])
        res_nodob['rmse_d_pos'][k] = np.sqrt(np.mean(rd_n[0:3]**2))
        _, res_nodob['nis_d_pos'][k] = compute_nis_scaled(rd_n[0:3], Pzz_n[0:3, 0:3], 3.0)
        res_nodob['rmse_d_vel'][k] = np.sqrt(np.mean(rd_n[3:6]**2))
        _, res_nodob['nis_d_vel'][k] = compute_nis_scaled(rd_n[3:6], Pzz_n[3:6, 3:6], 3.0)
        res_nodob['rmse_d_gyr'][k] = np.sqrt(np.mean(rd_n[9:12]**2))
        _, res_nodob['nis_d_gyr'][k] = compute_nis_scaled(rd_n[9:12], Pzz_n[9:12, 9:12], 3.0)
        res_nodob['est_euler'][k] = ukf_nodob.x[3:6].copy() # No-DOB 추정 자세 기록
        
        rd_d, Pzz_d = ukf_dob.step(z_12d[k], u_phys[k])
        res_dob['rmse_d_pos'][k] = np.sqrt(np.mean(rd_d[0:3]**2))
        _, res_dob['nis_d_pos'][k] = compute_nis_scaled(rd_d[0:3], Pzz_d[0:3, 0:3], 3.0)
        res_dob['rmse_d_vel'][k] = np.sqrt(np.mean(rd_d[3:6]**2))
        _, res_dob['nis_d_vel'][k] = compute_nis_scaled(rd_d[3:6], Pzz_d[3:6, 3:6], 3.0)
        res_dob['rmse_d_gyr'][k] = np.sqrt(np.mean(rd_d[9:12]**2))
        _, res_dob['nis_d_gyr'][k] = compute_nis_scaled(rd_d[9:12], Pzz_d[9:12, 9:12], 3.0)
        res_dob['est_euler'][k] = ukf_dob.x[3:6].copy() # DOB 추정 자세 기록
        res_dob['ext_force'][k] = ukf_dob.x[12:15].copy()

        if k == 0 or np.any(gps_pos_ned[k] != gps_pos_ned[k-1]):
            rl_indices.append(k)

    # Ground truth Euler angles 전달
    res_dob['gt_euler'] = euler_ned

    lbl_s = data.get('label_sensor', np.zeros(N, dtype=np.int32))
    lbl_h = data.get('label_hijack', np.zeros(N, dtype=np.int32))
    lbl_d = data.get('disturbance_label', np.zeros(N, dtype=np.int32))
    
    return res_nodob, res_dob, np.array(rl_indices), lbl_s, lbl_h, lbl_d


# ======================================================================
#  4. 실행 모드
# ======================================================================
def mode_single(args):
    data = np.load(args.input, allow_pickle=True)
    calib = load_calibration(args.calib)
    base = os.path.splitext(os.path.basename(args.input))[0]
    out_dir = f"results_{base}"; os.makedirs(out_dir, exist_ok=True)

    print(f"[*] Processing Single (Ablation Mode): {args.input}")
    res_nodob, res_dob, idx, lbl_s, lbl_h, lbl_d = process_episode(data, calib)
    p = str(data.get('pattern', 'unknown'))

    # 기존 4개의 개별 플롯 생성
    plot_single_rmse_nis(res_nodob, lbl_s, lbl_h, lbl_d, f"1. No-DOB (Baseline) RMSE & NIS\n({p})", os.path.join(out_dir, 'ablation_1_nodob_rmse_nis.png'), is_dob=False)
    plot_single_rmse_nis(res_dob, lbl_s, lbl_h, lbl_d, f"2. Proposed DOB RMSE & NIS\n({p})", os.path.join(out_dir, 'ablation_2_dob_rmse_nis.png'), is_dob=True)
    plot_compare_nis_only(res_nodob, res_dob, lbl_s, lbl_h, lbl_d, p, os.path.join(out_dir, 'ablation_3_compare_nis.png'))
    plot_ext_force_dob(res_dob, lbl_s, lbl_h, lbl_d, p, os.path.join(out_dir, 'ablation_4_ext_force.png'))
    
    # [★ 신규 플롯 호출] 오일러 각도 디버깅 플롯
    plot_euler_comparison(res_dob['gt_euler'], res_nodob['est_euler'], res_dob['est_euler'], lbl_s, lbl_h, lbl_d, p, os.path.join(out_dir, 'ablation_5_euler_corruption.png'))
    
    # RL Dataset
    rl_obs_rmse = np.column_stack([res_dob['rmse_d_pos'][idx], res_dob['rmse_d_vel'][idx], res_dob['rmse_d_gyr'][idx]])
    rl_obs_nis = np.column_stack([res_dob['nis_d_pos'][idx], res_dob['nis_d_vel'][idx], res_dob['nis_d_gyr'][idx]])
    rl_ext_force = res_dob['ext_force'][idx]
    
    np.savez_compressed(os.path.join(out_dir, "rl_dataset.npz"), 
                        rl_obs=rl_obs_rmse, rl_obs_nis=rl_obs_nis, rl_ext_force=rl_ext_force,
                        label_sensor=lbl_s[idx], label_hijack=lbl_h[idx], label_dist=lbl_d[idx])
    
    print(f"[*] Success. 5 Ablation Plots (including Euler Debugging) saved in {out_dir}/")

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
    else:
        print("[!] 해당 코드 블럭은 Single 모드 시각화에 최적화되어 있습니다. (Batch 처리는 이전과 동일)")

if __name__ == '__main__': main()
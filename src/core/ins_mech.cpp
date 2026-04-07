#include "core/eskf.h"

using namespace std;
using namespace Eigen;

/**
 * 基于相邻两帧 IMU 增量的惯导传播。
 * 输出传播后的名义状态、姿态矩阵与中间量（比力、角速度）。
 */
PropagationResult InsMech::Propagate(const State &state, const ImuData &imu_prev,
                                     const ImuData &imu_curr) {
  return BuildNominalPropagation(state, imu_prev, imu_curr);
}

/**
 * 构建离散化过程模型 Phi 与 Qd。
 * 使用NED系下的完整误差状态微分方程（参考kf-gins-docs.tex）。
 *
 * 状态顺序: [δr^n, δv^n, φ, ba, bg, sg, sa, δk_odo, α, δℓ_odo, δℓ_gnss]
 *
 * 误差微分方程（NED系）：
 *   δṙ^n = -(ω_en^n ×) δr^n + δv^n - (v^n ×) δθ
 *   δv̇^n = C_b^n δf^b + (f^n ×) φ - (2ω_ie^n + ω_en^n) × δv^n + δg^n
 *   φ̇ = -(ω_in^n ×) φ - C_b^n δω_ib^b + δω_in^n
 *
 * @param C_bn 姿态矩阵 C_b^n (body -> NED)
 * @param f_b_corr 机体系修正后比力
 * @param omega_ib_b_corr 机体系修正后角速度
 * @param f_b_unbiased 机体系去零偏、未做比例因子修正的比力
 * @param omega_ib_b_unbiased 机体系去零偏、未做比例因子修正的角速度
 * @param sf_a 加速度计比例因子修正系数 (1+sa)^-1
 * @param sf_g 陀螺比例因子修正系数 (1+sg)^-1
 * @param v_ned NED速度
 * @param lat 纬度（弧度）
 * @param h 高度（米）
 * @param dt 时间步长
 * @param np 噪声参数
 * @param Phi 输出状态转移矩阵
 * @param Qd 输出离散过程噪声
 */
void InsMech::BuildProcessModel(const Matrix3d &C_bn,
                                const Vector3d &f_b_corr,
                                const Vector3d &omega_ib_b_corr,
                                const Vector3d &f_b_unbiased,
                                const Vector3d &omega_ib_b_unbiased,
                                const Vector3d &sf_a,
                                const Vector3d &sf_g,
                                const Vector3d &v_ned,
                                double lat, double h, double dt,
                                const NoiseParams &np,
                                Matrix<double, kStateDim, kStateDim> &Phi,
                                Matrix<double, kStateDim, kStateDim> &Qd,
                                const InEkfManager *inekf) {
  ProcessModelResolvedInput input;
  input.semantics = (inekf != nullptr)
                        ? BuildProcessSemanticsFromInEkfConfig(*inekf)
                        : BuildStandardEskfSemantics();
  input.noise = np;
  input.C_bn = C_bn;
  input.f_b_corr = f_b_corr;
  input.omega_ib_b_corr = omega_ib_b_corr;
  input.f_b_unbiased = f_b_unbiased;
  input.omega_ib_b_unbiased = omega_ib_b_unbiased;
  input.sf_a = sf_a;
  input.sf_g = sf_g;
  input.v_ned = v_ned;
  input.lat = lat;
  input.h = h;
  input.dt = dt;
  if (inekf != nullptr) {
    input.ri_vel_gyro_noise_mode = inekf->ri_vel_gyro_noise_mode;
  }

  const ProcessLinearization linearization = BuildProcessLinearization(input);
  Phi = linearization.Phi;
  Qd = linearization.Qd;
}

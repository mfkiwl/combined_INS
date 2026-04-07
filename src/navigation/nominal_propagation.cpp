#include "navigation/nominal_propagation.h"

#include <cmath>

#include "utils/math_utils.h"

using namespace std;
using namespace Eigen;

namespace {

constexpr double kOmegaEarth = 7.292115e-5;
const Vector3d kOmegaIeE(0.0, 0.0, kOmegaEarth);
constexpr double kMinDt = 1e-9;

}  // namespace

PropagationResult BuildNominalPropagation(const State &state,
                                          const ImuData &imu_prev,
                                          const ImuData &imu_curr) {
  PropagationResult out;
  out.state = state;
  out.Cbn = QuatToRot(state.q);

  const double dt = imu_curr.dt;
  if (dt <= kMinDt) {
    return out;
  }

  const Vector3d omega_ie_b = out.Cbn.transpose() * kOmegaIeE;
  out.omega_ie_b = omega_ie_b;

  const Vector3d dtheta_bc_prev = imu_prev.dtheta - state.bg * imu_prev.dt;
  const Vector3d dtheta_bc_curr = imu_curr.dtheta - state.bg * dt;
  const Vector3d dvel_bc_prev = imu_prev.dvel - state.ba * imu_prev.dt;
  const Vector3d dvel_bc_curr = imu_curr.dvel - state.ba * dt;

  const Vector3d sf_g = (Vector3d::Ones() + state.sg).cwiseInverse();
  const Vector3d sf_a = (Vector3d::Ones() + state.sa).cwiseInverse();

  const Vector3d dtheta_prev_imu_corr = sf_g.cwiseProduct(dtheta_bc_prev);
  const Vector3d dtheta_curr_imu_corr = sf_g.cwiseProduct(dtheta_bc_curr);
  const Vector3d dtheta_prev = dtheta_prev_imu_corr - omega_ie_b * imu_prev.dt;
  const Vector3d dtheta_curr = dtheta_curr_imu_corr - omega_ie_b * dt;
  const Vector3d dvel_prev = sf_a.cwiseProduct(dvel_bc_prev);
  const Vector3d dvel_curr = sf_a.cwiseProduct(dvel_bc_curr);

  out.dtheta_prev_imu_corr = dtheta_prev_imu_corr;
  out.dtheta_curr_imu_corr = dtheta_curr_imu_corr;
  out.dtheta_prev_corr = dtheta_prev;
  out.dtheta_curr_corr = dtheta_curr;
  out.dvel_prev_corr = dvel_prev;
  out.dvel_curr_corr = dvel_curr;

  Vector3d coning = dtheta_curr;
  Vector3d sculling = dvel_curr + 0.5 * dtheta_curr.cross(dvel_curr);
  if (imu_prev.dt > kMinDt) {
    coning += dtheta_prev.cross(dtheta_curr) / 12.0;
    sculling +=
        (dtheta_prev.cross(dvel_curr) + dvel_prev.cross(dtheta_curr)) / 12.0;
  }
  out.coning = coning;
  out.sculling = sculling;

  const Vector4d dq_mid = QuatFromSmallAngle(coning * 0.5);
  const Vector4d dq = QuatFromSmallAngle(coning);
  const Vector4d q_mid = NormalizeQuat(QuatMultiply(state.q, dq_mid));
  const Matrix3d R_mid = QuatToRot(q_mid);

  State next = state;
  next.q = NormalizeQuat(QuatMultiply(state.q, dq));
  const Matrix3d Cbn_next = QuatToRot(next.q);
  const Vector3d dv_nav = R_mid * sculling;
  const Vector3d dv_nav_prev_att = out.Cbn * sculling;
  const Vector3d gravity_e = GravityEcef(state.p);
  const Vector3d coriolis = -2.0 * kOmegaIeE.cross(state.v);
  const Vector3d gravity_dt = gravity_e * dt;
  const Vector3d coriolis_dt = coriolis * dt;
  const Vector3d v_next = state.v + dv_nav + gravity_dt + coriolis_dt;
  next.v = v_next;
  next.p = state.p + 0.5 * (state.v + v_next) * dt;

  out.state = next;
  out.Cbn = Cbn_next;
  out.f_b = sculling / dt;
  out.omega_b = coning / dt;
  out.dv_nav = dv_nav;
  out.dv_nav_prev_att = dv_nav_prev_att;
  out.gravity_dt = gravity_dt;
  out.coriolis_dt = coriolis_dt;
  return out;
}

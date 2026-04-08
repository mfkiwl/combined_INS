#include "fusion_runtime_internal.h"

#include <cmath>
#include <iomanip>
#include <iostream>

#include "utils/math_utils.h"

using namespace Eigen;
using namespace fusion_runtime;
using namespace std;

bool fusion_runtime::ApplyRuntimeTruthAnchor(EskfEngine &engine,
                                             const TruthData &truth,
                                             const InitConfig &init, double t,
                                             int &cursor) {
  Vector3d p_truth = Vector3d::Zero();
  Vector3d v_truth = Vector3d::Zero();
  Vector4d q_truth(1.0, 0.0, 0.0, 0.0);
  if (!InterpolateTruthPva(truth, t, cursor, p_truth, v_truth, q_truth)) {
    return false;
  }

  State anchored_state = engine.state();
  Matrix<double, kStateDim, kStateDim> anchored_cov = engine.cov();
  auto anchor_block = [&](int start_idx, double variance) {
    for (int idx = start_idx; idx < start_idx + 3; ++idx) {
      anchored_cov.row(idx).setZero();
      anchored_cov.col(idx).setZero();
      anchored_cov(idx, idx) = variance;
    }
  };
  if (init.runtime_truth_anchor_position) {
    anchored_state.p = p_truth;
    anchor_block(StateIdx::kPos, kTruthAnchorPosVar);
  }
  if (init.runtime_truth_anchor_velocity) {
    anchored_state.v = v_truth;
    anchor_block(StateIdx::kVel, kTruthAnchorVelVar);
  }
  if (init.runtime_truth_anchor_attitude) {
    anchored_state.q = q_truth;
    anchor_block(StateIdx::kAtt, kTruthAnchorAttVar);
  }

  engine.OverrideStateAndCov(anchored_state, anchored_cov);
  return true;
}

bool fusion_runtime::ApplyRuntimePhaseEntryOverrides(
    EskfEngine &engine, const RuntimePhaseConfig &phase,
    const std::array<bool, kStateDim> &target_mask,
    const ConstraintConfig &effective_constraints, double t_now) {
  State overridden_state = engine.state();
  Matrix<double, kStateDim, kStateDim> overridden_cov = engine.cov();
  bool touched = false;
  const auto reset_cov_scalar = [&](int idx, double variance) {
    overridden_cov.row(idx).setZero();
    overridden_cov.col(idx).setZero();
    overridden_cov(idx, idx) = variance;
    touched = true;
  };
  const auto reset_cov_vec3 = [&](int start_idx, const Vector3d &variance) {
    for (int axis = 0; axis < 3; ++axis) {
      reset_cov_scalar(start_idx + axis, variance(axis));
    }
  };

  const RuntimePhaseEntryInitOverride &init_override =
      phase.phase_entry_init_overrides;
  if (init_override.has_ba0) {
    overridden_state.ba = init_override.ba0;
    touched = true;
  }
  if (init_override.has_bg0) {
    overridden_state.bg = init_override.bg0;
    touched = true;
  }
  if (init_override.has_sg0) {
    overridden_state.sg = init_override.sg0;
    touched = true;
  }
  if (init_override.has_sa0) {
    overridden_state.sa = init_override.sa0;
    touched = true;
  }
  if (init_override.has_odo_scale) {
    overridden_state.odo_scale = init_override.odo_scale;
    if (effective_constraints.enable_odo && overridden_state.odo_scale <= 0.0) {
      overridden_state.odo_scale = 1.0;
    }
    touched = true;
  }
  if (init_override.has_mounting_roll0) {
    overridden_state.mounting_roll = init_override.mounting_roll0 * kDegToRad;
    touched = true;
  }
  if (init_override.has_mounting_pitch0) {
    overridden_state.mounting_pitch = init_override.mounting_pitch0 * kDegToRad;
    touched = true;
  }
  if (init_override.has_mounting_yaw0) {
    overridden_state.mounting_yaw = init_override.mounting_yaw0 * kDegToRad;
    touched = true;
  }
  if (init_override.has_lever_arm0) {
    overridden_state.lever_arm = init_override.lever_arm0;
    touched = true;
  }
  if (init_override.has_gnss_lever_arm0) {
    overridden_state.gnss_lever_arm = init_override.gnss_lever_arm0;
    touched = true;
  }

  const RuntimePhaseEntryStdOverride &std_override =
      phase.phase_entry_std_overrides;
  if (std_override.has_std_ba) {
    reset_cov_vec3(StateIdx::kBa, std_override.std_ba.array().square().matrix());
  }
  if (std_override.has_std_bg) {
    reset_cov_vec3(StateIdx::kBg, std_override.std_bg.array().square().matrix());
  }
  if (std_override.has_std_sg) {
    reset_cov_vec3(StateIdx::kSg, std_override.std_sg.array().square().matrix());
  }
  if (std_override.has_std_sa) {
    reset_cov_vec3(StateIdx::kSa, std_override.std_sa.array().square().matrix());
  }
  if (std_override.has_std_odo_scale) {
    reset_cov_scalar(StateIdx::kOdoScale,
                     std_override.std_odo_scale * std_override.std_odo_scale);
  }
  if (std_override.has_std_mounting_roll) {
    const double var = std::pow(std_override.std_mounting_roll * kDegToRad, 2);
    reset_cov_scalar(StateIdx::kMountRoll, var);
  }
  if (std_override.has_std_mounting_pitch) {
    const double var = std::pow(std_override.std_mounting_pitch * kDegToRad, 2);
    reset_cov_scalar(StateIdx::kMountPitch, var);
  }
  if (std_override.has_std_mounting_yaw) {
    const double var = std::pow(std_override.std_mounting_yaw * kDegToRad, 2);
    reset_cov_scalar(StateIdx::kMountYaw, var);
  }
  if (std_override.has_std_lever_arm) {
    reset_cov_vec3(StateIdx::kLever,
                   std_override.std_lever_arm.array().square().matrix());
  }
  if (std_override.has_std_gnss_lever_arm) {
    reset_cov_vec3(StateIdx::kGnssLever,
                   std_override.std_gnss_lever_arm.array().square().matrix());
  }

  if (!touched) {
    return false;
  }
  engine.SetStateMask(target_mask);
  engine.OverrideStateAndCov(overridden_state, overridden_cov);
  cout << "[Runtime] phase-entry override t=" << fixed << setprecision(3)
       << t_now << " phase=" << phase.name << "\n";
  return true;
}

bool fusion_runtime::ApplyDebugSeedBeforeFirstNhc(EskfEngine &engine,
                                                  const ConstraintConfig &cfg,
                                                  bool &already_applied) {
  if (already_applied) {
    return false;
  }

  const double seed_mount_yaw_bgz_cov =
      cfg.debug_seed_mount_yaw_bgz_cov_before_first_nhc;
  const double seed_bg_z = cfg.debug_seed_bg_z_before_first_nhc;
  const bool seed_bg_z_att_cov =
      cfg.debug_seed_bg_z_att_cov_before_first_nhc.allFinite();
  const bool seed_mount_yaw_bgz_cov_enabled =
      std::isfinite(seed_mount_yaw_bgz_cov);
  const bool seed_bg_z_enabled = std::isfinite(seed_bg_z);
  if (!seed_mount_yaw_bgz_cov_enabled && !seed_bg_z_enabled &&
      !seed_bg_z_att_cov) {
    return false;
  }

  State seeded_state = engine.state();
  Matrix<double, kStateDim, kStateDim> seeded_cov = engine.cov();
  if (seed_mount_yaw_bgz_cov_enabled) {
    seeded_cov(StateIdx::kMountYaw, StateIdx::kBg + 2) =
        seed_mount_yaw_bgz_cov;
    seeded_cov(StateIdx::kBg + 2, StateIdx::kMountYaw) =
        seed_mount_yaw_bgz_cov;
  }
  if (seed_bg_z_att_cov) {
    for (int axis = 0; axis < 3; ++axis) {
      seeded_cov(StateIdx::kBg + 2, StateIdx::kAtt + axis) =
          cfg.debug_seed_bg_z_att_cov_before_first_nhc(axis);
      seeded_cov(StateIdx::kAtt + axis, StateIdx::kBg + 2) =
          cfg.debug_seed_bg_z_att_cov_before_first_nhc(axis);
    }
  }
  if (seed_bg_z_enabled) {
    seeded_state.bg.z() = seed_bg_z;
  }
  engine.OverrideStateAndCov(seeded_state, seeded_cov);
  already_applied = true;

  const double mount_var = std::max(
      0.0, seeded_cov(StateIdx::kMountYaw, StateIdx::kMountYaw));
  const double bg_var =
      std::max(0.0, seeded_cov(StateIdx::kBg + 2, StateIdx::kBg + 2));
  double corr = 0.0;
  if (seed_mount_yaw_bgz_cov_enabled && mount_var > 0.0 && bg_var > 0.0) {
    corr = seed_mount_yaw_bgz_cov / std::sqrt(mount_var * bg_var);
  }
  cout << "[Debug] Seed before first NHC: "
       << "P(mount_yaw,bg_z)=" << seed_mount_yaw_bgz_cov
       << " corr=" << corr
       << " bg_z=" << seed_bg_z
       << " P(bg_z,att_xyz)="
       << cfg.debug_seed_bg_z_att_cov_before_first_nhc.transpose() << "\n";
  return true;
}

bool fusion_runtime::ApplyDebugResetBgzStateAndCov(EskfEngine &engine,
                                                   const ConstraintConfig &cfg,
                                                   double t,
                                                   double time_tolerance,
                                                   bool &already_applied) {
  if (already_applied) {
    return false;
  }
  if (!std::isfinite(cfg.debug_reset_bg_z_state_and_cov_after_time) ||
      !std::isfinite(cfg.debug_reset_bg_z_value) ||
      t + time_tolerance < cfg.debug_reset_bg_z_state_and_cov_after_time) {
    return false;
  }

  State reset_state = engine.state();
  Matrix<double, kStateDim, kStateDim> reset_cov = engine.cov();
  const int bg_z_idx = StateIdx::kBg + 2;
  const double bg_z_before = reset_state.bg.z();
  const double bg_var_before = reset_cov(bg_z_idx, bg_z_idx);
  double offdiag_energy = 0.0;
  for (int idx = 0; idx < kStateDim; ++idx) {
    if (idx == bg_z_idx) {
      continue;
    }
    offdiag_energy += reset_cov(bg_z_idx, idx) * reset_cov(bg_z_idx, idx);
    reset_cov(bg_z_idx, idx) = 0.0;
    reset_cov(idx, bg_z_idx) = 0.0;
  }
  reset_cov(bg_z_idx, bg_z_idx) =
      std::isfinite(bg_var_before) ? std::max(0.0, bg_var_before) : 0.0;
  reset_state.bg.z() = cfg.debug_reset_bg_z_value;
  engine.OverrideStateAndCov(reset_state, reset_cov);
  already_applied = true;

  cout << "[Debug] Mid-run bg_z reset applied at t=" << fixed
       << setprecision(6) << t
       << " target_t=" << cfg.debug_reset_bg_z_state_and_cov_after_time
       << " bg_z_before=" << bg_z_before
       << " bg_z_after=" << cfg.debug_reset_bg_z_value
       << " bg_var=" << reset_cov(bg_z_idx, bg_z_idx)
       << " cleared_cross_cov_norm=" << std::sqrt(offdiag_energy) << "\n";
  return true;
}

bool fusion_runtime::IsWeakExcitation(const State &state, const ImuData &imu,
                                      const ConstraintConfig &cfg) {
  if (!cfg.freeze_extrinsics_when_weak_excitation || imu.dt <= 1e-9) {
    return false;
  }
  const Matrix3d Cbn = QuatToRot(state.q);
  const Vector3d v_b = Cbn.transpose() * state.v;
  const Vector3d omega = imu.dtheta / imu.dt;
  const Vector3d acc = imu.dvel / imu.dt;
  return std::abs(v_b.x()) < cfg.excitation_min_speed &&
         std::abs(omega.z()) < cfg.excitation_min_yaw_rate &&
         std::abs(acc.y()) < cfg.excitation_min_lateral_acc;
}

bool fusion_runtime::MeetsWeakExcitationThresholds(
    const State &state, const ImuData &imu, const ConstraintConfig &cfg) {
  if (imu.dt <= 1e-9) {
    return false;
  }
  const Matrix3d Cbn = QuatToRot(state.q);
  const Vector3d v_b = Cbn.transpose() * state.v;
  const Vector3d omega = imu.dtheta / imu.dt;
  const Vector3d acc = imu.dvel / imu.dt;
  return std::abs(v_b.x()) < cfg.excitation_min_speed &&
         std::abs(omega.z()) < cfg.excitation_min_yaw_rate &&
         std::abs(acc.y()) < cfg.excitation_min_lateral_acc;
}

bool fusion_runtime::IsTimeInWindow(double t, double start_time,
                                    double end_time, double tol) {
  return std::isfinite(start_time) && std::isfinite(end_time) &&
         t >= start_time - tol && t <= end_time + tol;
}

BgzObservabilityGateInfo fusion_runtime::ComputeBgzObservabilityGateInfo(
    const State &state, const ImuData &imu, const ConstraintConfig &cfg) {
  BgzObservabilityGateInfo info;
  if (imu.dt <= 1e-9) {
    return info;
  }

  const Matrix3d Cbn = QuatToRot(state.q);
  const Vector3d v_b = Cbn.transpose() * state.v;
  const Vector3d omega = imu.dtheta / imu.dt;
  const Vector3d acc = imu.dvel / imu.dt;
  const double yaw_rate_min = cfg.bgz_gate_yaw_rate_min_deg_s * kDegToRad;

  info.forward_speed_abs = std::abs(v_b.x());
  info.yaw_rate_abs = std::abs(omega.z());
  info.lateral_acc_abs = std::abs(acc.y());

  const double speed_score = std::clamp(
      info.forward_speed_abs /
          std::max(1e-12, cfg.bgz_gate_forward_speed_min),
      0.0, 1.0);
  const double yaw_score = std::clamp(
      info.yaw_rate_abs / std::max(1e-12, yaw_rate_min), 0.0, 1.0);
  const double lat_score = std::clamp(
      info.lateral_acc_abs /
          std::max(1e-12, cfg.bgz_gate_lateral_acc_min),
      0.0, 1.0);
  const double turn_score = std::max(yaw_score, lat_score);
  const double raw_gate = speed_score * turn_score;
  info.gate_scale =
      std::clamp(std::max(cfg.bgz_gate_min_scale, raw_gate), 0.0, 1.0);
  return info;
}

bool fusion_runtime::ApplyBgzCovarianceForgettingIfNeeded(
    EskfEngine &engine, const ImuData &imu, const ConstraintConfig &cfg) {
  if (!cfg.enable_bgz_covariance_forgetting || imu.dt <= 1e-9) {
    return false;
  }
  const BgzObservabilityGateInfo gate =
      ComputeBgzObservabilityGateInfo(engine.state(), imu, cfg);
  if (gate.gate_scale >= 1.0 - 1.0e-12) {
    return false;
  }

  const double tau = std::max(1.0e-12, cfg.bgz_cov_forgetting_tau_s);
  const double decay = std::exp(-(1.0 - gate.gate_scale) * imu.dt / tau);
  if (decay >= 1.0 - 1.0e-12) {
    return false;
  }

  Matrix<double, kStateDim, kStateDim> P = engine.cov();
  const int bg_z_idx = StateIdx::kBg + 2;
  for (int idx = 0; idx < kStateDim; ++idx) {
    if (idx == bg_z_idx) {
      continue;
    }
    P(bg_z_idx, idx) *= decay;
    P(idx, bg_z_idx) *= decay;
  }
  engine.OverrideStateAndCov(engine.state(), P);
  return true;
}

StateMask fusion_runtime::BuildStateMask(const StateAblationConfig &cfg) {
  StateMask mask;
  mask.fill(true);
  auto disable_range = [&](int start, int len) {
    for (int i = 0; i < len; ++i) {
      mask[start + i] = false;
    }
  };
  if (cfg.disable_accel_bias) {
    disable_range(StateIdx::kBa, 3);
  }
  if (cfg.disable_gyro_bias) {
    disable_range(StateIdx::kBg, 3);
  }
  if (cfg.disable_gyro_scale) {
    disable_range(StateIdx::kSg, 3);
  }
  if (cfg.disable_accel_scale) {
    disable_range(StateIdx::kSa, 3);
  }
  if (cfg.disable_odo_scale) {
    disable_range(StateIdx::kOdoScale, 1);
  }
  if (cfg.disable_mounting) {
    disable_range(StateIdx::kMountRoll, 3);
  }
  if (cfg.disable_mounting_roll) {
    disable_range(StateIdx::kMountRoll, 1);
  }
  if (cfg.disable_mounting_pitch) {
    disable_range(StateIdx::kMountPitch, 1);
  }
  if (cfg.disable_mounting_yaw) {
    disable_range(StateIdx::kMountYaw, 1);
  }
  if (cfg.disable_odo_lever_arm) {
    disable_range(StateIdx::kLever, 3);
  }
  if (cfg.disable_gnss_lever_arm) {
    disable_range(StateIdx::kGnssLever, 3);
  }
  if (cfg.disable_gnss_lever_z) {
    disable_range(StateIdx::kGnssLever + 2, 1);
  }
  return mask;
}

StateMask fusion_runtime::BuildGnssPosNonPositionMask() {
  StateMask mask;
  mask.fill(true);
  for (int axis = 0; axis < 3; ++axis) {
    mask[StateIdx::kPos + axis] = false;
  }
  return mask;
}

StateMask fusion_runtime::BuildGnssPosPositionOnlyMask() {
  StateMask mask;
  mask.fill(false);
  for (int axis = 0; axis < 3; ++axis) {
    mask[StateIdx::kPos + axis] = true;
  }
  return mask;
}

StateAblationConfig fusion_runtime::MergeAblationConfig(
    const StateAblationConfig &base, const StateAblationConfig &extra) {
  StateAblationConfig out = base;
  out.disable_gnss_lever_arm =
      out.disable_gnss_lever_arm || extra.disable_gnss_lever_arm;
  out.disable_gnss_lever_z =
      out.disable_gnss_lever_z || extra.disable_gnss_lever_z;
  out.disable_odo_lever_arm =
      out.disable_odo_lever_arm || extra.disable_odo_lever_arm;
  out.disable_odo_scale = out.disable_odo_scale || extra.disable_odo_scale;
  out.disable_accel_bias =
      out.disable_accel_bias || extra.disable_accel_bias;
  out.disable_gyro_bias = out.disable_gyro_bias || extra.disable_gyro_bias;
  out.disable_gyro_scale =
      out.disable_gyro_scale || extra.disable_gyro_scale;
  out.disable_accel_scale =
      out.disable_accel_scale || extra.disable_accel_scale;
  out.disable_mounting = out.disable_mounting || extra.disable_mounting;
  out.disable_mounting_roll =
      out.disable_mounting_roll || extra.disable_mounting_roll;
  out.disable_mounting_pitch =
      out.disable_mounting_pitch || extra.disable_mounting_pitch;
  out.disable_mounting_yaw =
      out.disable_mounting_yaw || extra.disable_mounting_yaw;
  return out;
}

void fusion_runtime::ApplyAblationToNoise(NoiseParams &noise,
                                          const StateAblationConfig &cfg) {
  if (cfg.disable_accel_bias) {
    noise.sigma_ba = 0.0;
    noise.sigma_ba_vec.setZero();
  }
  if (cfg.disable_gyro_bias) {
    noise.sigma_bg = 0.0;
    noise.sigma_bg_vec.setZero();
  }
  if (cfg.disable_gyro_scale) {
    noise.sigma_sg = 0.0;
    noise.sigma_sg_vec.setZero();
  }
  if (cfg.disable_accel_scale) {
    noise.sigma_sa = 0.0;
    noise.sigma_sa_vec.setZero();
  }
  if (cfg.disable_odo_scale) {
    noise.sigma_odo_scale = 0.0;
  }
  if (cfg.disable_mounting) {
    noise.sigma_mounting = 0.0;
    noise.sigma_mounting_roll = 0.0;
    noise.sigma_mounting_pitch = 0.0;
    noise.sigma_mounting_yaw = 0.0;
  } else {
    if (cfg.disable_mounting_roll) {
      noise.sigma_mounting_roll = 0.0;
    }
    if (cfg.disable_mounting_pitch) {
      noise.sigma_mounting_pitch = 0.0;
    }
    if (cfg.disable_mounting_yaw) {
      noise.sigma_mounting_yaw = 0.0;
    }
  }
  if (cfg.disable_odo_lever_arm) {
    noise.sigma_lever_arm = 0.0;
    noise.sigma_lever_arm_vec.setZero();
  }
  if (cfg.disable_gnss_lever_arm) {
    noise.sigma_gnss_lever_arm = 0.0;
    noise.sigma_gnss_lever_arm_vec.setZero();
  } else if (cfg.disable_gnss_lever_z &&
             (noise.sigma_gnss_lever_arm_vec.array() >= 0.0).all()) {
    noise.sigma_gnss_lever_arm_vec.z() = 0.0;
  }
}

void fusion_runtime::ApplyRuntimeNoiseOverride(
    NoiseParams &noise, const RuntimeNoiseOverride &override_cfg) {
  if (std::isfinite(override_cfg.sigma_acc)) {
    noise.sigma_acc = override_cfg.sigma_acc;
  }
  if (std::isfinite(override_cfg.sigma_gyro)) {
    noise.sigma_gyro = override_cfg.sigma_gyro;
  }
  if (std::isfinite(override_cfg.sigma_ba)) {
    noise.sigma_ba = override_cfg.sigma_ba;
  }
  if (std::isfinite(override_cfg.sigma_bg)) {
    noise.sigma_bg = override_cfg.sigma_bg;
  }
  if (std::isfinite(override_cfg.sigma_sg)) {
    noise.sigma_sg = override_cfg.sigma_sg;
  }
  if (std::isfinite(override_cfg.sigma_sa)) {
    noise.sigma_sa = override_cfg.sigma_sa;
  }
  if (override_cfg.sigma_ba_vec.allFinite()) {
    noise.sigma_ba_vec = override_cfg.sigma_ba_vec;
  }
  if (override_cfg.sigma_bg_vec.allFinite()) {
    noise.sigma_bg_vec = override_cfg.sigma_bg_vec;
  }
  if (override_cfg.sigma_sg_vec.allFinite()) {
    noise.sigma_sg_vec = override_cfg.sigma_sg_vec;
  }
  if (override_cfg.sigma_sa_vec.allFinite()) {
    noise.sigma_sa_vec = override_cfg.sigma_sa_vec;
  }
  if (std::isfinite(override_cfg.sigma_odo_scale)) {
    noise.sigma_odo_scale = override_cfg.sigma_odo_scale;
  }
  if (std::isfinite(override_cfg.sigma_mounting)) {
    noise.sigma_mounting = override_cfg.sigma_mounting;
  }
  if (std::isfinite(override_cfg.sigma_mounting_roll)) {
    noise.sigma_mounting_roll = override_cfg.sigma_mounting_roll;
  }
  if (std::isfinite(override_cfg.sigma_mounting_pitch)) {
    noise.sigma_mounting_pitch = override_cfg.sigma_mounting_pitch;
  }
  if (std::isfinite(override_cfg.sigma_mounting_yaw)) {
    noise.sigma_mounting_yaw = override_cfg.sigma_mounting_yaw;
  }
  if (std::isfinite(override_cfg.sigma_lever_arm)) {
    noise.sigma_lever_arm = override_cfg.sigma_lever_arm;
  }
  if (std::isfinite(override_cfg.sigma_gnss_lever_arm)) {
    noise.sigma_gnss_lever_arm = override_cfg.sigma_gnss_lever_arm;
  }
  if (override_cfg.sigma_lever_arm_vec.allFinite()) {
    noise.sigma_lever_arm_vec = override_cfg.sigma_lever_arm_vec;
  }
  if (override_cfg.sigma_gnss_lever_arm_vec.allFinite()) {
    noise.sigma_gnss_lever_arm_vec = override_cfg.sigma_gnss_lever_arm_vec;
  }
  if (std::isfinite(override_cfg.sigma_uwb)) {
    noise.sigma_uwb = override_cfg.sigma_uwb;
  }
  if (std::isfinite(override_cfg.sigma_gnss_pos)) {
    noise.sigma_gnss_pos = override_cfg.sigma_gnss_pos;
  }
  if (std::isfinite(override_cfg.markov_corr_time)) {
    noise.markov_corr_time = override_cfg.markov_corr_time;
  }
  if (override_cfg.has_disable_nominal_ba_bg_decay) {
    noise.disable_nominal_ba_bg_decay =
        override_cfg.disable_nominal_ba_bg_decay;
  }
}

ConstraintConfig fusion_runtime::ApplyRuntimeConstraintOverride(
    const ConstraintConfig &base,
    const RuntimeConstraintOverride &override_cfg) {
  ConstraintConfig out = base;
  if (override_cfg.has_enable_nhc) {
    out.enable_nhc = override_cfg.enable_nhc;
  }
  if (override_cfg.has_enable_odo) {
    out.enable_odo = override_cfg.enable_odo;
  }
  if (override_cfg.has_enable_covariance_floor) {
    out.enable_covariance_floor = override_cfg.enable_covariance_floor;
  }
  if (override_cfg.has_enable_nis_gating) {
    out.enable_nis_gating = override_cfg.enable_nis_gating;
  }
  if (override_cfg.has_odo_nis_gate_prob) {
    out.odo_nis_gate_prob = override_cfg.odo_nis_gate_prob;
  }
  if (override_cfg.has_nhc_nis_gate_prob) {
    out.nhc_nis_gate_prob = override_cfg.nhc_nis_gate_prob;
  }
  if (override_cfg.has_p_floor_odo_scale_var) {
    out.p_floor_odo_scale_var = override_cfg.p_floor_odo_scale_var;
  }
  if (override_cfg.has_p_floor_lever_arm_vec) {
    out.p_floor_lever_arm_vec = override_cfg.p_floor_lever_arm_vec;
  }
  if (override_cfg.has_p_floor_mounting_deg) {
    out.p_floor_mounting_deg = override_cfg.p_floor_mounting_deg;
  }
  return out;
}

std::string fusion_runtime::ComputeEffectiveGnssPosUpdateMode(
    const FusionOptions &options, double t_now) {
  std::string effective = options.gnss_pos_update_mode;
  for (const auto &phase : options.runtime_phases) {
    if (!phase.enabled ||
        !IsTimeInWindow(t_now, phase.start_time, phase.end_time,
                        options.gating.time_tolerance)) {
      continue;
    }
    if (phase.constraints.has_gnss_pos_update_mode) {
      effective = phase.constraints.gnss_pos_update_mode;
    }
  }
  return effective;
}

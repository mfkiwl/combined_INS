#include "fusion_runtime_internal.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>

using namespace Eigen;
using namespace fusion_runtime;
using namespace std;

namespace {

enum class NhcAdmissionVelocitySource {
  kBody,
  kWheelBody,
  kVehicle,
};

struct NhcAdmissionKinematics {
  Vector3d v_b = Vector3d::Zero();
  Vector3d v_wheel_b = Vector3d::Zero();
  Vector3d v_v = Vector3d::Zero();
};

struct NhcAdmissionDecision {
  bool accept = true;
  bool below_forward_speed = false;
  bool exceed_lateral_vertical_limit = false;
};

NhcAdmissionVelocitySource ResolveNhcAdmissionVelocitySource(
    const ConstraintConfig &cfg) {
  if (cfg.nhc_admission_velocity_source == "v_wheel_b") {
    return NhcAdmissionVelocitySource::kWheelBody;
  }
  if (cfg.nhc_admission_velocity_source == "v_v") {
    return NhcAdmissionVelocitySource::kVehicle;
  }
  return NhcAdmissionVelocitySource::kBody;
}

const char *NhcAdmissionVelocitySourceName(
    NhcAdmissionVelocitySource source) {
  switch (source) {
    case NhcAdmissionVelocitySource::kWheelBody:
      return "v_wheel_b";
    case NhcAdmissionVelocitySource::kVehicle:
      return "v_v";
    case NhcAdmissionVelocitySource::kBody:
    default:
      return "v_b";
  }
}

NhcAdmissionKinematics ComputeNhcAdmissionKinematics(
    const State &state, const Matrix3d &C_b_v,
    const Vector3d &omega_ib_b_raw) {
  NhcAdmissionKinematics kin;
  const Llh llh = EcefToLlh(state.p);
  const Matrix3d R_ne = RotNedToEcef(llh);
  const Matrix3d C_bn = R_ne.transpose() * QuatToRot(state.q);
  const Vector3d v_ned = R_ne.transpose() * state.v;
  kin.v_b = C_bn.transpose() * v_ned;

  const Vector3d omega_ie_n = OmegaIeNed(llh.lat);
  const Vector3d omega_en_n = OmegaEnNed(v_ned, llh.lat, llh.h);
  const Vector3d omega_in_n = omega_ie_n + omega_en_n;
  const Vector3d omega_in_b = C_bn.transpose() * omega_in_n;
  const Vector3d omega_ib_unbiased = omega_ib_b_raw - state.bg;
  const Vector3d sf_g = (Vector3d::Ones() + state.sg).cwiseInverse();
  const Vector3d omega_ib_corr = sf_g.cwiseProduct(omega_ib_unbiased);
  const Vector3d omega_nb_b = omega_ib_corr - omega_in_b;

  kin.v_wheel_b = kin.v_b + omega_nb_b.cross(state.lever_arm);
  kin.v_v = C_b_v * kin.v_wheel_b;
  return kin;
}

NhcAdmissionDecision EvaluateNhcAdmissionDecision(const Vector3d &v_ref,
                                                  const ConstraintConfig &cfg) {
  NhcAdmissionDecision decision;
  if (cfg.nhc_disable_below_forward_speed > 0.0) {
    decision.below_forward_speed =
        std::abs(v_ref.x()) < cfg.nhc_disable_below_forward_speed;
  }
  if (cfg.nhc_max_abs_v > 0.0) {
    decision.exceed_lateral_vertical_limit =
        std::abs(v_ref.y()) > cfg.nhc_max_abs_v ||
        std::abs(v_ref.z()) > cfg.nhc_max_abs_v;
  }
  decision.accept = !decision.below_forward_speed &&
                    !decision.exceed_lateral_vertical_limit;
  return decision;
}

Vector3d SelectNhcAdmissionVelocity(const NhcAdmissionKinematics &kin,
                                    NhcAdmissionVelocitySource source) {
  switch (source) {
    case NhcAdmissionVelocitySource::kWheelBody:
      return kin.v_wheel_b;
    case NhcAdmissionVelocitySource::kVehicle:
      return kin.v_v;
    case NhcAdmissionVelocitySource::kBody:
    default:
      return kin.v_b;
  }
}

void LogNhcAdmissionSample(std::ofstream *file, double t,
                           NhcAdmissionVelocitySource selected_source,
                           const NhcAdmissionKinematics &kin,
                           const ConstraintConfig &cfg,
                           const NhcAdmissionDecision &decision_v_b,
                           const NhcAdmissionDecision &decision_v_wheel_b,
                           const NhcAdmissionDecision &decision_v_v) {
  if (file == nullptr || !file->is_open()) {
    return;
  }
  const NhcAdmissionDecision *selected_decision = &decision_v_b;
  switch (selected_source) {
    case NhcAdmissionVelocitySource::kWheelBody:
      selected_decision = &decision_v_wheel_b;
      break;
    case NhcAdmissionVelocitySource::kVehicle:
      selected_decision = &decision_v_v;
      break;
    case NhcAdmissionVelocitySource::kBody:
    default:
      break;
  }
  (*file) << t << ","
          << NhcAdmissionVelocitySourceName(selected_source) << ","
          << (selected_decision->accept ? 1 : 0) << ","
          << (decision_v_b.accept ? 1 : 0) << ","
          << (decision_v_wheel_b.accept ? 1 : 0) << ","
          << (decision_v_v.accept ? 1 : 0) << ","
          << (decision_v_b.below_forward_speed ? 1 : 0) << ","
          << (decision_v_wheel_b.below_forward_speed ? 1 : 0) << ","
          << (decision_v_v.below_forward_speed ? 1 : 0) << ","
          << (decision_v_b.exceed_lateral_vertical_limit ? 1 : 0) << ","
          << (decision_v_wheel_b.exceed_lateral_vertical_limit ? 1 : 0) << ","
          << (decision_v_v.exceed_lateral_vertical_limit ? 1 : 0) << ","
          << cfg.nhc_disable_below_forward_speed << ","
          << cfg.nhc_max_abs_v << ","
          << kin.v_b.x() << "," << kin.v_b.y() << "," << kin.v_b.z() << ","
          << kin.v_wheel_b.x() << "," << kin.v_wheel_b.y() << ","
          << kin.v_wheel_b.z() << ","
          << kin.v_v.x() << "," << kin.v_v.y() << "," << kin.v_v.z()
          << "\n";
}

void ApplyStateGainScaleToKalmanGain(MatrixXd &K,
                                     const StateGainScale *gain_scale) {
  if (gain_scale == nullptr || K.rows() != kStateDim) {
    return;
  }
  for (int i = 0; i < kStateDim; ++i) {
    K.row(i) *= (*gain_scale)[i];
  }
}

void ApplyStateMeasurementGainScaleToKalmanGain(
    MatrixXd &K, const StateMeasurementGainScale *gain_element_scale) {
  if (gain_element_scale == nullptr ||
      gain_element_scale->rows() != K.rows() ||
      gain_element_scale->cols() != K.cols()) {
    return;
  }
  K.array() *= gain_element_scale->array();
}

double InvNormCdf(double p) {
  if (p <= 0.0 || p >= 1.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  static const double a1 = -3.969683028665376e+01;
  static const double a2 = 2.209460984245205e+02;
  static const double a3 = -2.759285104469687e+02;
  static const double a4 = 1.383577518672690e+02;
  static const double a5 = -3.066479806614716e+01;
  static const double a6 = 2.506628277459239e+00;
  static const double b1 = -5.447609879822406e+01;
  static const double b2 = 1.615858368580409e+02;
  static const double b3 = -1.556989798598866e+02;
  static const double b4 = 6.680131188771972e+01;
  static const double b5 = -1.328068155288572e+01;
  static const double c1 = -7.784894002430293e-03;
  static const double c2 = -3.223964580411365e-01;
  static const double c3 = -2.400758277161838e+00;
  static const double c4 = -2.549732539343734e+00;
  static const double c5 = 4.374664141464968e+00;
  static const double c6 = 2.938163982698783e+00;
  static const double d1 = 7.784695709041462e-03;
  static const double d2 = 3.224671290700398e-01;
  static const double d3 = 2.445134137142996e+00;
  static const double d4 = 3.754408661907416e+00;
  static const double plow = 0.02425;
  static const double phigh = 1.0 - plow;

  if (p < plow) {
    const double q = std::sqrt(-2.0 * std::log(p));
    return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
           ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
  }
  if (p > phigh) {
    const double q = std::sqrt(-2.0 * std::log(1.0 - p));
    return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
  }

  const double q = p - 0.5;
  const double r = q * q;
  return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q /
         (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0);
}

double ChiSquareQuantile(int dof, double prob) {
  const double p = std::clamp(prob, 1e-6, 1.0 - 1e-6);
  if (dof <= 0) {
    return 0.0;
  }
  if (dof == 1) {
    const double z = InvNormCdf(0.5 * (1.0 + p));
    return z * z;
  }
  if (dof == 2) {
    return -2.0 * std::log(1.0 - p);
  }
  const double z = InvNormCdf(p);
  const double a = 2.0 / (9.0 * static_cast<double>(dof));
  double term = 1.0 - a + z * std::sqrt(a);
  if (term < 1e-6) {
    term = 1e-6;
  }
  return static_cast<double>(dof) * term * term * term;
}

bool ComputeNis(const EskfEngine &engine, const MatrixXd &H, const MatrixXd &R,
                const VectorXd &y, double &nis_out) {
  const MatrixXd S = H * engine.cov() * H.transpose() + R;
  LDLT<MatrixXd> ldlt(S);
  if (ldlt.info() != Success) {
    return false;
  }
  const VectorXd w = ldlt.solve(y);
  if (ldlt.info() != Success || !w.allFinite()) {
    return false;
  }
  const double nis = y.dot(w);
  if (!std::isfinite(nis) || nis < 0.0) {
    return false;
  }
  nis_out = nis;
  return true;
}

double ComputeRobustWeight(const ConstraintConfig &cfg,
                           double whitened_norm) {
  if (!cfg.enable_robust_weighting) {
    return 1.0;
  }
  const double k = std::max(1e-6, cfg.robust_tuning);
  double w = 1.0;
  if (cfg.robust_kernel == "cauchy") {
    const double r = whitened_norm / k;
    w = 1.0 / (1.0 + r * r);
  } else if (whitened_norm > k) {
    w = k / whitened_norm;
  }
  return std::clamp(w, cfg.robust_min_weight, 1.0);
}

void ZeroExtrinsicSensitivity(MatrixXd &H, bool freeze_scale,
                              bool freeze_mounting, bool freeze_lever) {
  if (freeze_scale && H.cols() > StateIdx::kOdoScale) {
    H.col(StateIdx::kOdoScale).setZero();
  }
  if (freeze_mounting && H.cols() > StateIdx::kMountYaw) {
    H.block(0, StateIdx::kMountRoll, H.rows(), 3).setZero();
  }
  if (freeze_lever && H.cols() > StateIdx::kLever + 2) {
    H.block(0, StateIdx::kLever, H.rows(), 3).setZero();
  }
}

bool CorrectConstraintWithRobustness(
    EskfEngine &engine, const string &tag, double t, const ImuData &imu,
    const ConstraintConfig &cfg, const VectorXd &y, MatrixXd H, MatrixXd R,
    bool freeze_scale, bool freeze_mounting, bool freeze_lever,
    double nis_prob, bool apply_bgz_observability_gate,
    const StateMask *update_mask, DiagnosticsEngine &diag,
    ConstraintUpdateStats &stats) {
  ++stats.seen;

  if (IsWeakExcitation(engine.state(), imu, cfg)) {
    ZeroExtrinsicSensitivity(H, freeze_scale, freeze_mounting, freeze_lever);
  }

  StateGainScale bgz_gate_scale;
  const StateGainScale *bgz_gate_scale_ptr = nullptr;
  const BgzObservabilityGateInfo bgz_gate =
      ComputeBgzObservabilityGateInfo(engine.state(), imu, cfg);
  if (cfg.enable_bgz_observability_gate && apply_bgz_observability_gate &&
      bgz_gate.gate_scale < 1.0 - 1e-12) {
    bgz_gate_scale.fill(1.0);
    bgz_gate_scale[StateIdx::kBg + 2] = bgz_gate.gate_scale;
    bgz_gate_scale_ptr = &bgz_gate_scale;
  }

  double nis_gate = 0.0;
  if (!ComputeNis(engine, H, R, y, nis_gate)) {
    ++stats.rejected_numeric;
    return false;
  }

  const int dof = std::max(1, static_cast<int>(y.size()));
  const double gate = ChiSquareQuantile(dof, nis_prob);
  if (cfg.enable_nis_gating && nis_gate > gate) {
    ++stats.rejected_nis;
    if (cfg.enable_consistency_log &&
        (stats.rejected_nis <= 20 || (stats.rejected_nis % 5000) == 0)) {
      cout << "[Consistency] " << tag << " reject t=" << t
           << " NIS=" << nis_gate << " gate=" << gate << "\n";
    }
    return false;
  }

  const double robust_w =
      ComputeRobustWeight(cfg, std::sqrt(std::max(0.0, nis_gate)));
  MatrixXd R_eff = R;
  if (cfg.enable_robust_weighting) {
    R_eff /= (robust_w * robust_w);
  }

  double nis_update = 0.0;
  (void)ComputeNis(engine, H, R_eff, y, nis_update);

  const bool updated = diag.Correct(engine, tag, t, y, H, R_eff, update_mask,
                                    bgz_gate_scale_ptr, nullptr);
  if (!updated) {
    ++stats.rejected_numeric;
    return false;
  }
  ++stats.accepted;
  stats.nis_sum += nis_gate;
  stats.nis_max = std::max(stats.nis_max, nis_gate);
  stats.robust_weight_sum += robust_w;
  stats.noise_scale_sum += 1.0;
  return true;
}

double SafeCorrFromCov(const Matrix<double, kStateDim, kStateDim> &P, int a,
                       int b) {
  const double pa = P(a, a);
  const double pb = P(b, b);
  if (pa < 1e-30 || pb < 1e-30) {
    return 0.0;
  }
  return P(a, b) / std::sqrt(pa * pb);
}

void MaybeCaptureGnssSplitDebug(FusionDebugCapture *debug_capture,
                                const EskfEngine &engine, const string &tag,
                                double t_meas, double split_t, double tol) {
  if (debug_capture == nullptr ||
      !debug_capture->capture_last_gnss_before_split) {
    return;
  }
  if (!(t_meas <= split_t + tol)) {
    return;
  }
  if (debug_capture->gnss_split_cov.valid &&
      t_meas + tol < debug_capture->gnss_split_cov.t_meas) {
    return;
  }

  const auto &P = engine.cov();
  GnssSplitCovarianceCapture cov_capture;
  cov_capture.valid = true;
  cov_capture.tag = tag;
  cov_capture.split_t = split_t;
  cov_capture.t_meas = t_meas;
  cov_capture.t_state = engine.timestamp();
  cov_capture.P_att_bgz = P.block<3, 1>(StateIdx::kAtt, StateIdx::kBg + 2);
  cov_capture.att_var = P.diagonal().segment<3>(StateIdx::kAtt);
  cov_capture.bgz_var = P(StateIdx::kBg + 2, StateIdx::kBg + 2);
  for (int i = 0; i < 3; ++i) {
    cov_capture.corr_att_bgz(i) =
        SafeCorrFromCov(P, StateIdx::kAtt + i, StateIdx::kBg + 2);
  }
  debug_capture->gnss_split_cov = cov_capture;

  const auto &reset_snapshot = engine.last_inekf_correction();
  if (reset_snapshot.valid) {
    ResetConsistencyCapture reset_capture;
    reset_capture.valid = true;
    reset_capture.tag = tag;
    reset_capture.split_t = split_t;
    reset_capture.t_meas = t_meas;
    reset_capture.t_state = reset_snapshot.t_state;
    reset_capture.P_tilde = reset_snapshot.P_tilde;
    reset_capture.P_after_reset = reset_snapshot.P_after_reset;
    reset_capture.P_after_all = reset_snapshot.P_after_all;
    reset_capture.dx = reset_snapshot.dx;
    reset_capture.covariance_floor_applied =
        reset_snapshot.covariance_floor_applied;
    debug_capture->reset_consistency = reset_capture;
  }
}

bool DetectZupt(const ImuData &imu, const State &state,
                const ConstraintConfig &config) {
  if (imu.dt <= 1e-9) {
    return false;
  }
  const Vector3d omega = imu.dtheta / imu.dt;
  const Vector3d f_b = imu.dvel / imu.dt;
  const double acc_diff = std::abs(f_b.norm() - 9.81);
  if (state.v.norm() > config.zupt_max_speed) {
    return false;
  }
  return omega.norm() < config.zupt_max_gyro &&
         acc_diff < config.zupt_max_acc;
}

}  // namespace

bool fusion_runtime::RunZuptUpdate(EskfEngine &engine, const ImuData &imu,
                                   const ConstraintConfig &cfg,
                                   double &static_duration,
                                   DiagnosticsEngine &diag, double t) {
  if (!cfg.enable_zupt) {
    static_duration = 0.0;
    return false;
  }

  const bool is_static = DetectZupt(imu, engine.state(), cfg);
  const double dt = imu.dt;
  if (is_static && dt > 0.0) {
    static_duration += dt;
  } else if (!is_static) {
    static_duration = 0.0;
  }

  if (is_static && static_duration + 1e-12 >= cfg.zupt_min_duration) {
    const auto model =
        MeasModels::ComputeZuptModel(engine.state(), cfg.sigma_zupt);
    return diag.Correct(engine, "ZUPT", t, model.y, model.H, model.R);
  }
  return false;
}

void fusion_runtime::RunNhcUpdate(EskfEngine &engine, const ImuData &imu,
                                  const ConstraintConfig &cfg,
                                  const Vector3d &mounting_base_rpy,
                                  bool zupt_ready, DiagnosticsEngine &diag,
                                  double t, const InEkfManager *inekf,
                                  ConstraintUpdateStats &stats,
                                  double &last_nhc_update_t,
                                  double nhc_min_interval,
                                  std::ofstream *nhc_admission_log_file) {
  if (!cfg.enable_nhc || (cfg.enable_zupt && zupt_ready)) {
    return;
  }
  if (IsTimeInWindow(t, cfg.debug_nhc_disable_start_time,
                     cfg.debug_nhc_disable_end_time)) {
    return;
  }
  if (nhc_min_interval > 0.0 && last_nhc_update_t > -1e17 &&
      (t - last_nhc_update_t) < nhc_min_interval) {
    return;
  }

  const double dt = imu.dt;
  if (dt <= 1e-9) {
    return;
  }
  const State &state = engine.state();
  if (cfg.disable_nhc_when_weak_excitation &&
      MeetsWeakExcitationThresholds(state, imu, cfg)) {
    return;
  }

  const Vector3d mounting_rpy(mounting_base_rpy.x(),
                              mounting_base_rpy.y() + state.mounting_pitch,
                              mounting_base_rpy.z() + state.mounting_yaw);
  const Matrix3d C_b_v = QuatToRot(RpyToQuat(mounting_rpy)).transpose();
  const Vector3d omega_ib_b_raw = imu.dtheta / dt;
  const NhcAdmissionKinematics admission_kin =
      ComputeNhcAdmissionKinematics(state, C_b_v, omega_ib_b_raw);
  const NhcAdmissionDecision decision_v_b =
      EvaluateNhcAdmissionDecision(admission_kin.v_b, cfg);
  const NhcAdmissionDecision decision_v_wheel_b =
      EvaluateNhcAdmissionDecision(admission_kin.v_wheel_b, cfg);
  const NhcAdmissionDecision decision_v_v =
      EvaluateNhcAdmissionDecision(admission_kin.v_v, cfg);
  const NhcAdmissionVelocitySource selected_source =
      ResolveNhcAdmissionVelocitySource(cfg);
  const Vector3d selected_velocity =
      SelectNhcAdmissionVelocity(admission_kin, selected_source);

  LogNhcAdmissionSample(nhc_admission_log_file, t, selected_source,
                        admission_kin, cfg, decision_v_b, decision_v_wheel_b,
                        decision_v_v);

  const NhcAdmissionDecision *selected_decision = &decision_v_b;
  switch (selected_source) {
    case NhcAdmissionVelocitySource::kWheelBody:
      selected_decision = &decision_v_wheel_b;
      break;
    case NhcAdmissionVelocitySource::kVehicle:
      selected_decision = &decision_v_v;
      break;
    case NhcAdmissionVelocitySource::kBody:
    default:
      break;
  }

  if (selected_decision->below_forward_speed) {
    if (diag.enabled() && !diag.nhc_skip_warned()) {
      cout << "[Warn] NHC skipped at t=" << t
           << " source=" << NhcAdmissionVelocitySourceName(selected_source)
           << " |vx|=" << std::abs(selected_velocity.x())
           << " below forward-speed threshold="
           << cfg.nhc_disable_below_forward_speed << "\n";
    }
    diag.set_nhc_skip_warned(true);
    return;
  }
  if (selected_decision->exceed_lateral_vertical_limit) {
    if (diag.enabled() && !diag.nhc_skip_warned()) {
      cout << "[Warn] NHC skipped at t=" << t
           << " source=" << NhcAdmissionVelocitySourceName(selected_source)
           << " |vy|=" << std::abs(selected_velocity.y())
           << " |vz|=" << std::abs(selected_velocity.z()) << "\n";
    }
    diag.set_nhc_skip_warned(true);
    return;
  }
  diag.set_nhc_skip_warned(false);

  const auto model = MeasModels::ComputeNhcModel(
      state, C_b_v, omega_ib_b_raw, cfg.sigma_nhc_y, cfg.sigma_nhc_z, inekf);
  StateMask nhc_update_mask;
  const StateMask *nhc_update_mask_ptr = nullptr;
  if (cfg.debug_nhc_disable_bgz_state_update) {
    nhc_update_mask.fill(true);
    nhc_update_mask[StateIdx::kBg + 2] = false;
    nhc_update_mask_ptr = &nhc_update_mask;
  }
  if (CorrectConstraintWithRobustness(
          engine, "NHC", t, imu, cfg, model.y, model.H, model.R, false, true,
          true, cfg.nhc_nis_gate_prob, cfg.bgz_gate_apply_to_nhc,
          nhc_update_mask_ptr, diag, stats)) {
    last_nhc_update_t = t;
  }
}

double fusion_runtime::RunOdoUpdate(EskfEngine &engine, const Dataset &dataset,
                                    const ConstraintConfig &cfg,
                                    const GatingConfig &gating,
                                    const Vector3d &mounting_base_rpy,
                                    int &odo_idx, double t_curr,
                                    const ImuData &imu_curr,
                                    DiagnosticsEngine &diag,
                                    const InEkfManager *inekf,
                                    ConstraintUpdateStats &stats,
                                    double &last_odo_update_t,
                                    double odo_min_interval) {
  double last_odo_speed = 0.0;
  if (!cfg.enable_odo) {
    return last_odo_speed;
  }

  while (odo_idx < dataset.odo.rows()) {
    const double t_odo = dataset.odo(odo_idx, 0) + cfg.odo_time_offset;
    if (t_odo > t_curr + gating.time_tolerance) {
      break;
    }

    double odo_vel = dataset.odo(odo_idx, 1);
    bool interpolated_to_curr = false;
    const int next = odo_idx + 1;
    if (next < dataset.odo.rows()) {
      const double t_next = dataset.odo(next, 0) + cfg.odo_time_offset;
      const double dt_odo = t_next - t_odo;
      if (dt_odo > 1e-6 && t_curr >= t_odo && t_curr <= t_next) {
        const double alpha = (t_curr - t_odo) / dt_odo;
        odo_vel += alpha * (dataset.odo(next, 1) - odo_vel);
        interpolated_to_curr = true;
      }
    }
    const double t_meas = interpolated_to_curr ? t_curr : t_odo;
    if (IsTimeInWindow(t_meas, cfg.debug_odo_disable_start_time,
                       cfg.debug_odo_disable_end_time,
                       gating.time_tolerance)) {
      ++odo_idx;
      continue;
    }
    if (odo_min_interval > 0.0 && last_odo_update_t > -1e17 &&
        (t_meas - last_odo_update_t) < odo_min_interval) {
      ++odo_idx;
      continue;
    }
    last_odo_speed = odo_vel;

    const State &state = engine.state();
    if (cfg.disable_odo_when_weak_excitation &&
        MeetsWeakExcitationThresholds(state, imu_curr, cfg)) {
      ++odo_idx;
      continue;
    }
    Vector3d omega_ib_b_raw = Vector3d::Zero();
    if (imu_curr.dt > 1e-9) {
      omega_ib_b_raw = imu_curr.dtheta / imu_curr.dt;
    }

    const Vector3d rpy(mounting_base_rpy.x(),
                       mounting_base_rpy.y() + state.mounting_pitch,
                       mounting_base_rpy.z() + state.mounting_yaw);
    const Matrix3d C_b_v = QuatToRot(RpyToQuat(rpy)).transpose();
    auto model = MeasModels::ComputeOdoModel(state, odo_vel, C_b_v,
                                             omega_ib_b_raw, cfg.sigma_odo,
                                             inekf);
    if (cfg.debug_odo_disable_bgz_jacobian &&
        model.H.cols() > StateIdx::kBg + 2) {
      model.H(0, StateIdx::kBg + 2) = 0.0;
    }
    StateMask odo_update_mask;
    const StateMask *odo_update_mask_ptr = nullptr;
    if (cfg.debug_odo_disable_bgz_state_update) {
      odo_update_mask.fill(true);
      odo_update_mask[StateIdx::kBg + 2] = false;
      odo_update_mask_ptr = &odo_update_mask;
    }
    if (CorrectConstraintWithRobustness(
            engine, "ODO", t_meas, imu_curr, cfg, model.y, model.H, model.R,
            true, true, true, cfg.odo_nis_gate_prob,
            cfg.bgz_gate_apply_to_odo, odo_update_mask_ptr, diag, stats)) {
      last_odo_update_t = t_meas;
    }
    ++odo_idx;
  }
  return last_odo_speed;
}

void fusion_runtime::RunUwbUpdate(EskfEngine &engine, const Dataset &dataset,
                                  int &uwb_idx, double t_curr,
                                  const FusionOptions &options,
                                  bool schedule_active, double split_t,
                                  const vector<int> &head_idx,
                                  const vector<int> &tail_idx,
                                  DiagnosticsEngine &diag) {
  while (uwb_idx < dataset.uwb.rows()) {
    const double t_uwb = dataset.uwb(uwb_idx, 0);
    if (t_uwb > t_curr + options.gating.time_tolerance) {
      break;
    }

    const int n_meas_full = dataset.uwb.cols() - 1;
    const vector<int> *active = nullptr;
    if (schedule_active) {
      active = (t_uwb <= split_t) ? &head_idx : &tail_idx;
    }

    VectorXd z;
    MatrixXd anchors;
    if (active != nullptr) {
      const int k = static_cast<int>(active->size());
      z.resize(k);
      anchors.resize(k, 3);
      for (int j = 0; j < k; ++j) {
        const int idx = (*active)[j];
        z(j) = dataset.uwb(uwb_idx, 1 + idx);
        anchors.row(j) = dataset.anchors.positions.row(idx);
      }
    } else {
      z = dataset.uwb.block(uwb_idx, 1, 1, n_meas_full).transpose();
      anchors = dataset.anchors.positions;
    }

    if (z.size() == 0 || anchors.rows() == 0) {
      ++uwb_idx;
      continue;
    }

    const auto model = MeasModels::ComputeUwbModel(engine.state(), z, anchors,
                                                   options.noise.sigma_uwb);
    const double max_r = model.y.cwiseAbs().maxCoeff();
    if (max_r > options.gating.uwb_residual_max) {
      const int a0 = (active != nullptr) ? (*active)[0] : 0;
      const Vector3d a0_pos = anchors.row(0).transpose();
      cout << "[Warn] Large UWB residual at t=" << fixed << t_uwb
           << " | max_r=" << max_r << "\n"
           << "       State p: " << engine.state().p.transpose() << "\n"
           << "       Anchor " << a0 + 1 << ": " << a0_pos.transpose() << "\n"
           << "       Meas z: " << z.transpose() << "\n"
           << "       Pred h: " << (engine.state().p - a0_pos).norm() << "\n"
           << " -> Skipped\n";
    } else {
      (void)diag.Correct(engine, "UWB", t_uwb, model.y, model.H, model.R);
    }
    ++uwb_idx;
  }
}

bool fusion_runtime::RunGnssUpdate(EskfEngine &engine, const Dataset &dataset,
                                   int &gnss_idx, double t_curr,
                                   const FusionOptions &options,
                                   DiagnosticsEngine &diag,
                                   const ImuData *imu_curr,
                                   const InEkfManager *inekf,
                                   double gnss_split_t,
                                   FusionDebugCapture *debug_capture,
                                   FusionPerfStats *perf_stats) {
  if (dataset.gnss.timestamps.size() == 0) {
    return false;
  }
  constexpr double kSigmaVelMin = 1e-4;
  bool any_gnss_updated = false;
  const int gnss_idx_begin = gnss_idx;
  const SteadyClock::time_point call_start =
      (perf_stats != nullptr && perf_stats->enabled)
          ? SteadyClock::now()
          : SteadyClock::time_point{};

  while (gnss_idx < static_cast<int>(dataset.gnss.timestamps.size())) {
    const double t_gnss = dataset.gnss.timestamps(gnss_idx);
    if (t_gnss > t_curr + options.gating.time_tolerance) {
      break;
    }
    if (perf_stats != nullptr && perf_stats->enabled) {
      cerr << "[Perf][GNSS] begin idx=" << gnss_idx
           << " t_gnss=" << fixed << setprecision(6) << t_gnss
           << " t_curr=" << t_curr << endl;
    }

    const double dt_align = t_curr - t_gnss;
    Vector3d gnss_pos = Vector3d::Zero();
    Vector3d gnss_std = Vector3d::Zero();
    if (!ComputeAlignedGnssPositionMeasurement(dataset, options.noise, gnss_idx,
                                               t_curr, gnss_pos, gnss_std)) {
      if (perf_stats != nullptr && perf_stats->enabled) {
        cerr << "[Perf][GNSS] skip_invalid_measurement idx=" << gnss_idx
             << " t_gnss=" << t_gnss << endl;
      }
      ++gnss_idx;
      continue;
    }
    const bool has_gnss_vel_data = HasGnssVelocityData(dataset);

    const SteadyClock::time_point model_start =
        (perf_stats != nullptr && perf_stats->enabled)
            ? SteadyClock::now()
            : SteadyClock::time_point{};
    auto model = MeasModels::ComputeGnssPositionModel(
        engine.state(), gnss_pos, gnss_std, inekf);
    const double model_s =
        (perf_stats != nullptr && perf_stats->enabled)
            ? DurationSeconds(SteadyClock::now() - model_start)
            : 0.0;

    double omega_z_deg_s = 0.0;
    if (imu_curr != nullptr && imu_curr->dt > 1.0e-9) {
      omega_z_deg_s =
          (imu_curr->dtheta.z() / imu_curr->dt - engine.state().bg.z()) *
          (180.0 / 3.14159265358979323846);
    }
    double effective_pos_gain_scale = options.gnss_pos_position_gain_scale;
    double effective_lgy_from_y_gain_scale =
        options.gnss_pos_lgy_from_y_gain_scale;
    if (options.gnss_pos_turn_rate_threshold_deg_s > 0.0) {
      const bool strong_turn =
          std::abs(omega_z_deg_s) >=
          options.gnss_pos_turn_rate_threshold_deg_s;
      if (strong_turn && omega_z_deg_s > 0.0 &&
          options.gnss_pos_positive_turn_position_gain_scale >= 0.0) {
        effective_pos_gain_scale =
            options.gnss_pos_positive_turn_position_gain_scale;
      } else if (strong_turn && omega_z_deg_s < 0.0 &&
                 options.gnss_pos_negative_turn_position_gain_scale >= 0.0) {
        effective_pos_gain_scale =
            options.gnss_pos_negative_turn_position_gain_scale;
      }
      if (strong_turn && omega_z_deg_s > 0.0 &&
          options.gnss_pos_positive_turn_lgy_from_y_gain_scale >= 0.0) {
        effective_lgy_from_y_gain_scale =
            options.gnss_pos_positive_turn_lgy_from_y_gain_scale;
      } else if (strong_turn && omega_z_deg_s < 0.0 &&
                 options.gnss_pos_negative_turn_lgy_from_y_gain_scale >= 0.0) {
        effective_lgy_from_y_gain_scale =
            options.gnss_pos_negative_turn_lgy_from_y_gain_scale;
      }
    }

    StateGainScale gnss_pos_gain_scale;
    gnss_pos_gain_scale.fill(1.0);
    const StateGainScale *gnss_pos_gain_scale_ptr = nullptr;
    if (std::abs(effective_pos_gain_scale - 1.0) > 1.0e-12) {
      for (int axis = 0; axis < 3; ++axis) {
        gnss_pos_gain_scale[StateIdx::kPos + axis] =
            effective_pos_gain_scale;
      }
      gnss_pos_gain_scale_ptr = &gnss_pos_gain_scale;
    }

    StateMeasurementGainScale gnss_pos_gain_element_scale;
    const StateMeasurementGainScale *gnss_pos_gain_element_scale_ptr = nullptr;
    if ((std::abs(options.gnss_pos_lgx_from_y_gain_scale - 1.0) > 1.0e-12 ||
         std::abs(effective_lgy_from_y_gain_scale - 1.0) > 1.0e-12) &&
        model.y.size() >= 2) {
      gnss_pos_gain_element_scale =
          StateMeasurementGainScale::Ones(kStateDim, model.y.size());
      gnss_pos_gain_element_scale(StateIdx::kGnssLever + 0, 1) =
          options.gnss_pos_lgx_from_y_gain_scale;
      gnss_pos_gain_element_scale(StateIdx::kGnssLever + 1, 1) =
          effective_lgy_from_y_gain_scale;
      gnss_pos_gain_element_scale_ptr = &gnss_pos_gain_element_scale;
    }

    const string effective_gnss_pos_update_mode =
        ComputeEffectiveGnssPosUpdateMode(options, t_gnss);
    {
      const double h_att_norm =
          model.H.block<3, 3>(0, StateIdx::kAtt).norm();
      cout << "[GNSS_POS] t=" << fixed << setprecision(3) << t_gnss
           << " mode="
           << ((inekf != nullptr && inekf->enabled) ? "InEKF" : "ESKF")
           << " | ||H_att||_F=" << setprecision(6) << h_att_norm
           << " | update_mode=" << options.gnss_pos_update_mode
           << " | effective_update_mode=" << effective_gnss_pos_update_mode
           << " | pos_gain_scale=" << options.gnss_pos_position_gain_scale
           << " | effective_pos_gain_scale=" << effective_pos_gain_scale
           << " | lgx_from_y_gain_scale="
           << options.gnss_pos_lgx_from_y_gain_scale
           << " | lgy_from_y_gain_scale="
           << options.gnss_pos_lgy_from_y_gain_scale
           << " | effective_lgy_from_y_gain_scale="
           << effective_lgy_from_y_gain_scale
           << " | omega_z_deg_s=" << omega_z_deg_s;
      cout << "\n";
    }

    bool gnss_pos_updated = false;
    const SteadyClock::time_point pos_update_start =
        (perf_stats != nullptr && perf_stats->enabled)
            ? SteadyClock::now()
            : SteadyClock::time_point{};
    if (effective_gnss_pos_update_mode == "stage_nonpos_then_pos") {
      const StateMask non_position_mask = BuildGnssPosNonPositionMask();
      const StateMask position_only_mask = BuildGnssPosPositionOnlyMask();
      const bool stage1_updated = diag.Correct(
          engine, "GNSS_POS_STAGE1_NONPOS", t_gnss, model.y, model.H, model.R,
          &non_position_mask, nullptr, gnss_pos_gain_element_scale_ptr);
      any_gnss_updated = any_gnss_updated || stage1_updated;
      gnss_pos_updated = gnss_pos_updated || stage1_updated;

      auto stage2_model = MeasModels::ComputeGnssPositionModel(
          engine.state(), gnss_pos, gnss_std, inekf);
      const bool stage2_updated = diag.Correct(
          engine, "GNSS_POS_STAGE2_POS", t_gnss, stage2_model.y,
          stage2_model.H, stage2_model.R, &position_only_mask,
          gnss_pos_gain_scale_ptr, nullptr);
      any_gnss_updated = any_gnss_updated || stage2_updated;
      gnss_pos_updated = gnss_pos_updated || stage2_updated;
    } else if (effective_gnss_pos_update_mode == "position_only") {
      const StateMask position_only_mask = BuildGnssPosPositionOnlyMask();
      gnss_pos_updated = diag.Correct(
          engine, "GNSS_POS_POS_ONLY", t_gnss, model.y, model.H, model.R,
          &position_only_mask, gnss_pos_gain_scale_ptr, nullptr);
      any_gnss_updated = any_gnss_updated || gnss_pos_updated;
    } else {
      gnss_pos_updated = diag.Correct(
          engine, "GNSS_POS", t_gnss, model.y, model.H, model.R, nullptr,
          gnss_pos_gain_scale_ptr, gnss_pos_gain_element_scale_ptr);
      any_gnss_updated = any_gnss_updated || gnss_pos_updated;
    }
    const double pos_update_s =
        (perf_stats != nullptr && perf_stats->enabled)
            ? DurationSeconds(SteadyClock::now() - pos_update_start)
            : 0.0;
    if (gnss_pos_updated) {
      MaybeCaptureGnssSplitDebug(debug_capture, engine, "GNSS_POS", t_gnss,
                                 gnss_split_t, options.gating.time_tolerance);
    }

    double vel_update_s = 0.0;
    if (has_gnss_vel_data && options.enable_gnss_velocity) {
      Vector3d gnss_vel = dataset.gnss.velocities.row(gnss_idx).transpose();
      Vector3d gnss_vel_std = dataset.gnss.vel_std.row(gnss_idx).transpose();
      const int next_gnss = gnss_idx + 1;
      if (dt_align > 1e-9 &&
          next_gnss < static_cast<int>(dataset.gnss.timestamps.size())) {
        const double t_next = dataset.gnss.timestamps(next_gnss);
        const double dt_gnss = t_next - t_gnss;
        if (dt_gnss > 1e-6 && dt_align < dt_gnss) {
          const double alpha = dt_align / dt_gnss;
          const Vector3d vel_next =
              dataset.gnss.velocities.row(next_gnss).transpose();
          gnss_vel += alpha * (vel_next - gnss_vel);
        }
      }
      constexpr double sigma_vel_fallback = 0.5;
      for (int k = 0; k < 3; ++k) {
        if (!std::isfinite(gnss_vel_std(k)) || gnss_vel_std(k) <= 0.0) {
          gnss_vel_std(k) = sigma_vel_fallback;
        }
        if (gnss_vel_std(k) < kSigmaVelMin) {
          gnss_vel_std(k) = kSigmaVelMin;
        }
      }

      Vector3d omega_ib_b_raw = Vector3d::Zero();
      if (imu_curr != nullptr && imu_curr->dt > 1.0e-9) {
        omega_ib_b_raw = imu_curr->dtheta / imu_curr->dt;
      }
      const auto vel_model = MeasModels::ComputeGnssVelocityModel(
          engine.state(), gnss_vel, omega_ib_b_raw, gnss_vel_std, inekf);
      const SteadyClock::time_point vel_update_start =
          (perf_stats != nullptr && perf_stats->enabled)
              ? SteadyClock::now()
              : SteadyClock::time_point{};
      const bool gnss_vel_updated =
          diag.Correct(engine, "GNSS_VEL", t_gnss, vel_model.y, vel_model.H,
                       vel_model.R, nullptr);
      any_gnss_updated = any_gnss_updated || gnss_vel_updated;
      if (perf_stats != nullptr && perf_stats->enabled) {
        vel_update_s = DurationSeconds(SteadyClock::now() - vel_update_start);
      }
    }

    if (perf_stats != nullptr && perf_stats->enabled) {
      cerr << "[Perf][GNSS] end idx=" << gnss_idx
           << " t_gnss=" << t_gnss
           << " dt_align=" << dt_align
           << " model_s=" << model_s
           << " pos_update_s=" << pos_update_s
           << " vel_update_s=" << vel_update_s
           << " updated=" << (gnss_pos_updated ? 1 : 0) << endl;
    }

    ++gnss_idx;
  }

  if (perf_stats != nullptr) {
    ++perf_stats->gnss_update_calls;
    if (gnss_idx > gnss_idx_begin) {
      perf_stats->gnss_samples +=
          static_cast<size_t>(gnss_idx - gnss_idx_begin);
    }
  }
  if (perf_stats != nullptr && perf_stats->enabled) {
    perf_stats->gnss_update_s +=
        DurationSeconds(SteadyClock::now() - call_start);
  }
  return any_gnss_updated;
}

bool ComputeAlignedGnssPositionMeasurement(const Dataset &dataset,
                                          const NoiseParams &noise,
                                          int gnss_idx, double t_curr,
                                          Vector3d &gnss_pos_ecef,
                                          Vector3d &gnss_std_out) {
  (void)t_curr;
  if (dataset.gnss.timestamps.size() == 0) {
    return false;
  }
  if (gnss_idx < 0 || gnss_idx >= dataset.gnss.timestamps.size()) {
    return false;
  }

  constexpr double kSigmaPosMin = 1e-4;
  const double sigma_pos_fallback =
      noise.sigma_gnss_pos > 0.0 ? noise.sigma_gnss_pos : 1.0;

  gnss_pos_ecef = dataset.gnss.positions.row(gnss_idx).transpose();
  gnss_std_out = dataset.gnss.std.row(gnss_idx).transpose();
  for (int k = 0; k < 3; ++k) {
    if (!std::isfinite(gnss_std_out(k)) || gnss_std_out(k) <= 0.0) {
      gnss_std_out(k) = sigma_pos_fallback;
    }
    if (gnss_std_out(k) < kSigmaPosMin) {
      gnss_std_out(k) = kSigmaPosMin;
    }
  }
  return true;
}

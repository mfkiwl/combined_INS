#include "fusion_runtime_internal.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>

#include "io/data_io.h"

using namespace Eigen;
using namespace fusion_runtime;
using namespace std;

FusionRuntimeOutput RunFusionRuntime(
    const FusionOptions &options, const Dataset &dataset, const State &x0,
    const Matrix<double, kStateDim, kStateDim> &P0,
    FusionDebugCapture *debug_capture) {
  FusionRuntimeOutput output;
  FusionResult &result = output.result;
  FusionPerfStats perf_stats;
  perf_stats.enabled = IsPerfDebugEnabledFromEnv();
  if (perf_stats.enabled) {
    perf_stats.progress_stride = std::max<size_t>(1, dataset.imu.size() / 100);
    perf_stats.wall_start = SteadyClock::now();
    cerr << "[Perf] enabled progress_stride=" << perf_stats.progress_stride
         << " imu_rows=" << dataset.imu.size()
         << " gnss_rows=" << dataset.gnss.timestamps.size() << endl;
  }
  if (dataset.imu.size() < 2) {
    cout << "error: IMU 数据不足，至少需要两帧增量\n";
    return output;
  }

  DiagnosticsEngine diag(
      options.constraints,
      options.constraints.enable_diagnostics ||
          options.constraints.enable_mechanism_log ||
          !options.first_update_debug_output_path.empty() ||
          !options.gnss_update_debug_output_path.empty() ||
          !options.predict_debug_output_path.empty());
  diag.Initialize(dataset, options);

  std::ofstream nhc_admission_log_file;
  if (options.constraints.enable_nhc_admission_log) {
    namespace fs = std::filesystem;
    const fs::path sol_path(options.output_path);
    const fs::path log_path =
        sol_path.parent_path() /
        (sol_path.stem().string() + "_nhc_admission.csv");
    nhc_admission_log_file = io::OpenOutputFile(log_path.string());
    if (nhc_admission_log_file.is_open()) {
      nhc_admission_log_file << fixed << setprecision(9);
      nhc_admission_log_file
          << "t,selected_source,selected_accept,"
          << "accept_v_b,accept_v_wheel_b,accept_v_v,"
          << "below_forward_v_b,below_forward_v_wheel_b,below_forward_v_v,"
          << "exceed_lat_vert_v_b,exceed_lat_vert_v_wheel_b,exceed_lat_vert_v_v,"
          << "forward_speed_threshold,max_abs_v_threshold,"
          << "v_b_x,v_b_y,v_b_z,"
          << "v_wheel_b_x,v_wheel_b_y,v_wheel_b_z,"
          << "v_v_x,v_v_y,v_v_z\n";
    }
  }

  const bool runtime_truth_anchor_has_target =
      options.init.runtime_truth_anchor_position ||
      options.init.runtime_truth_anchor_velocity ||
      options.init.runtime_truth_anchor_attitude;
  const bool runtime_truth_anchor_enabled =
      options.init.runtime_truth_anchor_pva && runtime_truth_anchor_has_target;
  if (runtime_truth_anchor_enabled && dataset.truth.timestamps.size() <= 0) {
    cout << "error: runtime_truth_anchor_pva=true 但真值数据为空\n";
    return output;
  }
  if (options.init.runtime_truth_anchor_pva &&
      !runtime_truth_anchor_has_target) {
    cout << "[Init] WARNING: runtime_truth_anchor_pva=true，但 position/velocity/attitude "
            "全部关闭，已自动禁用 runtime anchor\n";
  }
  int truth_anchor_cursor = 0;

  StateAblationConfig active_ablation = options.ablation;
  ConstraintConfig effective_constraints = options.constraints;
  string last_runtime_phase_label;
  const auto build_active_phase_label = [&](double t_now) {
    string label;
    for (const auto &phase : options.runtime_phases) {
      if (!phase.enabled ||
          !IsTimeInWindow(t_now, phase.start_time, phase.end_time,
                          options.gating.time_tolerance)) {
        continue;
      }
      if (!label.empty()) {
        label += ",";
      }
      label += phase.name;
    }
    return label.empty() ? string("none") : label;
  };
  const auto compute_effective_ablation = [&](double t_now) {
    StateAblationConfig effective = active_ablation;
    for (const auto &phase : options.runtime_phases) {
      if (!phase.enabled ||
          !IsTimeInWindow(t_now, phase.start_time, phase.end_time,
                          options.gating.time_tolerance)) {
        continue;
      }
      effective = MergeAblationConfig(effective, phase.ablation);
    }
    if (IsTimeInWindow(
            t_now,
            options.constraints.debug_gnss_lever_arm_disable_start_time,
            options.constraints.debug_gnss_lever_arm_disable_end_time,
            options.gating.time_tolerance)) {
      effective.disable_gnss_lever_arm = true;
    }
    if (std::isfinite(options.constraints.debug_mounting_yaw_enable_after_time) &&
        t_now + options.gating.time_tolerance <
            options.constraints.debug_mounting_yaw_enable_after_time) {
      effective.disable_mounting_yaw = true;
    }
    return effective;
  };
  const auto compute_effective_constraints = [&](double t_now) {
    ConstraintConfig effective = options.constraints;
    for (const auto &phase : options.runtime_phases) {
      if (!phase.enabled ||
          !IsTimeInWindow(t_now, phase.start_time, phase.end_time,
                          options.gating.time_tolerance)) {
        continue;
      }
      effective = ApplyRuntimeConstraintOverride(effective, phase.constraints);
    }
    return effective;
  };
  const auto build_effective_noise = [&](double t_now,
                                         const StateAblationConfig &ablation) {
    NoiseParams effective = options.noise;
    for (const auto &phase : options.runtime_phases) {
      if (!phase.enabled ||
          !IsTimeInWindow(t_now, phase.start_time, phase.end_time,
                          options.gating.time_tolerance)) {
        continue;
      }
      ApplyRuntimeNoiseOverride(effective, phase.noise);
    }
    ApplyAblationToNoise(effective, ablation);
    return effective;
  };
  const auto build_covariance_floor = [&](const ConstraintConfig &cfg) {
    CovarianceFloor floor;
    floor.enabled = cfg.enable_covariance_floor;
    floor.pos_var = cfg.p_floor_pos_var;
    floor.vel_var = cfg.p_floor_vel_var;
    floor.att_var = std::pow(cfg.p_floor_att_deg * kDegToRad, 2);
    floor.odo_scale_var = cfg.p_floor_odo_scale_var;
    floor.lever_var = cfg.p_floor_lever_arm_vec;
    floor.mounting_var = std::pow(cfg.p_floor_mounting_deg * kDegToRad, 2);
    floor.bg_var = cfg.p_floor_bg_var;
    return floor;
  };

  StateAblationConfig effective_ablation =
      compute_effective_ablation(dataset.imu.front().t);
  effective_constraints = compute_effective_constraints(dataset.imu.front().t);
  NoiseParams runtime_noise =
      build_effective_noise(dataset.imu.front().t, effective_ablation);
  EskfEngine engine(runtime_noise);
  engine.SetStateMask(BuildStateMask(effective_ablation));

  CorrectionGuard guard;
  guard.enabled = options.constraints.enforce_extrinsic_bounds;
  guard.odo_scale_min = options.constraints.odo_scale_min;
  guard.odo_scale_max = options.constraints.odo_scale_max;
  guard.max_mounting_roll =
      options.constraints.max_mounting_roll_deg * kDegToRad;
  guard.max_mounting_pitch =
      options.constraints.max_mounting_pitch_deg * kDegToRad;
  guard.max_mounting_yaw =
      options.constraints.max_mounting_yaw_deg * kDegToRad;
  guard.max_lever_arm_norm = options.constraints.max_lever_arm_norm;
  guard.max_odo_scale_step = options.constraints.max_odo_scale_step;
  guard.max_mounting_step =
      options.constraints.max_mounting_step_deg * kDegToRad;
  guard.max_lever_arm_step = options.constraints.max_lever_arm_step;
  engine.SetCorrectionGuard(guard);
  engine.SetCovarianceFloor(build_covariance_floor(effective_constraints));

  const Vector3d cfg_mounting_rpy =
      options.constraints.imu_mounting_angle * kDegToRad;
  constexpr double kMountInitEps = 1e-12;
  const bool init_pitch_nonzero =
      std::abs(options.init.mounting_pitch0) > kMountInitEps;
  const bool init_yaw_nonzero =
      std::abs(options.init.mounting_yaw0) > kMountInitEps;
  if (options.init.strict_extrinsic_conflict &&
      ((init_pitch_nonzero &&
        std::abs(cfg_mounting_rpy.y()) > kMountInitEps) ||
       (init_yaw_nonzero &&
        std::abs(cfg_mounting_rpy.z()) > kMountInitEps))) {
    cout << "error: init.mounting_* 与 constraints.imu_mounting_angle 冲突，"
         << "且 strict_extrinsic_conflict=true\n";
    return output;
  }

  Vector3d mounting_base_rpy = cfg_mounting_rpy;
  if (options.init.use_legacy_mounting_base_logic) {
    const bool use_cfg_pitch_base = !init_pitch_nonzero;
    const bool use_cfg_yaw_base = !init_yaw_nonzero;
    mounting_base_rpy = Vector3d(
        cfg_mounting_rpy.x(),
        use_cfg_pitch_base ? cfg_mounting_rpy.y() : 0.0,
        use_cfg_yaw_base ? cfg_mounting_rpy.z() : 0.0);
  }
  cout << "[Init] C_b_v base rpy (deg): roll="
       << mounting_base_rpy.x() * kRadToDeg
       << " pitch=" << mounting_base_rpy.y() * kRadToDeg
       << " yaw=" << mounting_base_rpy.z() * kRadToDeg
       << " mode="
       << (options.init.use_legacy_mounting_base_logic ? "legacy"
                                                       : "constraints_base")
       << "\n";

  InEkfManager inekf;
  inekf.Enable(options.inekf.enable);
  inekf.apply_covariance_floor_after_reset =
      options.inekf.apply_covariance_floor_after_reset;
  inekf.ri_gnss_pos_use_p_ned_local = options.inekf.ri_gnss_pos_use_p_ned_local;
  inekf.ri_vel_gyro_noise_mode = options.inekf.ri_vel_gyro_noise_mode;
  inekf.ri_inject_pos_inverse = options.inekf.ri_inject_pos_inverse;
  inekf.debug_force_process_model = options.inekf.debug_force_process_model;
  inekf.debug_force_vel_jacobian = options.inekf.debug_force_vel_jacobian;
  inekf.debug_disable_true_reset_gamma =
      options.inekf.debug_disable_true_reset_gamma;
  inekf.debug_enable_standard_reset_gamma =
      options.inekf.debug_enable_standard_reset_gamma;
  engine.SetInEkfManager(&inekf);

  const auto apply_runtime_truth_anchor =
      [&](double t, const char *stage, bool gnss_updated) -> bool {
    if (!runtime_truth_anchor_enabled) {
      return true;
    }
    if (options.init.runtime_truth_anchor_gnss_only && !gnss_updated) {
      return true;
    }
    if (!ApplyRuntimeTruthAnchor(engine, dataset.truth, options.init, t,
                                 truth_anchor_cursor)) {
      cout << "error: runtime truth anchor failed at stage=" << stage
           << " t=" << fixed << setprecision(6) << t << "\n";
      return false;
    }
    return true;
  };

  engine.Initialize(x0, P0);
  if (!apply_runtime_truth_anchor(dataset.imu.front().t, "initialize", false)) {
    return output;
  }
  cout << "[Init] InEKF: " << (inekf.enabled ? "ON" : "OFF") << "\n";
  cout << "[Init] Runtime truth PVA anchor: "
       << (runtime_truth_anchor_enabled ? "ON" : "OFF") << "\n";
  if (!options.runtime_phases.empty()) {
    cout << "[Init] Runtime phases: " << options.runtime_phases.size()
         << " active_at_start="
         << build_active_phase_label(dataset.imu.front().t) << "\n";
  }
  if (options.constraints.debug_odo_disable_bgz_jacobian ||
      options.constraints.debug_odo_disable_bgz_state_update ||
      options.constraints.debug_nhc_disable_bgz_state_update ||
      options.constraints.enable_nhc_admission_log ||
      options.constraints.nhc_admission_velocity_source != "v_b" ||
      options.constraints.enable_bgz_observability_gate ||
      options.constraints.enable_bgz_covariance_forgetting ||
      options.constraints.debug_run_odo_before_nhc ||
      options.constraints.disable_nhc_when_weak_excitation ||
      options.constraints.disable_odo_when_weak_excitation ||
      std::isfinite(options.constraints.debug_nhc_disable_start_time) ||
      std::isfinite(options.constraints.debug_nhc_disable_end_time) ||
      std::isfinite(options.constraints.debug_odo_disable_start_time) ||
      std::isfinite(options.constraints.debug_odo_disable_end_time) ||
      std::isfinite(
          options.constraints.debug_gnss_lever_arm_disable_start_time) ||
      std::isfinite(
          options.constraints.debug_gnss_lever_arm_disable_end_time) ||
      std::isfinite(options.constraints.debug_nhc_enable_after_time) ||
      std::isfinite(options.constraints.debug_mounting_yaw_enable_after_time) ||
      std::isfinite(
          options.constraints.debug_reset_bg_z_state_and_cov_after_time) ||
      std::isfinite(
          options.constraints.debug_seed_mount_yaw_bgz_cov_before_first_nhc) ||
      std::isfinite(options.constraints.debug_seed_bg_z_before_first_nhc) ||
      options.constraints.debug_seed_bg_z_att_cov_before_first_nhc.allFinite()) {
    cout << "[Init] Constraint debug toggles: odo_disable_bgz_jacobian="
         << (options.constraints.debug_odo_disable_bgz_jacobian ? "ON"
                                                                : "OFF")
         << " nhc_admission_source="
         << options.constraints.nhc_admission_velocity_source
         << " nhc_admission_log="
         << (options.constraints.enable_nhc_admission_log ? "ON" : "OFF")
         << " odo_disable_bgz_state_update="
         << (options.constraints.debug_odo_disable_bgz_state_update ? "ON"
                                                                    : "OFF")
         << " nhc_disable_bgz_state_update="
         << (options.constraints.debug_nhc_disable_bgz_state_update ? "ON"
                                                                    : "OFF")
         << " bgz_observability_gate="
         << (options.constraints.enable_bgz_observability_gate ? "ON"
                                                               : "OFF")
         << " bgz_gate_apply_to_odo="
         << (options.constraints.bgz_gate_apply_to_odo ? "ON" : "OFF")
         << " bgz_gate_apply_to_nhc="
         << (options.constraints.bgz_gate_apply_to_nhc ? "ON" : "OFF")
         << " bgz_gate_forward_speed_min="
         << options.constraints.bgz_gate_forward_speed_min
         << " bgz_gate_yaw_rate_min_deg_s="
         << options.constraints.bgz_gate_yaw_rate_min_deg_s
         << " bgz_gate_lateral_acc_min="
         << options.constraints.bgz_gate_lateral_acc_min
         << " bgz_gate_min_scale=" << options.constraints.bgz_gate_min_scale
         << " bgz_covariance_forgetting="
         << (options.constraints.enable_bgz_covariance_forgetting ? "ON"
                                                                  : "OFF")
         << " bgz_cov_forgetting_tau_s="
         << options.constraints.bgz_cov_forgetting_tau_s
         << " disable_nhc_when_weak_excitation="
         << (options.constraints.disable_nhc_when_weak_excitation ? "ON"
                                                                  : "OFF")
         << " disable_odo_when_weak_excitation="
         << (options.constraints.disable_odo_when_weak_excitation ? "ON"
                                                                  : "OFF")
         << " odo_before_nhc="
         << (options.constraints.debug_run_odo_before_nhc ? "ON" : "OFF")
         << " nhc_disable_window=["
         << options.constraints.debug_nhc_disable_start_time << ", "
         << options.constraints.debug_nhc_disable_end_time << "]"
         << " odo_disable_window=["
         << options.constraints.debug_odo_disable_start_time << ", "
         << options.constraints.debug_odo_disable_end_time << "]"
         << " gnss_lever_disable_window=["
         << options.constraints.debug_gnss_lever_arm_disable_start_time << ", "
         << options.constraints.debug_gnss_lever_arm_disable_end_time << "]"
         << " nhc_enable_after_time="
         << options.constraints.debug_nhc_enable_after_time
         << " mounting_yaw_enable_after_time="
         << options.constraints.debug_mounting_yaw_enable_after_time
         << " reset_bg_z_state_and_cov_after_time="
         << options.constraints.debug_reset_bg_z_state_and_cov_after_time
         << " reset_bg_z_value=" << options.constraints.debug_reset_bg_z_value
         << " seed_mount_yaw_bgz_cov_before_first_nhc="
         << options.constraints.debug_seed_mount_yaw_bgz_cov_before_first_nhc
         << " seed_bg_z_before_first_nhc="
         << options.constraints.debug_seed_bg_z_before_first_nhc
         << " seed_bg_z_att_cov_before_first_nhc="
         << options.constraints.debug_seed_bg_z_att_cov_before_first_nhc
                .transpose()
         << "\n";
  }
  if (runtime_truth_anchor_enabled) {
    cout << "[Init] Runtime truth anchor components: pos="
         << (options.init.runtime_truth_anchor_position ? "ON" : "OFF")
         << " vel="
         << (options.init.runtime_truth_anchor_velocity ? "ON" : "OFF")
         << " att="
         << (options.init.runtime_truth_anchor_attitude ? "ON" : "OFF")
         << " trigger="
         << (options.init.runtime_truth_anchor_gnss_only ? "gnss_only"
                                                         : "all_stages")
         << "\n";
  }
  if (inekf.enabled) {
    cout << "[Init] RI Jacobian sign consistency check skipped "
         << "(new InEKF H is not expected to be opposite-sign to ESKF)\n";
    cout << "[Init] InEKF mode: unified InEKF\n";
    cout << "[Init] InEKF toggles: gnss_pos_p_term="
         << (inekf.ri_gnss_pos_use_p_ned_local ? "ON" : "OFF")
         << " g_vel_gyro_mode=" << inekf.ri_vel_gyro_noise_mode
         << " inject_pos_inverse="
         << (inekf.ri_inject_pos_inverse ? "ON" : "OFF")
         << " debug_force_process=" << inekf.debug_force_process_model
         << " debug_force_vel_jac=" << inekf.debug_force_vel_jacobian
         << " debug_disable_reset_gamma="
         << (inekf.debug_disable_true_reset_gamma ? "ON" : "OFF")
         << " debug_enable_std_reset_gamma="
         << (inekf.debug_enable_standard_reset_gamma ? "ON" : "OFF")
         << " reset_floor="
         << (inekf.apply_covariance_floor_after_reset ? "ON" : "OFF") << "\n";
  }
  cout << "[Init] Ablation: "
       << "gnss_lever="
       << (effective_ablation.disable_gnss_lever_arm ? "OFF" : "ON")
       << " gnss_lever_z="
       << ((effective_ablation.disable_gnss_lever_arm ||
            effective_ablation.disable_gnss_lever_z)
               ? "OFF"
               : "ON")
       << " odo_lever="
       << (effective_ablation.disable_odo_lever_arm ? "OFF" : "ON")
       << " odo_scale="
       << (effective_ablation.disable_odo_scale ? "OFF" : "ON")
       << " accel_bias="
       << (effective_ablation.disable_accel_bias ? "OFF" : "ON")
       << " gyro_bias="
       << (effective_ablation.disable_gyro_bias ? "OFF" : "ON")
       << " gyro_scale="
       << (effective_ablation.disable_gyro_scale ? "OFF" : "ON")
       << " accel_scale="
       << (effective_ablation.disable_accel_scale ? "OFF" : "ON")
       << " mounting="
       << (effective_ablation.disable_mounting ? "OFF" : "ON")
       << " mounting_roll="
       << ((effective_ablation.disable_mounting ||
            effective_ablation.disable_mounting_roll)
               ? "OFF"
               : "ON")
       << " mounting_pitch="
       << ((effective_ablation.disable_mounting ||
            effective_ablation.disable_mounting_pitch)
               ? "OFF"
               : "ON")
       << " mounting_yaw="
       << ((effective_ablation.disable_mounting ||
            effective_ablation.disable_mounting_yaw)
               ? "OFF"
               : "ON")
       << "\n";

  bool uwb_schedule_active = false;
  double uwb_schedule_split_t = 0.0;
  vector<int> uwb_head_indices;
  vector<int> uwb_tail_indices;
  if (options.uwb_anchor_schedule.enabled) {
    if (dataset.uwb.rows() == 0 || dataset.uwb.cols() <= 1 ||
        dataset.anchors.positions.rows() == 0) {
      cout << "[Warn] UWB anchor schedule enabled but no UWB data/anchors\n";
    } else {
      const int n_a = dataset.anchors.positions.rows();
      vector<int> hu;
      vector<int> tu;
      bool ok = BuildUniqueAnchorIndices(
          options.uwb_anchor_schedule.head_anchors, n_a,
          "uwb_anchor_schedule.head_anchors", hu);
      ok = ok && BuildUniqueAnchorIndices(
                     options.uwb_anchor_schedule.tail_anchors, n_a,
                     "uwb_anchor_schedule.tail_anchors", tu);
      double t0_uwb = 0.0;
      double t1_uwb = 0.0;
      if (ok && ComputeUwbTimeRange(dataset.uwb, t0_uwb, t1_uwb) &&
          t1_uwb > t0_uwb) {
        uwb_schedule_split_t =
            t0_uwb +
            options.uwb_anchor_schedule.head_ratio * (t1_uwb - t0_uwb);
        uwb_head_indices = hu;
        uwb_tail_indices = tu;
        uwb_schedule_active = true;
      } else if (ok) {
        cout << "[Warn] UWB anchor schedule skipped: invalid UWB time range\n";
      }
    }
  }

  bool gnss_schedule_active = false;
  double gnss_schedule_split_t = std::numeric_limits<double>::infinity();
  bool gnss_schedule_use_windows = false;
  double gnss_schedule_last_window_end =
      -std::numeric_limits<double>::infinity();
  if (options.gnss_schedule.enabled) {
    if (dataset.gnss.timestamps.size() <= 0) {
      cout << "[Warn] GNSS schedule enabled but GNSS data is empty\n";
    } else if (dataset.imu.empty() ||
               dataset.imu.back().t <= dataset.imu.front().t) {
      cout << "[Warn] GNSS schedule skipped: invalid IMU time range\n";
    } else if (!options.gnss_schedule.enabled_windows.empty()) {
      gnss_schedule_use_windows = true;
      gnss_schedule_active = true;
      gnss_schedule_last_window_end =
          options.gnss_schedule.enabled_windows.back().second;
      cout << "[GNSS] schedule ON: windows="
           << options.gnss_schedule.enabled_windows.size()
           << " last_window_end=" << gnss_schedule_last_window_end << "\n";
    } else {
      const double t0_nav = dataset.imu.front().t;
      const double t1_nav = dataset.imu.back().t;
      gnss_schedule_split_t =
          t0_nav + options.gnss_schedule.head_ratio * (t1_nav - t0_nav);
      gnss_schedule_active = true;
      cout << "[GNSS] schedule ON: head_ratio="
           << options.gnss_schedule.head_ratio
           << " split_t=" << gnss_schedule_split_t << "\n";
    }
  }
  bool post_gnss_ablation_applied = false;

  double static_duration = 0.0;
  int uwb_idx = 0;
  int odo_idx = 0;
  int gnss_idx = 0;
  double last_odo_speed = 0.0;
  double last_nhc_update_t = -1e18;
  double last_odo_update_t = -1e18;
  bool debug_seed_mount_yaw_bgz_cov_applied = false;
  bool debug_reset_bg_z_state_and_cov_applied = false;
  ConstraintUpdateStats nhc_stats;
  ConstraintUpdateStats odo_stats;
  const auto sync_runtime_controls = [&](double t_now) {
    effective_constraints = compute_effective_constraints(t_now);
    effective_ablation = compute_effective_ablation(t_now);
    const NoiseParams effective_noise =
        build_effective_noise(t_now, effective_ablation);
    engine.SetNoiseParams(effective_noise);
    engine.SetStateMask(BuildStateMask(effective_ablation));
    engine.SetCovarianceFloor(build_covariance_floor(effective_constraints));
    const string phase_label = build_active_phase_label(t_now);
    if (phase_label != last_runtime_phase_label) {
      cout << "[Runtime] phases t=" << fixed << setprecision(3) << t_now
           << " active=" << phase_label
           << " enable_odo="
           << (effective_constraints.enable_odo ? "ON" : "OFF")
           << " enable_nhc="
           << (effective_constraints.enable_nhc ? "ON" : "OFF")
           << " cov_floor="
           << (effective_constraints.enable_covariance_floor ? "ON" : "OFF")
           << " p_floor_odo_scale_var="
           << effective_constraints.p_floor_odo_scale_var
           << " p_floor_mounting_deg="
           << effective_constraints.p_floor_mounting_deg
           << " nis_gating="
           << (effective_constraints.enable_nis_gating ? "ON" : "OFF")
           << " odo_gate_prob=" << effective_constraints.odo_nis_gate_prob
           << " nhc_gate_prob=" << effective_constraints.nhc_nis_gate_prob
           << "\n";
      last_runtime_phase_label = phase_label;
    }
  };

  engine.AddImu(dataset.imu[0]);
  for (size_t i = 1; i < dataset.imu.size(); ++i) {
    engine.AddImu(dataset.imu[i]);
    const double t = dataset.imu[i].t;
    const double dt = dataset.imu[i].dt;
    sync_runtime_controls(t);

    const InEkfManager *inekf_ptr = inekf.enabled ? &inekf : nullptr;
    const InEkfManager *inekf_ptr_mut = inekf.enabled ? &inekf : nullptr;

    bool gnss_enabled_now = true;
    bool gnss_schedule_finished = false;
    if (gnss_schedule_active) {
      if (gnss_schedule_use_windows) {
        gnss_enabled_now = IsTimeInAnyWindow(
            t, options.gnss_schedule.enabled_windows,
            options.gating.time_tolerance);
        gnss_schedule_finished =
            t > gnss_schedule_last_window_end + options.gating.time_tolerance;
      } else {
        gnss_enabled_now =
            t <= gnss_schedule_split_t + options.gating.time_tolerance;
        gnss_schedule_finished = !gnss_enabled_now;
      }
    }
    if (gnss_schedule_use_windows && !gnss_enabled_now &&
        !gnss_schedule_finished) {
      const int drop_begin = gnss_idx;
      const int dropped = AdvanceTimestampIndexToTime(
          dataset.gnss.timestamps, gnss_idx, t, options.gating.time_tolerance);
      if (dropped > 0) {
        const double first_dropped_t = dataset.gnss.timestamps(drop_begin);
        const double last_dropped_t = dataset.gnss.timestamps(gnss_idx - 1);
        cout << "[GNSS] schedule dropped " << dropped
             << " sample(s) in off-window at t=" << fixed
             << setprecision(3) << t
             << " | sample_range=[" << first_dropped_t << ", "
             << last_dropped_t << "]\n";
      }
    }

    bool gnss_updated = false;
    bool predict_ok = false;
    if (gnss_enabled_now) {
      predict_ok = PredictCurrentIntervalWithExactGnss(
          engine, dataset, gnss_idx, t, options, diag, &dataset.imu[i],
          inekf_ptr_mut, gnss_schedule_split_t, debug_capture, gnss_updated,
          &perf_stats);
    } else {
      const SteadyClock::time_point predict_start =
          perf_stats.enabled ? SteadyClock::now() : SteadyClock::time_point{};
      predict_ok = engine.Predict();
      ++perf_stats.direct_predict_count;
      if (perf_stats.enabled) {
        perf_stats.predict_s +=
            DurationSeconds(SteadyClock::now() - predict_start);
      }
      if (predict_ok) {
        diag.LogPredict("direct", engine);
      }
    }
    if (!predict_ok) {
      continue;
    }

    if (!apply_runtime_truth_anchor(t, "predict", false)) {
      break;
    }
    ApplyBgzCovarianceForgettingIfNeeded(engine, dataset.imu[i],
                                         effective_constraints);
    ApplyDebugResetBgzStateAndCov(engine, effective_constraints, t,
                                  options.gating.time_tolerance,
                                  debug_reset_bg_z_state_and_cov_applied);

    const bool nhc_time_gate_open =
        !std::isfinite(effective_constraints.debug_nhc_enable_after_time) ||
        t + options.gating.time_tolerance >=
            effective_constraints.debug_nhc_enable_after_time;

    const SteadyClock::time_point zupt_start =
        perf_stats.enabled ? SteadyClock::now() : SteadyClock::time_point{};
    const bool zupt_ready = RunZuptUpdate(engine, dataset.imu[i],
                                          effective_constraints,
                                          static_duration, diag, t);
    if (perf_stats.enabled) {
      perf_stats.zupt_s += DurationSeconds(SteadyClock::now() - zupt_start);
    }
    if (!apply_runtime_truth_anchor(t, "zupt", false)) {
      break;
    }

    const SteadyClock::time_point gravity_diag_start =
        perf_stats.enabled ? SteadyClock::now() : SteadyClock::time_point{};
    diag.CheckGravityAlignment(t, dt, dataset.imu[i], engine.state(),
                               dataset.truth);
    if (perf_stats.enabled) {
      perf_stats.gravity_diag_s +=
          DurationSeconds(SteadyClock::now() - gravity_diag_start);
    }

    const double nhc_min_interval =
        std::max(0.0, effective_constraints.nhc_min_update_interval);
    const double odo_min_interval =
        std::max(0.0, effective_constraints.odo_min_update_interval);
    if (!effective_constraints.enable_odo) {
      AdvanceMatrixTimeIndexToTime(dataset.odo, odo_idx, 0,
                                   effective_constraints.odo_time_offset, t,
                                   options.gating.time_tolerance);
    }
    if (effective_constraints.debug_run_odo_before_nhc) {
      last_odo_speed = RunOdoUpdate(engine, dataset, effective_constraints,
                                    options.gating, mounting_base_rpy, odo_idx,
                                    t, dataset.imu[i], diag, inekf_ptr,
                                    odo_stats,
                                    last_odo_update_t, odo_min_interval);
      if (!apply_runtime_truth_anchor(t, "odo", false)) {
        break;
      }

      ApplyDebugSeedBeforeFirstNhc(
          engine, effective_constraints, debug_seed_mount_yaw_bgz_cov_applied);
      if (nhc_time_gate_open) {
        RunNhcUpdate(engine, dataset.imu[i], effective_constraints,
                     mounting_base_rpy, zupt_ready, diag, t, inekf_ptr,
                     nhc_stats,
                     last_nhc_update_t, nhc_min_interval,
                     &nhc_admission_log_file);
      }
      if (!apply_runtime_truth_anchor(t, "nhc", false)) {
        break;
      }
    } else {
      ApplyDebugSeedBeforeFirstNhc(
          engine, effective_constraints, debug_seed_mount_yaw_bgz_cov_applied);
      if (nhc_time_gate_open) {
        RunNhcUpdate(engine, dataset.imu[i], effective_constraints,
                     mounting_base_rpy, zupt_ready, diag, t, inekf_ptr,
                     nhc_stats,
                     last_nhc_update_t, nhc_min_interval,
                     &nhc_admission_log_file);
      }
      if (!apply_runtime_truth_anchor(t, "nhc", false)) {
        break;
      }

      last_odo_speed = RunOdoUpdate(engine, dataset, effective_constraints,
                                    options.gating, mounting_base_rpy, odo_idx,
                                    t, dataset.imu[i], diag, inekf_ptr,
                                    odo_stats,
                                    last_odo_update_t, odo_min_interval);
      if (!apply_runtime_truth_anchor(t, "odo", false)) {
        break;
      }
    }

    const SteadyClock::time_point uwb_start =
        perf_stats.enabled ? SteadyClock::now() : SteadyClock::time_point{};
    RunUwbUpdate(engine, dataset, uwb_idx, t, options, uwb_schedule_active,
                 uwb_schedule_split_t, uwb_head_indices, uwb_tail_indices,
                 diag);
    if (perf_stats.enabled) {
      perf_stats.uwb_s += DurationSeconds(SteadyClock::now() - uwb_start);
    }
    if (!apply_runtime_truth_anchor(t, "uwb", false)) {
      break;
    }

    if (gnss_schedule_finished && !post_gnss_ablation_applied) {
      gnss_idx = static_cast<int>(dataset.gnss.timestamps.size());
      if (options.post_gnss_ablation.enabled) {
        active_ablation = MergeAblationConfig(active_ablation,
                                              options.post_gnss_ablation.ablation);
        sync_runtime_controls(t);
        cout << "[GNSS] post-gnss ablation applied at t=" << fixed
             << setprecision(3) << t
             << " | gnss_lever="
             << (effective_ablation.disable_gnss_lever_arm ? "OFF" : "ON")
             << " gnss_lever_z="
             << ((effective_ablation.disable_gnss_lever_arm ||
                  effective_ablation.disable_gnss_lever_z)
                     ? "OFF"
                     : "ON")
             << " odo_lever="
             << (effective_ablation.disable_odo_lever_arm ? "OFF" : "ON")
             << " odo_scale="
             << (effective_ablation.disable_odo_scale ? "OFF" : "ON")
             << " accel_bias="
             << (effective_ablation.disable_accel_bias ? "OFF" : "ON")
             << " gyro_bias="
             << (effective_ablation.disable_gyro_bias ? "OFF" : "ON")
             << " mounting="
             << (effective_ablation.disable_mounting ? "OFF" : "ON")
             << " mounting_roll="
             << ((effective_ablation.disable_mounting ||
                  effective_ablation.disable_mounting_roll)
                     ? "OFF"
                     : "ON")
             << " mounting_pitch="
             << ((effective_ablation.disable_mounting ||
                  effective_ablation.disable_mounting_pitch)
                     ? "OFF"
                     : "ON")
             << " mounting_yaw="
             << ((effective_ablation.disable_mounting ||
                  effective_ablation.disable_mounting_yaw)
                     ? "OFF"
                     : "ON")
             << "\n";
      } else {
        cout << "[GNSS] updates disabled after split at t=" << fixed
             << setprecision(3) << t << "\n";
      }
      post_gnss_ablation_applied = true;
    }
    if (!apply_runtime_truth_anchor(t, "gnss", gnss_updated)) {
      break;
    }

    const SteadyClock::time_point step_diag_start =
        perf_stats.enabled ? SteadyClock::now() : SteadyClock::time_point{};
    diag.OnStepComplete(t, dt, engine.state(), dataset.imu[i], zupt_ready,
                        dataset.truth);
    if (perf_stats.enabled) {
      perf_stats.step_diag_s +=
          DurationSeconds(SteadyClock::now() - step_diag_start);
    }

    RecordResult(result, engine.state(), t);

    const SteadyClock::time_point diag_write_start =
        perf_stats.enabled ? SteadyClock::now() : SteadyClock::time_point{};
    diag.WriteDiagLine(t, dt, engine, dataset.imu[i], last_odo_speed,
                       dataset.imu.front().t);
    ++perf_stats.imu_steps;
    if (perf_stats.enabled) {
      perf_stats.diag_write_s +=
          DurationSeconds(SteadyClock::now() - diag_write_start);
      if (((i + 1) % perf_stats.progress_stride) == 0) {
        const double wall_s =
            DurationSeconds(SteadyClock::now() - perf_stats.wall_start);
        cerr << "[Perf] step=" << (i + 1) << "/" << dataset.imu.size()
             << " t=" << fixed << setprecision(3) << t
             << " wall_s=" << wall_s
             << " gnss_idx=" << gnss_idx
             << " exact_calls=" << perf_stats.gnss_exact_calls
             << " gnss_update_calls=" << perf_stats.gnss_update_calls
             << " gnss_samples=" << perf_stats.gnss_samples
             << " predict_s=" << perf_stats.predict_s
             << " gnss_exact_s=" << perf_stats.gnss_exact_s
             << " gnss_update_s=" << perf_stats.gnss_update_s
             << " gravity_diag_s=" << perf_stats.gravity_diag_s
             << " step_diag_s=" << perf_stats.step_diag_s
             << " diag_write_s=" << perf_stats.diag_write_s
             << " split_predict_count=" << perf_stats.split_predict_count
             << " align_curr_predict_count="
             << perf_stats.align_curr_predict_count
             << " tail_predict_count=" << perf_stats.tail_predict_count
             << " direct_predict_count=" << perf_stats.direct_predict_count
             << endl;
      }
    }
  }

  if (options.constraints.enable_consistency_log) {
    PrintConstraintStats("NHC", nhc_stats);
    PrintConstraintStats("ODO", odo_stats);
  }
  if (debug_capture != nullptr && debug_capture->gnss_split_cov.valid) {
    const auto &cov = debug_capture->gnss_split_cov;
    cout << "[GNSS_SPLIT_COV] tag=" << cov.tag
         << " split_t=" << cov.split_t
         << " t_meas=" << cov.t_meas
         << " t_state=" << cov.t_state
         << " P_att_bgz=" << cov.P_att_bgz.transpose()
         << " corr_att_bgz=" << cov.corr_att_bgz.transpose() << "\n";
  }
  if (debug_capture != nullptr && debug_capture->reset_consistency.valid) {
    const auto &reset = debug_capture->reset_consistency;
    cout << "[RESET_SNAPSHOT] tag=" << reset.tag
         << " split_t=" << reset.split_t
         << " t_meas=" << reset.t_meas
         << " t_state=" << reset.t_state
         << " floor_after_reset="
         << (reset.covariance_floor_applied ? "ON" : "OFF") << "\n";
  }
  diag.Finalize(result.time_axis.empty() ? 0.0 : result.time_axis.back());
  if (perf_stats.enabled) {
    const double wall_s =
        DurationSeconds(SteadyClock::now() - perf_stats.wall_start);
    cerr << "[Perf] final"
         << " wall_s=" << wall_s
         << " steps=" << perf_stats.imu_steps
         << " gnss_idx=" << gnss_idx
         << " exact_calls=" << perf_stats.gnss_exact_calls
         << " gnss_update_calls=" << perf_stats.gnss_update_calls
         << " gnss_samples=" << perf_stats.gnss_samples
         << " predict_s=" << perf_stats.predict_s
         << " gnss_exact_s=" << perf_stats.gnss_exact_s
         << " gnss_update_s=" << perf_stats.gnss_update_s
         << " zupt_s=" << perf_stats.zupt_s
         << " gravity_diag_s=" << perf_stats.gravity_diag_s
         << " uwb_s=" << perf_stats.uwb_s
         << " step_diag_s=" << perf_stats.step_diag_s
         << " diag_write_s=" << perf_stats.diag_write_s
         << " split_predict_count=" << perf_stats.split_predict_count
         << " align_curr_predict_count="
         << perf_stats.align_curr_predict_count
         << " tail_predict_count=" << perf_stats.tail_predict_count
         << " direct_predict_count=" << perf_stats.direct_predict_count
         << endl;
  }
  output.stats = BuildRuntimeStatsSnapshot(perf_stats);
  return output;
}

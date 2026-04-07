#include "fusion_runtime_internal.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iostream>

using namespace fusion_runtime;

double fusion_runtime::DurationSeconds(const SteadyClock::duration &duration) {
  return std::chrono::duration_cast<std::chrono::duration<double>>(duration)
      .count();
}

bool fusion_runtime::IsPerfDebugEnabledFromEnv() {
  const char *env = std::getenv("UWB_PERF_DEBUG");
  if (env == nullptr) {
    return false;
  }
  std::string value(env);
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) {
                   return static_cast<char>(std::tolower(c));
                 });
  return !(value.empty() || value == "0" || value == "false" ||
           value == "off" || value == "no");
}

void fusion_runtime::PrintConstraintStats(const std::string &tag,
                                          const ConstraintUpdateStats &stats) {
  if (stats.seen <= 0) {
    return;
  }
  const double accept_ratio =
      static_cast<double>(stats.accepted) / static_cast<double>(stats.seen);
  const double nis_mean =
      (stats.accepted > 0)
          ? stats.nis_sum / static_cast<double>(stats.accepted)
          : 0.0;
  const double robust_mean =
      (stats.accepted > 0)
          ? stats.robust_weight_sum / static_cast<double>(stats.accepted)
          : 0.0;
  const double noise_scale_mean =
      (stats.accepted > 0)
          ? stats.noise_scale_sum / static_cast<double>(stats.accepted)
          : 0.0;
  std::cout << "[Consistency] " << tag
            << " seen=" << stats.seen
            << " accepted=" << stats.accepted
            << " accept_ratio=" << accept_ratio
            << " reject_nis=" << stats.rejected_nis
            << " reject_numeric=" << stats.rejected_numeric
            << " nis_mean=" << nis_mean
            << " nis_max=" << stats.nis_max
            << " robust_w_mean=" << robust_mean
            << " noise_scale_mean=" << noise_scale_mean << "\n";
}

void fusion_runtime::RecordResult(FusionResult &result, const State &state,
                                  double t) {
  result.fused_positions.push_back(state.p);
  result.fused_velocities.push_back(state.v);
  result.fused_quaternions.push_back(state.q);
  result.mounting_roll.push_back(state.mounting_roll);
  result.mounting_pitch.push_back(state.mounting_pitch);
  result.mounting_yaw.push_back(state.mounting_yaw);
  result.odo_scale.push_back(state.odo_scale);
  result.sg.push_back(state.sg);
  result.sa.push_back(state.sa);
  result.ba.push_back(state.ba);
  result.bg.push_back(state.bg);
  result.lever_arm.push_back(state.lever_arm);
  result.gnss_lever_arm.push_back(state.gnss_lever_arm);
  result.time_axis.push_back(t);
}

FusionRuntimeStats fusion_runtime::BuildRuntimeStatsSnapshot(
    const FusionPerfStats &perf_stats) {
  FusionRuntimeStats stats;
  stats.imu_steps = perf_stats.imu_steps;
  stats.gnss_exact_calls = perf_stats.gnss_exact_calls;
  stats.gnss_update_calls = perf_stats.gnss_update_calls;
  stats.gnss_samples = perf_stats.gnss_samples;
  stats.gnss_align_prev_count = perf_stats.gnss_align_prev_count;
  stats.gnss_split_predict_count = perf_stats.split_predict_count;
  stats.gnss_align_curr_predict_count = perf_stats.align_curr_predict_count;
  stats.gnss_tail_predict_count = perf_stats.tail_predict_count;
  stats.direct_predict_count = perf_stats.direct_predict_count;
  stats.predict_s = perf_stats.predict_s;
  stats.gnss_exact_s = perf_stats.gnss_exact_s;
  stats.gnss_update_s = perf_stats.gnss_update_s;
  stats.zupt_s = perf_stats.zupt_s;
  stats.gravity_diag_s = perf_stats.gravity_diag_s;
  stats.uwb_s = perf_stats.uwb_s;
  stats.step_diag_s = perf_stats.step_diag_s;
  stats.diag_write_s = perf_stats.diag_write_s;
  return stats;
}

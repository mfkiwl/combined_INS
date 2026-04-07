#pragma once

#include <chrono>
#include <fstream>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "app/diagnostics.h"
#include "app/fusion.h"
#include "core/eskf.h"

namespace fusion_runtime {

constexpr double kDegToRad = 3.14159265358979323846 / 180.0;
constexpr double kRadToDeg = 180.0 / 3.14159265358979323846;
constexpr double kTruthAnchorPosVar = 1.0e-12;
constexpr double kTruthAnchorVelVar = 1.0e-12;
constexpr double kTruthAnchorAttVar = 1.0e-14;
using SteadyClock = std::chrono::steady_clock;

struct FusionPerfStats {
  bool enabled = false;
  size_t progress_stride = 2000;
  size_t imu_steps = 0;
  size_t gnss_exact_calls = 0;
  size_t gnss_update_calls = 0;
  size_t gnss_samples = 0;
  size_t gnss_align_prev_count = 0;
  size_t split_predict_count = 0;
  size_t align_curr_predict_count = 0;
  size_t tail_predict_count = 0;
  size_t direct_predict_count = 0;
  double predict_s = 0.0;
  double gnss_exact_s = 0.0;
  double gnss_update_s = 0.0;
  double zupt_s = 0.0;
  double gravity_diag_s = 0.0;
  double uwb_s = 0.0;
  double step_diag_s = 0.0;
  double diag_write_s = 0.0;
  SteadyClock::time_point wall_start = SteadyClock::time_point{};
};

struct ConstraintUpdateStats {
  int seen = 0;
  int accepted = 0;
  int rejected_nis = 0;
  int rejected_numeric = 0;
  double nis_sum = 0.0;
  double nis_max = 0.0;
  double robust_weight_sum = 0.0;
  double noise_scale_sum = 0.0;
};

double DurationSeconds(const SteadyClock::duration &duration);
bool IsPerfDebugEnabledFromEnv();

bool ApplyRuntimeTruthAnchor(EskfEngine &engine, const TruthData &truth,
                             const InitConfig &init, double t, int &cursor);
bool ApplyDebugSeedBeforeFirstNhc(EskfEngine &engine,
                                  const ConstraintConfig &cfg,
                                  bool &already_applied);
bool ApplyDebugResetBgzStateAndCov(EskfEngine &engine,
                                   const ConstraintConfig &cfg, double t,
                                   double time_tolerance,
                                   bool &already_applied);
bool ApplyBgzCovarianceForgettingIfNeeded(EskfEngine &engine,
                                          const ImuData &imu,
                                          const ConstraintConfig &cfg);
bool IsWeakExcitation(const State &state, const ImuData &imu,
                      const ConstraintConfig &cfg);
bool MeetsWeakExcitationThresholds(const State &state, const ImuData &imu,
                                   const ConstraintConfig &cfg);
bool IsTimeInWindow(double t, double start_time, double end_time,
                    double tol = 1.0e-6);
struct BgzObservabilityGateInfo {
  double forward_speed_abs = 0.0;
  double yaw_rate_abs = 0.0;
  double lateral_acc_abs = 0.0;
  double gate_scale = 1.0;
};
BgzObservabilityGateInfo ComputeBgzObservabilityGateInfo(
    const State &state, const ImuData &imu, const ConstraintConfig &cfg);
StateMask BuildStateMask(const StateAblationConfig &cfg);
StateMask BuildGnssPosNonPositionMask();
StateMask BuildGnssPosPositionOnlyMask();
StateAblationConfig MergeAblationConfig(const StateAblationConfig &base,
                                        const StateAblationConfig &extra);
void ApplyAblationToNoise(NoiseParams &noise, const StateAblationConfig &cfg);
void ApplyRuntimeNoiseOverride(NoiseParams &noise,
                               const RuntimeNoiseOverride &override_cfg);
ConstraintConfig ApplyRuntimeConstraintOverride(
    const ConstraintConfig &base,
    const RuntimeConstraintOverride &override_cfg);
std::string ComputeEffectiveGnssPosUpdateMode(const FusionOptions &options,
                                              double t_now);
bool IsTimeInAnyWindow(double t,
                       const std::vector<std::pair<double, double>> &windows,
                       double tol);

int AdvanceTimestampIndexToTime(const Eigen::VectorXd &timestamps, int &idx,
                                double t_curr, double tol);
int AdvanceMatrixTimeIndexToTime(const Eigen::MatrixXd &data, int &idx,
                                 int time_col, double time_offset,
                                 double t_curr, double tol);
bool BuildUniqueAnchorIndices(const std::vector<int> &indices, int n_anchors,
                              const std::string &label,
                              std::vector<int> &out);
bool ComputeUwbTimeRange(const Eigen::MatrixXd &uwb, double &t0, double &t1);
bool SplitImuMeasurementAtTimestamp(const ImuData &imu_prev,
                                    const ImuData &imu_curr, double timestamp,
                                    ImuData &mid_imu, ImuData &tail_imu);

bool RunZuptUpdate(EskfEngine &engine, const ImuData &imu,
                   const ConstraintConfig &cfg, double &static_duration,
                   DiagnosticsEngine &diag, double t);
void RunNhcUpdate(EskfEngine &engine, const ImuData &imu,
                  const ConstraintConfig &cfg,
                  const Eigen::Vector3d &mounting_base_rpy, bool zupt_ready,
                  DiagnosticsEngine &diag, double t, const InEkfManager *inekf,
                  ConstraintUpdateStats &stats, double &last_nhc_update_t,
                  double nhc_min_interval,
                  std::ofstream *nhc_admission_log_file);
double RunOdoUpdate(EskfEngine &engine, const Dataset &dataset,
                    const ConstraintConfig &cfg, const GatingConfig &gating,
                    const Eigen::Vector3d &mounting_base_rpy, int &odo_idx,
                    double t_curr, const ImuData &imu_curr,
                    DiagnosticsEngine &diag, const InEkfManager *inekf,
                    ConstraintUpdateStats &stats, double &last_odo_update_t,
                    double odo_min_interval);
void RunUwbUpdate(EskfEngine &engine, const Dataset &dataset, int &uwb_idx,
                  double t_curr, const FusionOptions &options,
                  bool schedule_active, double split_t,
                  const std::vector<int> &head_idx,
                  const std::vector<int> &tail_idx, DiagnosticsEngine &diag);
bool RunGnssUpdate(EskfEngine &engine, const Dataset &dataset, int &gnss_idx,
                   double t_curr, const FusionOptions &options,
                   DiagnosticsEngine &diag, const ImuData *imu_curr,
                   const InEkfManager *inekf, double gnss_split_t,
                   FusionDebugCapture *debug_capture,
                   FusionPerfStats *perf_stats);

bool PredictCurrentIntervalWithExactGnss(EskfEngine &engine,
                                         const Dataset &dataset, int &gnss_idx,
                                         double t_curr,
                                         const FusionOptions &options,
                                         DiagnosticsEngine &diag,
                                         const ImuData *imu_curr,
                                         const InEkfManager *inekf,
                                         double gnss_split_t,
                                         FusionDebugCapture *debug_capture,
                                         bool &gnss_updated_out,
                                         FusionPerfStats *perf_stats);

void PrintConstraintStats(const std::string &tag,
                          const ConstraintUpdateStats &stats);
void RecordResult(FusionResult &result, const State &state, double t);
FusionRuntimeStats BuildRuntimeStatsSnapshot(const FusionPerfStats &perf_stats);

}  // namespace fusion_runtime

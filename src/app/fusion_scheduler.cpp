#include "fusion_runtime_internal.h"

#include <algorithm>
#include <iomanip>
#include <iostream>

using namespace Eigen;
using namespace fusion_runtime;
using namespace std;

bool fusion_runtime::IsTimeInAnyWindow(
    double t, const vector<pair<double, double>> &windows, double tol) {
  for (const auto &window : windows) {
    if (t >= window.first - tol && t <= window.second + tol) {
      return true;
    }
  }
  return false;
}

int fusion_runtime::AdvanceTimestampIndexToTime(const VectorXd &timestamps,
                                                int &idx, double t_curr,
                                                double tol) {
  const int n = static_cast<int>(timestamps.size());
  const int start_idx = std::clamp(idx, 0, n);
  idx = start_idx;
  while (idx < n && timestamps(idx) <= t_curr + tol) {
    ++idx;
  }
  return idx - start_idx;
}

int fusion_runtime::AdvanceMatrixTimeIndexToTime(const MatrixXd &data, int &idx,
                                                 int time_col,
                                                 double time_offset,
                                                 double t_curr, double tol) {
  const int n = data.rows();
  const int start_idx = std::clamp(idx, 0, n);
  idx = start_idx;
  while (idx < n && data(idx, time_col) + time_offset <= t_curr + tol) {
    ++idx;
  }
  return idx - start_idx;
}

bool fusion_runtime::BuildUniqueAnchorIndices(const vector<int> &indices,
                                              int n_anchors,
                                              const string &label,
                                              vector<int> &out) {
  out.clear();
  for (int idx : indices) {
    if (idx < 0 || idx >= n_anchors) {
      cout << "error: " << label << " 索引超出基站数量范围\n";
      return false;
    }
    bool exists = false;
    for (int value : out) {
      if (value == idx) {
        exists = true;
        break;
      }
    }
    if (!exists) {
      out.push_back(idx);
    }
  }
  return true;
}

bool fusion_runtime::ComputeUwbTimeRange(const MatrixXd &uwb, double &t0,
                                         double &t1) {
  if (uwb.rows() == 0) {
    return false;
  }
  t0 = t1 = uwb(0, 0);
  for (int i = 1; i < uwb.rows(); ++i) {
    const double t = uwb(i, 0);
    if (t < t0) {
      t0 = t;
    }
    if (t > t1) {
      t1 = t;
    }
  }
  return true;
}

bool fusion_runtime::SplitImuMeasurementAtTimestamp(const ImuData &imu_prev,
                                                    const ImuData &imu_curr,
                                                    double timestamp,
                                                    ImuData &mid_imu,
                                                    ImuData &tail_imu) {
  const double interval = imu_curr.t - imu_prev.t;
  if (!(interval > 1.0e-9) || timestamp <= imu_prev.t ||
      timestamp >= imu_curr.t) {
    return false;
  }

  const double lambda = (timestamp - imu_prev.t) / interval;
  if (!(lambda > 0.0 && lambda < 1.0)) {
    return false;
  }

  mid_imu = imu_curr;
  mid_imu.t = timestamp;
  mid_imu.dtheta = imu_curr.dtheta * lambda;
  mid_imu.dvel = imu_curr.dvel * lambda;
  mid_imu.dt = timestamp - imu_prev.t;

  tail_imu = imu_curr;
  tail_imu.dtheta = imu_curr.dtheta - mid_imu.dtheta;
  tail_imu.dvel = imu_curr.dvel - mid_imu.dvel;
  tail_imu.dt = imu_curr.dt - mid_imu.dt;
  return tail_imu.dt > 1.0e-9;
}

bool fusion_runtime::PredictCurrentIntervalWithExactGnss(
    EskfEngine &engine, const Dataset &dataset, int &gnss_idx, double t_curr,
    const FusionOptions &options, DiagnosticsEngine &diag,
    const ImuData *imu_curr, const InEkfManager *inekf, double gnss_split_t,
    FusionDebugCapture *debug_capture, bool &gnss_updated_out,
    FusionPerfStats *perf_stats) {
  gnss_updated_out = false;
  if (dataset.gnss.timestamps.size() == 0) {
    return engine.Predict();
  }

  const SteadyClock::time_point call_start =
      (perf_stats != nullptr && perf_stats->enabled)
          ? SteadyClock::now()
          : SteadyClock::time_point{};
  if (perf_stats != nullptr) {
    ++perf_stats->gnss_exact_calls;
  }

  const double tol = options.gating.time_tolerance;
  ImuData seg_prev = engine.prev_imu();
  ImuData seg_curr = engine.curr_imu();
  bool propagated_to_curr = false;
  size_t dummy_predict_counter = 0;
  size_t &split_predict_counter =
      (perf_stats != nullptr) ? perf_stats->split_predict_count
                              : dummy_predict_counter;
  size_t &align_curr_predict_counter =
      (perf_stats != nullptr) ? perf_stats->align_curr_predict_count
                              : dummy_predict_counter;
  size_t &tail_predict_counter =
      (perf_stats != nullptr) ? perf_stats->tail_predict_count
                              : dummy_predict_counter;
  const auto timed_predict =
      [&](const ImuData &imu_prev_local, const ImuData &imu_curr_local,
          size_t &counter, const char *debug_tag) -> bool {
    const SteadyClock::time_point predict_start =
        (perf_stats != nullptr && perf_stats->enabled)
            ? SteadyClock::now()
            : SteadyClock::time_point{};
    const bool ok = engine.PredictWithImuPair(imu_prev_local, imu_curr_local);
    if (perf_stats != nullptr) {
      ++counter;
    }
    if (perf_stats != nullptr && perf_stats->enabled) {
      perf_stats->predict_s +=
          DurationSeconds(SteadyClock::now() - predict_start);
    }
    if (ok) {
      diag.LogPredict(debug_tag, engine);
    }
    return ok;
  };

  while (gnss_idx < static_cast<int>(dataset.gnss.timestamps.size())) {
    const double t_gnss = dataset.gnss.timestamps(gnss_idx);
    if (t_gnss > t_curr + tol) {
      break;
    }
    if (t_gnss < seg_prev.t - tol) {
      if (perf_stats != nullptr && perf_stats->enabled) {
        cerr << "[Perf][GNSS_EXACT] drop_stale idx=" << gnss_idx
             << " t_gnss=" << fixed << setprecision(6) << t_gnss
             << " seg_prev_t=" << seg_prev.t
             << " t_curr=" << t_curr << endl;
      }
      ++gnss_idx;
      continue;
    }
    if (IsTimestampAlignedToReference(t_gnss, seg_prev.t, tol)) {
      if (perf_stats != nullptr) {
        ++perf_stats->gnss_align_prev_count;
      }
      if (perf_stats != nullptr && perf_stats->enabled) {
        cerr << "[Perf][GNSS_EXACT] align_prev idx=" << gnss_idx
             << " t_gnss=" << t_gnss
             << " seg_prev_t=" << seg_prev.t << endl;
      }
      gnss_updated_out =
          RunGnssUpdate(engine, dataset, gnss_idx, seg_prev.t, options, diag,
                        &seg_prev, inekf, gnss_split_t, debug_capture,
                        perf_stats) ||
          gnss_updated_out;
      continue;
    }
    if (IsTimestampAlignedToReference(t_gnss, seg_curr.t, tol)) {
      if (perf_stats != nullptr && perf_stats->enabled) {
        cerr << "[Perf][GNSS_EXACT] align_curr idx=" << gnss_idx
             << " t_gnss=" << t_gnss
             << " seg_curr_t=" << seg_curr.t << endl;
      }
      if (!propagated_to_curr) {
        if (!timed_predict(seg_prev, seg_curr, align_curr_predict_counter,
                           "align_curr")) {
          return false;
        }
        propagated_to_curr = true;
      }
      gnss_updated_out =
          RunGnssUpdate(engine, dataset, gnss_idx, seg_curr.t, options, diag,
                        imu_curr, inekf, gnss_split_t, debug_capture,
                        perf_stats) ||
          gnss_updated_out;
      continue;
    }

    ImuData mid_imu;
    ImuData tail_imu;
    if (!SplitImuMeasurementAtTimestamp(seg_prev, seg_curr, t_gnss, mid_imu,
                                        tail_imu)) {
      if (perf_stats != nullptr && perf_stats->enabled) {
        cerr << "[Perf][GNSS_EXACT] split_failed idx=" << gnss_idx
             << " t_gnss=" << t_gnss
             << " seg_prev_t=" << seg_prev.t
             << " seg_curr_t=" << seg_curr.t << endl;
      }
      break;
    }
    if (perf_stats != nullptr && perf_stats->enabled) {
      cerr << "[Perf][GNSS_EXACT] split idx=" << gnss_idx
           << " t_gnss=" << t_gnss
           << " seg_prev_t=" << seg_prev.t
           << " seg_curr_t=" << seg_curr.t
           << " mid_dt=" << mid_imu.dt
           << " tail_dt=" << tail_imu.dt << endl;
    }
    if (!timed_predict(seg_prev, mid_imu, split_predict_counter, "split")) {
      return false;
    }
    gnss_updated_out =
        RunGnssUpdate(engine, dataset, gnss_idx, mid_imu.t, options, diag,
                      &mid_imu, inekf, gnss_split_t, debug_capture, perf_stats) ||
        gnss_updated_out;
    seg_prev = mid_imu;
    seg_curr = tail_imu;
  }

  if (!propagated_to_curr) {
    const bool ok = timed_predict(seg_prev, seg_curr, tail_predict_counter,
                                  "tail");
    if (perf_stats != nullptr && perf_stats->enabled) {
      perf_stats->gnss_exact_s +=
          DurationSeconds(SteadyClock::now() - call_start);
    }
    return ok;
  }
  if (perf_stats != nullptr && perf_stats->enabled) {
    perf_stats->gnss_exact_s +=
        DurationSeconds(SteadyClock::now() - call_start);
  }
  return true;
}

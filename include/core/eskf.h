// ESKF core compatibility header: nominal/process wrappers plus IMU-cache adapter.
#pragma once

#include <Eigen/Dense>

#include "navigation/filter_engine.h"
#include "navigation/process_model.h"
#include "utils/math_utils.h"

using namespace std;
using namespace Eigen;

class InsMech {
 public:
  static PropagationResult Propagate(const State &state, const ImuData &imu_prev,
                                     const ImuData &imu_curr);

  static void BuildProcessModel(const Matrix3d &C_bn,
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
                                const InEkfManager *inekf = nullptr);
};

class EskfEngine {
 public:
  explicit EskfEngine(const NoiseParams &noise);

  void Initialize(const State &state,
                  const Matrix<double, kStateDim, kStateDim> &P0);
  void AddImu(const ImuData &imu);
  bool Predict();
  bool PredictWithImuPair(const ImuData &imu_prev, const ImuData &imu_curr);

  bool Correct(const VectorXd &y, const MatrixXd &H,
               const MatrixXd &R, VectorXd *dx_out = nullptr,
               const StateMask *update_mask = nullptr,
               const StateGainScale *gain_scale = nullptr,
               const StateMeasurementGainScale *gain_element_scale = nullptr);

  const State &state() const { return filter_.state(); }
  const Matrix<double, kStateDim, kStateDim> &cov() const { return filter_.cov(); }
  const InEkfCorrectionSnapshot &last_inekf_correction() const {
    return filter_.last_inekf_correction();
  }
  const CorrectionDebugSnapshot &last_correction_debug() const {
    return filter_.last_correction_debug();
  }
  const PredictDebugSnapshot &last_predict_debug() const {
    return filter_.last_predict_debug();
  }
  double timestamp() const { return curr_imu_.t; }
  const ImuData &prev_imu() const { return prev_imu_; }
  const ImuData &curr_imu() const { return curr_imu_; }
  void SetInEkfManager(InEkfManager *inekf);
  void SetStateMask(const StateMask &mask) { filter_.SetStateMask(mask); }
  void SetCorrectionGuard(const CorrectionGuard &guard) {
    filter_.SetCorrectionGuard(guard);
  }
  void SetNoiseParams(const NoiseParams &noise) { filter_.SetNoiseParams(noise); }
  void SetCovarianceFloor(const CovarianceFloor &floor) {
    filter_.SetCovarianceFloor(floor);
  }
  void OverrideStateAndCov(const State &state,
                           const Matrix<double, kStateDim, kStateDim> &P) {
    filter_.OverrideStateAndCov(state, P);
  }

 private:
  enum class ImuCacheState { Empty, HasCurr, Ready };

 NavigationFilterEngine filter_;
  ImuData prev_imu_{};
  ImuData curr_imu_{};
  ImuCacheState imu_state_{ImuCacheState::Empty};
  InEkfManager *inekf_ = nullptr;
};

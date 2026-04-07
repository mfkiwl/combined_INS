// Navigation-layer Kalman execution core shared by wrappers and direct tests.
#pragma once

#include <Eigen/Dense>

#include "navigation/filter_contracts.h"
#include "navigation/measurement_model.h"
#include "navigation/process_model.h"

using namespace Eigen;

struct CorrectionGuard {
  bool enabled = false;
  double odo_scale_min = 0.5;
  double odo_scale_max = 1.5;
  double max_mounting_roll = 45.0 * EIGEN_PI / 180.0;
  double max_mounting_pitch = 30.0 * EIGEN_PI / 180.0;
  double max_mounting_yaw = 45.0 * EIGEN_PI / 180.0;
  double max_lever_arm_norm = 5.0;
  double max_odo_scale_step = 0.02;
  double max_mounting_step = 0.5 * EIGEN_PI / 180.0;
  double max_lever_arm_step = 0.05;
};

struct CovarianceFloor {
  bool enabled = false;
  double pos_var = 0.01;
  double vel_var = 0.001;
  double att_var =
      (0.01 * EIGEN_PI / 180.0) * (0.01 * EIGEN_PI / 180.0);
  double odo_scale_var = 0.0;
  Vector3d lever_var = Vector3d::Zero();
  double mounting_var =
      (0.1 * EIGEN_PI / 180.0) * (0.1 * EIGEN_PI / 180.0);
  double bg_var = 1.0e-8;
};

struct InEkfCorrectionSnapshot {
  bool valid = false;
  double t_state = 0.0;
  Matrix<double, kStateDim, kStateDim> P_tilde =
      Matrix<double, kStateDim, kStateDim>::Zero();
  Matrix<double, kStateDim, kStateDim> P_after_reset =
      Matrix<double, kStateDim, kStateDim>::Zero();
  Matrix<double, kStateDim, kStateDim> P_after_all =
      Matrix<double, kStateDim, kStateDim>::Zero();
  Matrix<double, kStateDim, 1> dx =
      Matrix<double, kStateDim, 1>::Zero();
  bool covariance_floor_applied = false;
};

struct CorrectionDebugSnapshot {
  bool valid = false;
  bool used_inekf = false;
  double t_state = 0.0;
  Matrix<double, kStateDim, kStateDim> P_prior =
      Matrix<double, kStateDim, kStateDim>::Zero();
  VectorXd y;
  MatrixXd H;
  MatrixXd R;
  MatrixXd S;
  MatrixXd K;
  Matrix<double, kStateDim, 1> dx =
      Matrix<double, kStateDim, 1>::Zero();
};

struct PredictDebugSnapshot {
  bool valid = false;
  double t_prev = 0.0;
  double t_curr = 0.0;
  double dt = 0.0;
  Matrix<double, kPredictDebugCommonDim, kPredictDebugCommonDim> P_before_common =
      Matrix<double, kPredictDebugCommonDim, kPredictDebugCommonDim>::Zero();
  Matrix<double, kPredictDebugCommonDim, kPredictDebugCommonDim> Phi_common =
      Matrix<double, kPredictDebugCommonDim, kPredictDebugCommonDim>::Zero();
  Matrix<double, kPredictDebugCommonDim, kPredictDebugCommonDim> Qd_common =
      Matrix<double, kPredictDebugCommonDim, kPredictDebugCommonDim>::Zero();
  Matrix<double, kPredictDebugCommonDim, kPredictDebugCommonDim> PhiP_common =
      Matrix<double, kPredictDebugCommonDim, kPredictDebugCommonDim>::Zero();
  Matrix<double, kPredictDebugCommonDim, kPredictDebugCommonDim> P_after_raw_common =
      Matrix<double, kPredictDebugCommonDim, kPredictDebugCommonDim>::Zero();
  Matrix<double, kPredictDebugCommonDim, kPredictDebugCommonDim> P_after_final_common =
      Matrix<double, kPredictDebugCommonDim, kPredictDebugCommonDim>::Zero();
  Vector3d omega_ie_b = Vector3d::Zero();
  Vector3d dtheta_prev_imu_corr = Vector3d::Zero();
  Vector3d dtheta_curr_imu_corr = Vector3d::Zero();
  Vector3d dtheta_prev_corr = Vector3d::Zero();
  Vector3d dtheta_curr_corr = Vector3d::Zero();
  Vector3d dvel_prev_corr = Vector3d::Zero();
  Vector3d dvel_curr_corr = Vector3d::Zero();
  Vector3d coning = Vector3d::Zero();
  Vector3d sculling = Vector3d::Zero();
  Vector3d dv_nav = Vector3d::Zero();
  Vector3d dv_nav_prev_att = Vector3d::Zero();
  Vector3d gravity_dt = Vector3d::Zero();
  Vector3d coriolis_dt = Vector3d::Zero();
};

class NavigationFilterEngine {
 public:
  explicit NavigationFilterEngine(
      const NoiseParams &noise,
      const FilterSemantics &semantics = BuildStandardEskfSemantics());

  void Initialize(const State &state,
                  const Matrix<double, kStateDim, kStateDim> &P0);
  void OverrideStateAndCov(const State &state,
                           const Matrix<double, kStateDim, kStateDim> &P);
  bool Predict(const ProcessLinearization &process,
               double t_prev,
               double t_curr);

  bool Correct(const MeasurementLinearization &measurement,
               VectorXd *dx_out = nullptr,
               const StateMask *update_mask = nullptr,
               const StateGainScale *gain_scale = nullptr,
               const StateMeasurementGainScale *gain_element_scale = nullptr);
  bool Correct(const VectorXd &y, const MatrixXd &H, const MatrixXd &R,
               VectorXd *dx_out = nullptr,
               const StateMask *update_mask = nullptr,
               const StateGainScale *gain_scale = nullptr,
               const StateMeasurementGainScale *gain_element_scale = nullptr);

  const State &state() const { return state_; }
  const Matrix<double, kStateDim, kStateDim> &cov() const { return P_; }
  const NoiseParams &noise() const { return noise_; }
  const FilterSemantics &base_semantics() const { return semantics_; }
  bool initialized() const { return initialized_; }
  double state_timestamp() const { return state_timestamp_; }
  const InEkfCorrectionSnapshot &last_inekf_correction() const {
    return last_inekf_correction_;
  }
  const CorrectionDebugSnapshot &last_correction_debug() const {
    return last_correction_debug_;
  }
  const PredictDebugSnapshot &last_predict_debug() const {
    return last_predict_debug_;
  }

  void SetStateTimestamp(double t) { state_timestamp_ = t; }
  void ClearPredictDebugSnapshot() { last_predict_debug_.valid = false; }
  void SetInEkfManager(InEkfManager *inekf);
  void SetStateMask(const StateMask &mask);
  void SetCorrectionGuard(const CorrectionGuard &guard) {
    correction_guard_ = guard;
  }
  void SetNoiseParams(const NoiseParams &noise) { noise_ = noise; }
  void SetCovarianceFloor(const CovarianceFloor &floor) {
    covariance_floor_ = floor;
  }

 private:
  FilterSemantics ResolveEffectiveSemantics() const;

  MatrixXd ComputeKalmanGain(const MatrixXd &H, const MatrixXd &R) const;
  void InjectErrorState(const VectorXd &dx,
                        const StateMask *update_mask = nullptr);
  void UpdateCovarianceJoseph(const MatrixXd &K, const MatrixXd &H,
                              const MatrixXd &R);
  void ApplyInEkfReset(const VectorXd &dx);
  void ApplyStandardEskfReset(const VectorXd &dx);
  bool IsStateEnabledByMasks(int idx, const StateMask *update_mask) const;
  void ApplyStateMaskToDx(VectorXd &dx, const StateMask *update_mask) const;
  void ApplyUpdateMaskToKalmanGain(MatrixXd &K,
                                   const StateMask *update_mask) const;
  void ApplyGainScaleToKalmanGain(MatrixXd &K,
                                  const StateGainScale *gain_scale) const;
  void ApplyElementGainScaleToKalmanGain(
      MatrixXd &K,
      const StateMeasurementGainScale *gain_element_scale) const;
  void ApplyStateMaskToCov();
  void ApplyCovarianceFloor();
  void ApplyMarkovNominalPropagation(State &state, double dt) const;
  void ApplyPredictionResult(
      const State &state,
      const Matrix<double, kStateDim, kStateDim> &P,
      const PredictDebugSnapshot *predict_debug = nullptr);
  Matrix<double, kStateDim, kStateDim> BuildInEkfResetGamma(
      const VectorXd &dx) const;
  Matrix<double, kStateDim, kStateDim> BuildStandardEskfResetGamma(
      const VectorXd &dx) const;

  NoiseParams noise_;
  FilterSemantics semantics_;
  State state_;
  Matrix<double, kStateDim, kStateDim> P_ =
      Matrix<double, kStateDim, kStateDim>::Zero();
  bool initialized_ = false;
  double state_timestamp_ = 0.0;
  InEkfManager *inekf_ = nullptr;
  StateMask state_mask_{};
  CorrectionGuard correction_guard_{};
  CovarianceFloor covariance_floor_{};
  InEkfCorrectionSnapshot last_inekf_correction_{};
  CorrectionDebugSnapshot last_correction_debug_{};
  PredictDebugSnapshot last_predict_debug_{};
};

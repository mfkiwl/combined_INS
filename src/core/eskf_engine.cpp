#include "core/eskf.h"

using namespace Eigen;

namespace {

}  // namespace

EskfEngine::EskfEngine(const NoiseParams &noise)
    : filter_(noise, BuildStandardEskfSemantics()) {}

void EskfEngine::Initialize(const State &state,
                            const Matrix<double, kStateDim, kStateDim> &P0) {
  filter_.Initialize(state, P0);
}

void EskfEngine::AddImu(const ImuData &imu) {
  switch (imu_state_) {
    case ImuCacheState::Empty:
      curr_imu_ = imu;
      imu_state_ = ImuCacheState::HasCurr;
      break;
    case ImuCacheState::HasCurr:
    case ImuCacheState::Ready:
      prev_imu_ = curr_imu_;
      curr_imu_ = imu;
      if (curr_imu_.dt <= 0.0) {
        curr_imu_.dt = curr_imu_.t - prev_imu_.t;
      }
      imu_state_ = ImuCacheState::Ready;
      break;
  }
}

bool EskfEngine::Predict() {
  if (!filter_.initialized() || imu_state_ != ImuCacheState::Ready) {
    return false;
  }
  return PredictWithImuPair(prev_imu_, curr_imu_);
}

bool EskfEngine::PredictWithImuPair(const ImuData &imu_prev,
                                    const ImuData &imu_curr) {
  if (!filter_.initialized()) {
    return false;
  }

  prev_imu_ = imu_prev;
  curr_imu_ = imu_curr;
  if (curr_imu_.dt <= 0.0) {
    curr_imu_.dt = curr_imu_.t - prev_imu_.t;
  }
  imu_state_ = ImuCacheState::Ready;
  if (curr_imu_.dt <= 1.0e-9) {
    return false;
  }

  ProcessModelInput input;
  input.nominal = filter_.state();
  input.imu_prev = prev_imu_;
  input.imu_curr = curr_imu_;
  input.noise = filter_.noise();
  input.semantics = (inekf_ != nullptr)
                        ? BuildProcessSemanticsFromInEkfConfig(*inekf_)
                        : filter_.base_semantics();
  const ProcessLinearization process = BuildProcessLinearization(input);
  return filter_.Predict(process, imu_prev.t, curr_imu_.t);
}

bool EskfEngine::Correct(const VectorXd &y, const MatrixXd &H,
                         const MatrixXd &R, VectorXd *dx_out,
                         const StateMask *update_mask,
                         const StateGainScale *gain_scale,
                         const StateMeasurementGainScale *gain_element_scale) {
  filter_.SetStateTimestamp(curr_imu_.t);
  return filter_.Correct(y, H, R, dx_out, update_mask, gain_scale,
                         gain_element_scale);
}

void EskfEngine::SetInEkfManager(InEkfManager *inekf) {
  inekf_ = inekf;
  filter_.SetInEkfManager(inekf);
}

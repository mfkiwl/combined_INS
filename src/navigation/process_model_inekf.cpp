#include "navigation/process_model.h"

#include "utils/math_utils.h"

using namespace Eigen;

namespace {

constexpr double kMinDt = 1e-9;

void ApplyFirstOrderMarkovDynamics(
    double corr_time,
    Matrix<double, kStateDim, kStateDim> &F) {
  if (corr_time <= 0.0) {
    return;
  }
  const double invT = 1.0 / corr_time;
  const Matrix3d neg_invT_I = -invT * Matrix3d::Identity();
  F.block<3, 3>(StateIdx::kBa, StateIdx::kBa) = neg_invT_I;
  F.block<3, 3>(StateIdx::kBg, StateIdx::kBg) = neg_invT_I;
  F.block<3, 3>(StateIdx::kSg, StateIdx::kSg) = neg_invT_I;
  F.block<3, 3>(StateIdx::kSa, StateIdx::kSa) = neg_invT_I;
}

}  // namespace

ProcessLinearization BuildInEkfProcessLinearization(
    const ProcessModelInput &input) {
  ProcessModelResolvedInput resolved = ResolveProcessModelInput(input);
  if (resolved.semantics.flavor == FilterFlavor::kStandardEskf) {
    resolved.semantics = BuildInEkfSemantics();
  }
  return BuildInEkfProcessLinearization(resolved);
}

ProcessLinearization BuildInEkfProcessLinearization(
    const ProcessModelResolvedInput &input) {
  ProcessLinearization out;
  out.propagation = input.propagation;
  if (input.dt <= kMinDt) {
    return out;
  }

  ProcessModelResolvedInput standard_input = input;
  standard_input.semantics = BuildStandardEskfSemantics();
  out = BuildStandardEskfProcessLinearization(standard_input);
  out.propagation = input.propagation;

  const auto [R_M, R_N] = process_model_detail::ComputeEarthRadii(input.lat);
  const Vector3d omega_ie_n =
      process_model_detail::ComputeOmegaIeNed(input.lat);
  const Vector3d omega_en_n = process_model_detail::ComputeOmegaEnNed(
      input.v_ned, input.lat, input.h, R_M, R_N);
  (void)omega_ie_n;
  (void)omega_en_n;

  Matrix<double, kStateDim, kStateDim> F = out.F;
  const Matrix3d neg_skew_omega = -Skew(input.omega_ib_b_corr);
  const Matrix3d sf_a_diag = input.sf_a.asDiagonal().toDenseMatrix();
  const Matrix3d sf_g_diag = input.sf_g.asDiagonal().toDenseMatrix();
  const Matrix3d diag_fb_unbiased =
      (input.f_b_unbiased.cwiseProduct(
           input.sf_a.cwiseProduct(input.sf_a)))
          .asDiagonal();
  const Matrix3d diag_wib_unbiased =
      (input.omega_ib_b_unbiased.cwiseProduct(
           input.sf_g.cwiseProduct(input.sf_g)))
          .asDiagonal();

  F.block<3, 3>(StateIdx::kPos, StateIdx::kPos) = neg_skew_omega;
  F.block<3, 3>(StateIdx::kPos, StateIdx::kVel) = Matrix3d::Identity();
  F.block<3, 3>(StateIdx::kPos, StateIdx::kAtt).setZero();

  F.block<3, 3>(StateIdx::kVel, StateIdx::kPos).setZero();
  F.block<3, 3>(StateIdx::kVel, StateIdx::kVel) = neg_skew_omega;
  F.block<3, 3>(StateIdx::kVel, StateIdx::kAtt) = -Skew(input.f_b_corr);
  F.block<3, 3>(StateIdx::kVel, StateIdx::kBa) = -sf_a_diag;
  F.block<3, 3>(StateIdx::kVel, StateIdx::kSa) = -diag_fb_unbiased;

  F.block<3, 3>(StateIdx::kAtt, StateIdx::kPos).setZero();
  F.block<3, 3>(StateIdx::kAtt, StateIdx::kVel).setZero();
  F.block<3, 3>(StateIdx::kAtt, StateIdx::kAtt) = neg_skew_omega;
  F.block<3, 3>(StateIdx::kAtt, StateIdx::kBg) = -sf_g_diag;
  F.block<3, 3>(StateIdx::kAtt, StateIdx::kSg) = -diag_wib_unbiased;

  ApplyFirstOrderMarkovDynamics(input.noise.markov_corr_time, F);
  out.F = F;
  out.Phi = process_model_detail::BuildDiscreteTransition(F, input.dt);
  const Matrix<double, kStateDim, kStateDim> Qc_cont =
      process_model_detail::BuildContinuousNoiseCovariance(input);
  out.Qd = process_model_detail::BuildDiscreteNoiseCovariance(
      out.Phi, Qc_cont, input.dt);
  return out;
}

#include "navigation/process_model.h"

#include <cmath>
#include <iostream>

#include "utils/math_utils.h"

using namespace std;
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

void CheckLargeProcessEntries(const Matrix<double, kStateDim, kStateDim> &F,
                              double lat,
                              double h,
                              const Vector3d &v_ned) {
  const double max_F = F.cwiseAbs().maxCoeff();
  if (max_F <= 1e6) {
    return;
  }
  std::cerr << "[BuildProcessModel] WARNING: max |F| = " << max_F
            << " at lat=" << lat << " h=" << h << "\n";
  int max_row = 0;
  int max_col = 0;
  F.cwiseAbs().maxCoeff(&max_row, &max_col);
  std::cerr << "[BuildProcessModel] Max F at (" << max_row << "," << max_col
            << "): " << F(max_row, max_col) << "\n";
  std::cerr << "[BuildProcessModel] tan(lat)=" << tan(lat)
            << " cos(lat)=" << cos(lat)
            << " v_ned=" << v_ned.transpose() << "\n";
}

}  // namespace

ProcessModelResolvedInput ResolveProcessModelInput(const ProcessModelInput &input) {
  ProcessModelResolvedInput resolved;
  resolved.semantics = input.semantics;
  resolved.noise = input.noise;
  resolved.dt = input.imu_curr.dt;
  resolved.propagation =
      BuildNominalPropagation(input.nominal, input.imu_prev, input.imu_curr);
  if (resolved.dt <= kMinDt) {
    return resolved;
  }

  const Llh llh = EcefToLlh(resolved.propagation.state.p);
  const Matrix3d R_ne = RotNedToEcef(llh);
  resolved.C_bn = R_ne.transpose() * resolved.propagation.Cbn;
  resolved.f_b_corr = resolved.propagation.f_b;
  resolved.v_ned = R_ne.transpose() * resolved.propagation.state.v;
  resolved.lat = llh.lat;
  resolved.h = llh.h;

  const Vector3d omega_ib_b_raw = input.imu_curr.dtheta / resolved.dt;
  resolved.omega_ib_b_unbiased = omega_ib_b_raw - input.nominal.bg;
  resolved.f_b_unbiased = input.imu_curr.dvel / resolved.dt - input.nominal.ba;
  resolved.sf_g = (Vector3d::Ones() + input.nominal.sg).cwiseInverse();
  resolved.sf_a = (Vector3d::Ones() + input.nominal.sa).cwiseInverse();
  resolved.omega_ib_b_corr =
      resolved.sf_g.cwiseProduct(resolved.omega_ib_b_unbiased);
  return resolved;
}

ProcessLinearization BuildProcessLinearization(const ProcessModelInput &input) {
  return BuildProcessLinearization(ResolveProcessModelInput(input));
}

ProcessLinearization BuildProcessLinearization(
    const ProcessModelResolvedInput &input) {
  if (input.semantics.flavor == FilterFlavor::kStandardEskf) {
    return BuildStandardEskfProcessLinearization(input);
  }
  return BuildInEkfProcessLinearization(input);
}

ProcessLinearization BuildStandardEskfProcessLinearization(
    const ProcessModelInput &input) {
  ProcessModelResolvedInput resolved = ResolveProcessModelInput(input);
  resolved.semantics = BuildStandardEskfSemantics();
  return BuildStandardEskfProcessLinearization(resolved);
}

ProcessLinearization BuildStandardEskfProcessLinearization(
    const ProcessModelResolvedInput &input) {
  ProcessLinearization out;
  out.propagation = input.propagation;
  if (input.dt <= kMinDt) {
    return out;
  }

  process_model_detail::ValidateProcessModelInput(input);

  const auto [R_M, R_N] = process_model_detail::ComputeEarthRadii(input.lat);
  const Vector3d omega_ie_n =
      process_model_detail::ComputeOmegaIeNed(input.lat);
  const Vector3d omega_en_n = process_model_detail::ComputeOmegaEnNed(
      input.v_ned, input.lat, input.h, R_M, R_N);
  const Vector3d omega_in_n = omega_ie_n + omega_en_n;
  const Vector3d a_m_ned = input.C_bn * input.f_b_corr;

  Matrix<double, kStateDim, kStateDim> F =
      Matrix<double, kStateDim, kStateDim>::Zero();
  F.block<3, 3>(StateIdx::kPos, StateIdx::kPos) = -Skew(omega_en_n);
  F.block<3, 3>(StateIdx::kPos, StateIdx::kVel) = Matrix3d::Identity();

  const double g = process_model_detail::ComputeLocalGravity(input.lat, input.h);
  Matrix3d F_vr = Matrix3d::Zero();
  F_vr(2, 2) = 2.0 * g / (sqrt(R_M * R_N) + input.h);
  Vector3d F_vr_col0;
  F_vr_col0 << 2.0 * (-omega_ie_n.z()) / (R_M + input.h), 0.0,
      2.0 * (-omega_ie_n.x()) / (R_M + input.h);
  F_vr.col(0) = F_vr_col0;
  F.block<3, 3>(StateIdx::kVel, StateIdx::kPos) = F_vr;
  F.block<3, 3>(StateIdx::kVel, StateIdx::kVel) =
      -Skew(2.0 * omega_ie_n + omega_en_n);
  F.block<3, 3>(StateIdx::kVel, StateIdx::kAtt) = Skew(a_m_ned);

  const Matrix3d sf_a_diag = input.sf_a.asDiagonal().toDenseMatrix();
  F.block<3, 3>(StateIdx::kVel, StateIdx::kBa) = -input.C_bn * sf_a_diag;
  const Matrix3d diag_fb_unbiased =
      (input.f_b_unbiased.cwiseProduct(
           input.sf_a.cwiseProduct(input.sf_a)))
          .asDiagonal();
  F.block<3, 3>(StateIdx::kVel, StateIdx::kSa) =
      -input.C_bn * diag_fb_unbiased;

  Matrix3d F_phir = Matrix3d::Zero();
  F_phir(0, 0) = -omega_ie_n.z() / (R_M + input.h);
  F_phir(2, 0) = -omega_ie_n.x() / (R_M + input.h);
  F_phir(0, 2) = input.v_ned.y() / ((R_N + input.h) * (R_N + input.h));
  F_phir(1, 2) = -input.v_ned.x() / ((R_M + input.h) * (R_M + input.h));
  F_phir(2, 0) += -input.v_ned.y() /
                   ((R_M + input.h) * (R_N + input.h) * cos(input.lat) *
                    cos(input.lat));
  F_phir(2, 2) = -input.v_ned.y() * tan(input.lat) /
                 ((R_N + input.h) * (R_N + input.h));
  F.block<3, 3>(StateIdx::kAtt, StateIdx::kPos) = F_phir;

  Matrix3d F_phiv = Matrix3d::Zero();
  F_phiv(0, 1) = 1.0 / (R_N + input.h);
  F_phiv(1, 0) = -1.0 / (R_M + input.h);
  F_phiv(2, 1) = -tan(input.lat) / (R_N + input.h);
  F.block<3, 3>(StateIdx::kAtt, StateIdx::kVel) = F_phiv;
  F.block<3, 3>(StateIdx::kAtt, StateIdx::kAtt) = -Skew(omega_in_n);

  const Matrix3d sf_g_diag = input.sf_g.asDiagonal().toDenseMatrix();
  F.block<3, 3>(StateIdx::kAtt, StateIdx::kBg) = input.C_bn * sf_g_diag;
  const Matrix3d diag_wib_unbiased =
      (input.omega_ib_b_unbiased.cwiseProduct(
           input.sf_g.cwiseProduct(input.sf_g)))
          .asDiagonal();
  F.block<3, 3>(StateIdx::kAtt, StateIdx::kSg) =
      input.C_bn * diag_wib_unbiased;

  ApplyFirstOrderMarkovDynamics(input.noise.markov_corr_time, F);
  CheckLargeProcessEntries(F, input.lat, input.h, input.v_ned);

  out.F = F;
  out.Phi = process_model_detail::BuildDiscreteTransition(F, input.dt);
  const Matrix<double, kStateDim, kStateDim> Qc_cont =
      process_model_detail::BuildContinuousNoiseCovariance(input);
  out.Qd = process_model_detail::BuildDiscreteNoiseCovariance(
      out.Phi, Qc_cont, input.dt);
  return out;
}

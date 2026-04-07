#include "navigation/process_model.h"

#include <cassert>
#include <cmath>
#include <iostream>

#include "utils/math_utils.h"

using namespace std;
using namespace Eigen;

namespace {

constexpr double kOmegaEarth = 7.292115e-5;
constexpr int kNoiseDim = 28;
constexpr double kLatGuard = 1.57079632679 + 0.1;

}  // namespace

namespace process_model_detail {

void ValidateProcessModelInput(const ProcessModelResolvedInput &input) {
  assert(std::isfinite(input.noise.sigma_sg));
  assert(std::isfinite(input.noise.sigma_sa));
  assert(std::isfinite(input.noise.sigma_odo_scale));
  assert(std::isfinite(input.noise.sigma_mounting));
  assert(std::isfinite(input.noise.sigma_lever_arm));
  assert(std::isfinite(input.noise.sigma_gnss_lever_arm));
  assert(input.noise.sigma_ba_vec.allFinite());
  assert(input.noise.sigma_bg_vec.allFinite());
  assert(input.noise.sigma_sg_vec.allFinite());
  assert(input.noise.sigma_sa_vec.allFinite());
  assert(input.noise.sigma_lever_arm_vec.allFinite());
  assert(input.noise.sigma_gnss_lever_arm_vec.allFinite());

  if (std::abs(input.lat) > kLatGuard) {
    std::cerr << "[BuildProcessModel] WARNING: lat out of range: "
              << input.lat << "\n";
  }
  if (std::abs(input.h) > 1e7) {
    std::cerr << "[BuildProcessModel] WARNING: h out of range: " << input.h
              << "\n";
  }
  if (input.v_ned.norm() > 1e5) {
    std::cerr << "[BuildProcessModel] WARNING: v_ned too large: "
              << input.v_ned.transpose() << "\n";
  }
}

std::pair<double, double> ComputeEarthRadii(double lat) {
  constexpr double kA = 6378137.0;
  constexpr double kE2 = 6.69437999014e-3;
  const double sin_lat = sin(lat);
  const double sin2_lat = sin_lat * sin_lat;
  const double R_M = kA * (1.0 - kE2) / pow(1.0 - kE2 * sin2_lat, 1.5);
  const double R_N = kA / sqrt(1.0 - kE2 * sin2_lat);
  return {R_M, R_N};
}

Vector3d ComputeOmegaIeNed(double lat) {
  const double cos_lat = cos(lat);
  const double sin_lat = sin(lat);
  return Vector3d(kOmegaEarth * cos_lat, 0.0, -kOmegaEarth * sin_lat);
}

Vector3d ComputeOmegaEnNed(const Vector3d &v_ned,
                          double lat,
                          double h,
                          double R_M,
                          double R_N) {
  const double v_N = v_ned.x();
  const double v_E = v_ned.y();

  const double omega_en_N = v_E / (R_N + h);
  const double omega_en_E = -v_N / (R_M + h);
  const double omega_en_D = -v_E * tan(lat) / (R_N + h);
  return Vector3d(omega_en_N, omega_en_E, omega_en_D);
}

double ComputeLocalGravity(double lat, double h) {
  const double sin_lat = sin(lat);
  const double sin2_lat = sin_lat * sin_lat;
  const double sin4_lat = sin2_lat * sin2_lat;
  constexpr double kGammaA = 9.7803267715;
  const double g0 =
      kGammaA * (1.0 + 0.0052790414 * sin2_lat + 0.0000232718 * sin4_lat);
  return g0 - (3.0877e-6 - 4.3e-9 * sin2_lat) * h + 0.72e-12 * h * h;
}

Vector3d ResolveVectorNoise(const Vector3d &vec, double scalar) {
  if ((vec.array() >= 0.0).all()) {
    return vec;
  }
  return Vector3d::Constant(scalar);
}

Matrix<double, kStateDim, kStateDim> BuildDiscreteTransition(
    const Matrix<double, kStateDim, kStateDim> &F,
    double dt) {
  Matrix<double, kStateDim, kStateDim> Phi =
      Matrix<double, kStateDim, kStateDim>::Identity();
  Phi += F * dt;
  return Phi;
}

Matrix<double, kStateDim, kStateDim> BuildContinuousNoiseCovariance(
    const ProcessModelResolvedInput &input) {
  Matrix<double, kStateDim, kNoiseDim> G =
      Matrix<double, kStateDim, kNoiseDim>::Zero();
  G.block<3, 3>(StateIdx::kVel, 0) = -input.C_bn;
  G.block<3, 3>(StateIdx::kAtt, 3) = -input.C_bn;
  if (input.semantics.flavor != FilterFlavor::kStandardEskf) {
    G.block<3, 3>(StateIdx::kVel, 0) = -Matrix3d::Identity();
    G.block<3, 3>(StateIdx::kAtt, 3) = -Matrix3d::Identity();
    G.block<3, 3>(StateIdx::kVel, 3).setZero();
  }

  G.block<3, 3>(StateIdx::kBa, 6) = Matrix3d::Identity();
  G.block<3, 3>(StateIdx::kBg, 9) = Matrix3d::Identity();
  G.block<3, 3>(StateIdx::kSg, 12) = Matrix3d::Identity();
  G.block<3, 3>(StateIdx::kSa, 15) = Matrix3d::Identity();
  G(StateIdx::kOdoScale, 18) = 1.0;
  G(StateIdx::kMountRoll, 19) = 1.0;
  G(StateIdx::kMountPitch, 20) = 1.0;
  G(StateIdx::kMountYaw, 21) = 1.0;
  G.block<3, 3>(StateIdx::kLever, 22) = Matrix3d::Identity();
  G.block<3, 3>(StateIdx::kGnssLever, 25) = Matrix3d::Identity();

  const double T = input.noise.markov_corr_time;
  const Vector3d ba_sigma =
      ResolveVectorNoise(input.noise.sigma_ba_vec, input.noise.sigma_ba);
  const Vector3d bg_sigma =
      ResolveVectorNoise(input.noise.sigma_bg_vec, input.noise.sigma_bg);
  const Vector3d sg_sigma =
      ResolveVectorNoise(input.noise.sigma_sg_vec, input.noise.sigma_sg);
  const Vector3d sa_sigma =
      ResolveVectorNoise(input.noise.sigma_sa_vec, input.noise.sigma_sa);
  const Vector3d lever_sigma = ResolveVectorNoise(
      input.noise.sigma_lever_arm_vec, input.noise.sigma_lever_arm);
  const Vector3d gnss_lever_sigma = ResolveVectorNoise(
      input.noise.sigma_gnss_lever_arm_vec, input.noise.sigma_gnss_lever_arm);

  Vector3d ba_w = ba_sigma;
  Vector3d bg_w = bg_sigma;
  Vector3d sg_w = sg_sigma;
  Vector3d sa_w = sa_sigma;
  if (T > 0.0) {
    const double scale = sqrt(2.0 / T);
    ba_w *= scale;
    bg_w *= scale;
    sg_w *= scale;
    sa_w *= scale;
  }

  const double sigma_mounting_roll =
      (input.noise.sigma_mounting_roll >= 0.0)
          ? input.noise.sigma_mounting_roll
          : input.noise.sigma_mounting;
  const double sigma_mounting_pitch =
      (input.noise.sigma_mounting_pitch >= 0.0)
          ? input.noise.sigma_mounting_pitch
          : input.noise.sigma_mounting;
  const double sigma_mounting_yaw =
      (input.noise.sigma_mounting_yaw >= 0.0)
          ? input.noise.sigma_mounting_yaw
          : input.noise.sigma_mounting;

  Matrix<double, kNoiseDim, kNoiseDim> Qc =
      Matrix<double, kNoiseDim, kNoiseDim>::Zero();
  const double sa2 = input.noise.sigma_acc * input.noise.sigma_acc;
  const double sg2 = input.noise.sigma_gyro * input.noise.sigma_gyro;
  Qc.diagonal()
      << sa2, sa2, sa2, sg2, sg2, sg2, ba_w.x() * ba_w.x(),
      ba_w.y() * ba_w.y(), ba_w.z() * ba_w.z(), bg_w.x() * bg_w.x(),
      bg_w.y() * bg_w.y(), bg_w.z() * bg_w.z(), sg_w.x() * sg_w.x(),
      sg_w.y() * sg_w.y(), sg_w.z() * sg_w.z(), sa_w.x() * sa_w.x(),
      sa_w.y() * sa_w.y(), sa_w.z() * sa_w.z(),
      input.noise.sigma_odo_scale * input.noise.sigma_odo_scale,
      sigma_mounting_roll * sigma_mounting_roll,
      sigma_mounting_pitch * sigma_mounting_pitch,
      sigma_mounting_yaw * sigma_mounting_yaw,
      lever_sigma.x() * lever_sigma.x(), lever_sigma.y() * lever_sigma.y(),
      lever_sigma.z() * lever_sigma.z(),
      gnss_lever_sigma.x() * gnss_lever_sigma.x(),
      gnss_lever_sigma.y() * gnss_lever_sigma.y(),
      gnss_lever_sigma.z() * gnss_lever_sigma.z();

  return G * Qc * G.transpose();
}

Matrix<double, kStateDim, kStateDim> BuildDiscreteNoiseCovariance(
    const Matrix<double, kStateDim, kStateDim> &Phi,
    const Matrix<double, kStateDim, kStateDim> &Qc_cont,
    double dt) {
  Matrix<double, kStateDim, kStateDim> Qd =
      0.5 * (Phi * Qc_cont * Phi.transpose() + Qc_cont) * dt;
  return 0.5 * (Qd + Qd.transpose());
}

}  // namespace process_model_detail

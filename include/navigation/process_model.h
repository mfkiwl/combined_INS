// Process-model builders shared by standard ESKF and InEKF flavors.
#pragma once

#include <utility>

#include <Eigen/Dense>

#include "navigation/filter_contracts.h"
#include "navigation/nominal_propagation.h"

using namespace Eigen;

struct ProcessModelInput {
  State nominal{};
  ImuData imu_prev{};
  ImuData imu_curr{};
  NoiseParams noise{};
  FilterSemantics semantics{};
};

struct ProcessModelResolvedInput {
  FilterSemantics semantics{};
  NoiseParams noise{};
  PropagationResult propagation{};
  Matrix3d C_bn = Matrix3d::Identity();
  Vector3d f_b_corr = Vector3d::Zero();
  Vector3d omega_ib_b_corr = Vector3d::Zero();
  Vector3d f_b_unbiased = Vector3d::Zero();
  Vector3d omega_ib_b_unbiased = Vector3d::Zero();
  Vector3d sf_a = Vector3d::Ones();
  Vector3d sf_g = Vector3d::Ones();
  Vector3d v_ned = Vector3d::Zero();
  double lat = 0.0;
  double h = 0.0;
  double dt = 0.0;
  int ri_vel_gyro_noise_mode = -1;
};

struct ProcessLinearization {
  PropagationResult propagation{};
  Matrix<double, kStateDim, kStateDim> Phi =
      Matrix<double, kStateDim, kStateDim>::Identity();
  Matrix<double, kStateDim, kStateDim> Qd =
      Matrix<double, kStateDim, kStateDim>::Zero();
  Matrix<double, kStateDim, kStateDim> F =
      Matrix<double, kStateDim, kStateDim>::Zero();
};

ProcessModelResolvedInput ResolveProcessModelInput(const ProcessModelInput &input);

ProcessLinearization BuildProcessLinearization(const ProcessModelInput &input);
ProcessLinearization BuildProcessLinearization(
    const ProcessModelResolvedInput &input);

ProcessLinearization BuildStandardEskfProcessLinearization(
    const ProcessModelInput &input);
ProcessLinearization BuildStandardEskfProcessLinearization(
    const ProcessModelResolvedInput &input);

ProcessLinearization BuildInEkfProcessLinearization(
    const ProcessModelInput &input);
ProcessLinearization BuildInEkfProcessLinearization(
    const ProcessModelResolvedInput &input);

namespace process_model_detail {

void ValidateProcessModelInput(const ProcessModelResolvedInput &input);
std::pair<double, double> ComputeEarthRadii(double lat);
Vector3d ComputeOmegaIeNed(double lat);
Vector3d ComputeOmegaEnNed(const Vector3d &v_ned,
                          double lat,
                          double h,
                          double R_M,
                          double R_N);
double ComputeLocalGravity(double lat, double h);
Vector3d ResolveVectorNoise(const Vector3d &vec, double scalar);

Matrix<double, kStateDim, kStateDim> BuildDiscreteTransition(
    const Matrix<double, kStateDim, kStateDim> &F,
    double dt);
Matrix<double, kStateDim, kStateDim> BuildContinuousNoiseCovariance(
    const ProcessModelResolvedInput &input);
Matrix<double, kStateDim, kStateDim> BuildDiscreteNoiseCovariance(
    const Matrix<double, kStateDim, kStateDim> &Phi,
    const Matrix<double, kStateDim, kStateDim> &Qc_cont,
    double dt);

}  // namespace process_model_detail

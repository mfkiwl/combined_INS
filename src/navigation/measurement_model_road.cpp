#include "navigation/measurement_model.h"

using namespace std;
using namespace Eigen;

MeasurementLinearization BuildNhcMeasurement(
    const NhcMeasurementInput &input) {
  using namespace measurement_model_detail;

  const bool runtime_inekf = UseInEkf(input.context);
  const VelJacobianMode vel_mode = ResolveVelJacobianMode(input.context);
  MeasurementLinearization model =
      MakeMeasurementLinearization(input.context, 2, "NHC", "VEHICLE");

  const MeasurementFrameContext frame =
      BuildMeasurementFrameContext(input.state);
  const AngularRateContext rates =
      BuildAngularRateContext(input.state, frame, input.omega_ib_b_raw);

  const Vector3d &lever_arm = input.state.lever_arm;
  const Vector3d v_wheel_b = frame.v_b + rates.omega_nb_b.cross(lever_arm);
  const Vector3d v_v = input.C_b_v * v_wheel_b;

  model.y << -v_v.y(), -v_v.z();

  Matrix3d H_v = Matrix3d::Zero();
  Matrix3d H_theta = Matrix3d::Zero();
  if (vel_mode == VelJacobianMode::kInEkf) {
    H_v = input.C_b_v;
    H_theta = input.C_b_v * Skew(frame.v_b);
  } else {
    H_v = input.C_b_v * frame.C_bn.transpose();
    H_theta = -input.C_b_v * Skew(frame.v_b) * frame.C_bn.transpose();
  }

  const Matrix3d H_bg =
      input.C_b_v * Skew(lever_arm) * rates.sf_g.asDiagonal();
  const Matrix3d H_sg =
      input.C_b_v * Skew(lever_arm) *
      (rates.omega_ib_unbiased.cwiseProduct(
           rates.sf_g.cwiseProduct(rates.sf_g)))
          .asDiagonal();

  model.H.block<2, 3>(0, StateIdx::kVel) = H_v.block<2, 3>(1, 0);
  model.H.block<2, 3>(0, StateIdx::kAtt) = H_theta.block<2, 3>(1, 0);
  model.H.block<2, 3>(0, StateIdx::kBg) = H_bg.block<2, 3>(1, 0);
  model.H.block<2, 3>(0, StateIdx::kSg) = H_sg.block<2, 3>(1, 0);

  const Vector3d e_pitch_v(0.0, 1.0, 0.0);
  const Vector3d e_yaw_v(0.0, 0.0, 1.0);
  const Vector3d dv_dpitch = -e_pitch_v.cross(v_v);
  const Vector3d dv_dyaw = -e_yaw_v.cross(v_v);
  model.H(0, StateIdx::kMountPitch) = dv_dpitch(1);
  model.H(0, StateIdx::kMountYaw) = dv_dyaw(1);
  model.H(1, StateIdx::kMountPitch) = dv_dpitch(2);
  model.H(1, StateIdx::kMountYaw) = dv_dyaw(2);

  const Matrix3d H_lever = input.C_b_v * Skew(rates.omega_nb_b);
  model.H.block<2, 3>(0, StateIdx::kLever) = H_lever.block<2, 3>(1, 0);

  if (lever_arm.norm() > 1e-3) {
    Matrix3d H_theta_lever = Matrix3d::Zero();
    if (vel_mode == VelJacobianMode::kInEkf) {
      H_theta_lever =
          input.C_b_v * Skew(lever_arm) * Skew(rates.omega_in_b);
    } else {
      H_theta_lever =
          -input.C_b_v * Skew(lever_arm) * frame.C_bn.transpose() *
          Skew(rates.omega_in_n);
    }
    model.H.block<2, 3>(0, StateIdx::kAtt) +=
        H_theta_lever.block<2, 3>(1, 0);
  }

  if (runtime_inekf && vel_mode != VelJacobianMode::kInEkf) {
    TransformAdditiveCoreJacobianToInEkf(model.H, frame.C_bn);
  }

  model.R = Matrix2d::Zero();
  model.R(0, 0) = input.sigma_nhc_y * input.sigma_nhc_y;
  model.R(1, 1) = input.sigma_nhc_z * input.sigma_nhc_z;
  return model;
}

MeasurementLinearization BuildOdoMeasurement(
    const OdoMeasurementInput &input) {
  using namespace measurement_model_detail;

  const bool runtime_inekf = UseInEkf(input.context);
  const VelJacobianMode vel_mode = ResolveVelJacobianMode(input.context);
  MeasurementLinearization model =
      MakeMeasurementLinearization(input.context, 1, "ODO", "VEHICLE");

  const MeasurementFrameContext frame =
      BuildMeasurementFrameContext(input.state);
  const AngularRateContext rates =
      BuildAngularRateContext(input.state, frame, input.omega_ib_b_raw);

  const Vector3d &lever_arm = input.state.lever_arm;
  const Vector3d v_wheel_b = frame.v_b + rates.omega_nb_b.cross(lever_arm);
  const Vector3d v_phys_v = input.C_b_v * v_wheel_b;
  const double pred_reading = input.state.odo_scale * v_phys_v.x();

  model.y(0) = input.odo_speed - pred_reading;

  const double s = input.state.odo_scale;

  RowVector3d H_v_phys = RowVector3d::Zero();
  if (vel_mode == VelJacobianMode::kInEkf) {
    H_v_phys = input.C_b_v.row(0);
  } else {
    H_v_phys = (input.C_b_v * frame.C_bn.transpose()).row(0);
  }
  model.H.block<1, 3>(0, StateIdx::kVel) = s * H_v_phys;

  Matrix3d H_theta_full = Matrix3d::Zero();
  if (vel_mode == VelJacobianMode::kInEkf) {
    H_theta_full = input.C_b_v * Skew(frame.v_b);
  } else {
    H_theta_full =
        -input.C_b_v * Skew(frame.v_b) * frame.C_bn.transpose();
  }

  if (lever_arm.norm() > 1e-3) {
    if (vel_mode == VelJacobianMode::kInEkf) {
      H_theta_full +=
          input.C_b_v * Skew(lever_arm) * Skew(rates.omega_in_b);
    } else {
      H_theta_full +=
          -input.C_b_v * Skew(lever_arm) * frame.C_bn.transpose() *
          Skew(rates.omega_in_n);
    }
  }
  model.H.block<1, 3>(0, StateIdx::kAtt) = s * H_theta_full.row(0);

  const RowVector3d H_bg_phys =
      input.C_b_v.row(0) * Skew(lever_arm) * rates.sf_g.asDiagonal();
  model.H.block<1, 3>(0, StateIdx::kBg) = s * H_bg_phys;

  const RowVector3d H_sg_phys =
      input.C_b_v.row(0) * Skew(lever_arm) *
      (rates.omega_ib_unbiased.cwiseProduct(
           rates.sf_g.cwiseProduct(rates.sf_g)))
          .asDiagonal();
  model.H.block<1, 3>(0, StateIdx::kSg) = s * H_sg_phys;

  model.H(0, StateIdx::kOdoScale) = v_phys_v.x();

  const Vector3d e_pitch_v(0.0, 1.0, 0.0);
  const Vector3d e_yaw_v(0.0, 0.0, 1.0);
  const Vector3d dv_dpitch = -e_pitch_v.cross(v_phys_v);
  const Vector3d dv_dyaw = -e_yaw_v.cross(v_phys_v);
  model.H(0, StateIdx::kMountPitch) = s * dv_dpitch(0);
  model.H(0, StateIdx::kMountYaw) = s * dv_dyaw(0);

  const RowVector3d H_lever_phys =
      input.C_b_v.row(0) * Skew(rates.omega_nb_b);
  model.H.block<1, 3>(0, StateIdx::kLever) = s * H_lever_phys;

  if (runtime_inekf && vel_mode != VelJacobianMode::kInEkf) {
    TransformAdditiveCoreJacobianToInEkf(model.H, frame.C_bn);
  }

  model.R(0, 0) = input.sigma_odo * input.sigma_odo;
  return model;
}

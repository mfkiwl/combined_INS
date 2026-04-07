// Nominal INS propagation builder shared by legacy wrappers and refactored APIs.
#pragma once

#include <Eigen/Dense>

#include "navigation/state_defs.h"

using namespace Eigen;

struct PropagationResult {
  State state;
  Matrix3d Cbn = Matrix3d::Identity();
  Vector3d f_b = Vector3d::Zero();
  Vector3d omega_b = Vector3d::Zero();
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

PropagationResult BuildNominalPropagation(const State &state,
                                          const ImuData &imu_prev,
                                          const ImuData &imu_curr);

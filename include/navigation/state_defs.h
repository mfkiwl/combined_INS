// Navigation state and noise definitions shared across filter layers.
#pragma once

#include <array>
#include <string>

#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

/**
 * InEKF runtime config.
 */
struct InEkfConfig {
  bool enabled = false;
  bool apply_covariance_floor_after_reset = false;
  Vector3d p_init_ecef = Vector3d::Zero();
  bool ri_gnss_pos_use_p_ned_local = true;
  int ri_vel_gyro_noise_mode = -1;
  bool ri_inject_pos_inverse = true;
  string debug_force_process_model = "auto";
  string debug_force_vel_jacobian = "auto";
  bool debug_disable_true_reset_gamma = false;
  bool debug_enable_standard_reset_gamma = false;

  void Enable(bool flag) { enabled = flag; }
  bool IsEnabled() const { return enabled; }
};

using InEkfManager = InEkfConfig;

/**
 * ESKF nominal state definition.
 * p/v/q are position, velocity and attitude quaternion (wxyz);
 * ba/bg are accelerometer and gyro biases.
 */
struct State {
  Vector3d p{Vector3d::Zero()};
  Vector3d v{Vector3d::Zero()};
  Vector4d q{1.0, 0.0, 0.0, 0.0};
  Vector3d ba{Vector3d::Zero()};
  Vector3d bg{Vector3d::Zero()};
  Vector3d sg{Vector3d::Zero()};
  Vector3d sa{Vector3d::Zero()};
  double odo_scale = 1.0;
  double mounting_roll = 0.0;
  double mounting_pitch = 0.0;
  double mounting_yaw = 0.0;
  Vector3d lever_arm{Vector3d::Zero()};
  Vector3d gnss_lever_arm{Vector3d::Zero()};
};

/**
 * Process and measurement noise parameters.
 */
struct NoiseParams {
  double sigma_acc = 0.0;
  double sigma_gyro = 0.0;
  double sigma_ba = 0.0;
  double sigma_bg = 0.0;
  double sigma_sg = 0.0;
  double sigma_sa = 0.0;
  Vector3d sigma_ba_vec = Vector3d::Constant(-1.0);
  Vector3d sigma_bg_vec = Vector3d::Constant(-1.0);
  Vector3d sigma_sg_vec = Vector3d::Constant(-1.0);
  Vector3d sigma_sa_vec = Vector3d::Constant(-1.0);
  double sigma_odo_scale = 0.0;
  double sigma_mounting = 0.0;
  double sigma_mounting_roll = -1.0;
  double sigma_mounting_pitch = -1.0;
  double sigma_mounting_yaw = -1.0;
  double sigma_lever_arm = 0.0;
  double sigma_gnss_lever_arm = 0.0;
  Vector3d sigma_lever_arm_vec = Vector3d::Constant(-1.0);
  Vector3d sigma_gnss_lever_arm_vec = Vector3d::Constant(-1.0);
  double sigma_uwb = 0.0;
  double sigma_gnss_pos = 0.0;
  double markov_corr_time = 0.0;
  bool disable_nominal_ba_bg_decay = false;
};

// State dimension: p(3)+v(3)+phi(3)+ba(3)+bg(3)+sg(3)+sa(3)+odo_scale(1)
// +mounting(3)+lever(3)+gnss_lever(3)=31.
constexpr int kStateDim = 31;
constexpr int kActualStateDim = 31;
constexpr int kPredictDebugCommonDim = 21;

using StateMask = std::array<bool, kStateDim>;
using StateGainScale = std::array<double, kStateDim>;
using StateMeasurementGainScale = Eigen::MatrixXd;

// 31D state index catalog.
struct StateIdx {
  static constexpr int kPos = 0;
  static constexpr int kVel = 3;
  static constexpr int kAtt = 6;
  static constexpr int kBa = 9;
  static constexpr int kBg = 12;
  static constexpr int kSg = 15;
  static constexpr int kSa = 18;
  static constexpr int kOdoScale = 21;
  static constexpr int kMountRoll = 22;
  static constexpr int kMountPitch = 23;
  static constexpr int kMountYaw = 24;
  static constexpr int kLever = 25;
  static constexpr int kGnssLever = 28;
};

/**
 * Per-sample IMU increment data.
 */
struct ImuData {
  double t = 0.0;
  double dt = 0.0;
  Vector3d dtheta{Vector3d::Zero()};
  Vector3d dvel{Vector3d::Zero()};
};

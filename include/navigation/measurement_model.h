// Unified measurement-model builders shared across legacy wrappers and audits.
#pragma once

#include <string>

#include <Eigen/Dense>

#include "navigation/filter_contracts.h"
#include "utils/math_utils.h"

using namespace std;
using namespace Eigen;

struct MeasurementLinearization {
  VectorXd y;
  MatrixXd H;
  MatrixXd R;
  string model_name;
  string frame_tag;
  ResidualConvention residual_convention =
      ResidualConvention::kMeasurementMinusPrediction;
};

struct MeasurementModelContext {
  FilterSemantics semantics = BuildStandardEskfSemantics();
  Vector3d p_init_ecef = Vector3d::Zero();
  bool ri_gnss_pos_use_p_ned_local = true;
  string debug_force_vel_jacobian = "auto";
};

MeasurementModelContext BuildMeasurementModelContext(
    const FilterSemantics &semantics);
MeasurementModelContext BuildMeasurementModelContextFromInEkfConfig(
    const InEkfManager *inekf);

struct UwbMeasurementInput {
  State state;
  VectorXd z;
  MatrixXd anchors;
  double sigma_uwb = 0.0;
  MeasurementModelContext context;
};

struct ZuptMeasurementInput {
  State state;
  double sigma_zupt = 0.0;
  MeasurementModelContext context;
};

struct NhcMeasurementInput {
  State state;
  Matrix3d C_b_v = Matrix3d::Identity();
  Vector3d omega_ib_b_raw = Vector3d::Zero();
  double sigma_nhc_y = 0.0;
  double sigma_nhc_z = 0.0;
  MeasurementModelContext context;
};

struct OdoMeasurementInput {
  State state;
  double odo_speed = 0.0;
  Matrix3d C_b_v = Matrix3d::Identity();
  Vector3d omega_ib_b_raw = Vector3d::Zero();
  double sigma_odo = 0.0;
  MeasurementModelContext context;
};

struct GnssPositionMeasurementInput {
  State state;
  Vector3d z_ecef = Vector3d::Zero();
  Vector3d sigma_gnss = Vector3d::Zero();
  MeasurementModelContext context;
};

struct GnssVelocityMeasurementInput {
  State state;
  Vector3d z_gnss_vel_ecef = Vector3d::Zero();
  Vector3d omega_ib_b_raw = Vector3d::Zero();
  Vector3d sigma_gnss_vel = Vector3d::Zero();
  MeasurementModelContext context;
};

MeasurementLinearization BuildUwbMeasurement(
    const UwbMeasurementInput &input);

MeasurementLinearization BuildZuptMeasurement(
    const ZuptMeasurementInput &input);

MeasurementLinearization BuildNhcMeasurement(
    const NhcMeasurementInput &input);

MeasurementLinearization BuildOdoMeasurement(
    const OdoMeasurementInput &input);

MeasurementLinearization BuildGnssPositionMeasurement(
    const GnssPositionMeasurementInput &input);

MeasurementLinearization BuildGnssVelocityMeasurement(
    const GnssVelocityMeasurementInput &input);

namespace measurement_model_detail {

enum class VelJacobianMode {
  kEskf,
  kInEkf,
};

struct MeasurementFrameContext {
  Llh llh{0.0, 0.0, 0.0};
  Matrix3d R_ne = Matrix3d::Identity();
  Matrix3d C_bn = Matrix3d::Identity();
  Vector3d v_ned = Vector3d::Zero();
  Vector3d v_b = Vector3d::Zero();
};

struct AngularRateContext {
  Vector3d omega_ie_n = Vector3d::Zero();
  Vector3d omega_en_n = Vector3d::Zero();
  Vector3d omega_in_n = Vector3d::Zero();
  Vector3d omega_in_b = Vector3d::Zero();
  Vector3d omega_ib_unbiased = Vector3d::Zero();
  Vector3d sf_g = Vector3d::Ones();
  Vector3d omega_ib_corr = Vector3d::Zero();
  Vector3d omega_nb_b = Vector3d::Zero();
};

MeasurementLinearization MakeMeasurementLinearization(
    const MeasurementModelContext &context,
    int rows,
    const string &model_name,
    const string &frame_tag);

MeasurementFrameContext BuildMeasurementFrameContext(const State &state);
AngularRateContext BuildAngularRateContext(
    const State &state,
    const MeasurementFrameContext &frame,
    const Vector3d &omega_ib_b_raw);

bool UseInEkf(const MeasurementModelContext &context);
VelJacobianMode ResolveVelJacobianMode(
    const MeasurementModelContext &context);
void TransformAdditiveCoreJacobianToInEkf(MatrixXd &H,
                                          const Matrix3d &C_bn);

}  // namespace measurement_model_detail

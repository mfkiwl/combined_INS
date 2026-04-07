#include "core/uwb.h"

#include "navigation/filter_contracts.h"
#include "navigation/measurement_model.h"

using namespace std;
using namespace Eigen;

namespace MeasModels {

namespace {

UwbModel ToLegacyUwbModel(const MeasurementLinearization &model) {
  UwbModel legacy;
  legacy.y = model.y;
  legacy.H = model.H;
  legacy.R = model.R;
  return legacy;
}

VelConstraintModel ToLegacyVelConstraintModel(
    const MeasurementLinearization &model) {
  VelConstraintModel legacy;
  legacy.y = model.y;
  legacy.H = model.H;
  legacy.R = model.R;
  return legacy;
}

}  // namespace

UwbModel ComputeUwbModel(const State &state,
                         const VectorXd &z,
                         const MatrixXd &anchors,
                         double sigma_uwb) {
  UwbMeasurementInput input;
  input.state = state;
  input.z = z;
  input.anchors = anchors;
  input.sigma_uwb = sigma_uwb;
  input.context = BuildMeasurementModelContext(BuildStandardEskfSemantics());
  return ToLegacyUwbModel(BuildUwbMeasurement(input));
}

VelConstraintModel ComputeZuptModel(const State &state, double sigma_zupt) {
  ZuptMeasurementInput input;
  input.state = state;
  input.sigma_zupt = sigma_zupt;
  input.context = BuildMeasurementModelContext(BuildStandardEskfSemantics());
  return ToLegacyVelConstraintModel(BuildZuptMeasurement(input));
}

VelConstraintModel ComputeNhcModel(const State &state,
                                   const Matrix3d &C_b_v,
                                   const Vector3d &omega_ib_b_raw,
                                   double sigma_nhc_y,
                                   double sigma_nhc_z,
                                   const InEkfManager *inekf) {
  NhcMeasurementInput input;
  input.state = state;
  input.C_b_v = C_b_v;
  input.omega_ib_b_raw = omega_ib_b_raw;
  input.sigma_nhc_y = sigma_nhc_y;
  input.sigma_nhc_z = sigma_nhc_z;
  input.context = BuildMeasurementModelContextFromInEkfConfig(inekf);
  return ToLegacyVelConstraintModel(BuildNhcMeasurement(input));
}

VelConstraintModel ComputeOdoModel(const State &state,
                                   double odo_speed,
                                   const Matrix3d &C_b_v,
                                   const Vector3d &omega_ib_b_raw,
                                   double sigma_odo,
                                   const InEkfManager *inekf) {
  OdoMeasurementInput input;
  input.state = state;
  input.odo_speed = odo_speed;
  input.C_b_v = C_b_v;
  input.omega_ib_b_raw = omega_ib_b_raw;
  input.sigma_odo = sigma_odo;
  input.context = BuildMeasurementModelContextFromInEkfConfig(inekf);
  return ToLegacyVelConstraintModel(BuildOdoMeasurement(input));
}

UwbModel ComputeGnssPositionModel(const State &state,
                                  const Vector3d &z,
                                  const Vector3d &sigma_gnss,
                                  const InEkfManager *inekf) {
  GnssPositionMeasurementInput input;
  input.state = state;
  input.z_ecef = z;
  input.sigma_gnss = sigma_gnss;
  input.context = BuildMeasurementModelContextFromInEkfConfig(inekf);
  return ToLegacyUwbModel(BuildGnssPositionMeasurement(input));
}

UwbModel ComputeGnssVelocityModel(const State &state,
                                  const Vector3d &z_gnss_vel,
                                  const Vector3d &omega_ib_b_raw,
                                  const Vector3d &sigma_gnss_vel,
                                  const InEkfManager *inekf) {
  GnssVelocityMeasurementInput input;
  input.state = state;
  input.z_gnss_vel_ecef = z_gnss_vel;
  input.omega_ib_b_raw = omega_ib_b_raw;
  input.sigma_gnss_vel = sigma_gnss_vel;
  input.context = BuildMeasurementModelContextFromInEkfConfig(inekf);
  return ToLegacyUwbModel(BuildGnssVelocityMeasurement(input));
}

}  // namespace MeasModels

#include "navigation/measurement_model.h"

using namespace std;
using namespace Eigen;

namespace measurement_model_detail {

MeasurementLinearization MakeMeasurementLinearization(
    const MeasurementModelContext &context,
    int rows,
    const string &model_name,
    const string &frame_tag) {
  MeasurementLinearization model;
  model.y = VectorXd::Zero(rows);
  model.H = MatrixXd::Zero(rows, kStateDim);
  model.R = MatrixXd::Zero(rows, rows);
  model.model_name = model_name;
  model.frame_tag = frame_tag;
  model.residual_convention = context.semantics.residual_convention;
  return model;
}

}  // namespace measurement_model_detail

MeasurementModelContext BuildMeasurementModelContext(
    const FilterSemantics &semantics) {
  MeasurementModelContext context;
  context.semantics = semantics;
  return context;
}

MeasurementModelContext BuildMeasurementModelContextFromInEkfConfig(
    const InEkfManager *inekf) {
  if (inekf == nullptr) {
    return BuildMeasurementModelContext(BuildStandardEskfSemantics());
  }
  MeasurementModelContext context;
  context.semantics = BuildFilterSemanticsFromInEkfConfig(*inekf);
  context.p_init_ecef = inekf->p_init_ecef;
  context.ri_gnss_pos_use_p_ned_local = inekf->ri_gnss_pos_use_p_ned_local;
  context.debug_force_vel_jacobian = inekf->debug_force_vel_jacobian;
  return context;
}

namespace measurement_model_detail {

MeasurementFrameContext BuildMeasurementFrameContext(const State &state) {
  MeasurementFrameContext frame;
  frame.llh = EcefToLlh(state.p);
  frame.R_ne = RotNedToEcef(frame.llh);
  frame.C_bn = frame.R_ne.transpose() * QuatToRot(state.q);
  frame.v_ned = frame.R_ne.transpose() * state.v;
  frame.v_b = frame.C_bn.transpose() * frame.v_ned;
  return frame;
}

AngularRateContext BuildAngularRateContext(
    const State &state,
    const MeasurementFrameContext &frame,
    const Vector3d &omega_ib_b_raw) {
  AngularRateContext ctx;
  ctx.omega_ie_n = OmegaIeNed(frame.llh.lat);
  ctx.omega_en_n = OmegaEnNed(frame.v_ned, frame.llh.lat, frame.llh.h);
  ctx.omega_in_n = ctx.omega_ie_n + ctx.omega_en_n;
  ctx.omega_in_b = frame.C_bn.transpose() * ctx.omega_in_n;
  ctx.omega_ib_unbiased = omega_ib_b_raw - state.bg;
  ctx.sf_g = (Vector3d::Ones() + state.sg).cwiseInverse();
  ctx.omega_ib_corr = ctx.sf_g.cwiseProduct(ctx.omega_ib_unbiased);
  ctx.omega_nb_b = ctx.omega_ib_corr - ctx.omega_in_b;
  return ctx;
}

bool UseInEkf(const MeasurementModelContext &context) {
  return context.semantics.flavor != FilterFlavor::kStandardEskf;
}

VelJacobianMode ResolveVelJacobianMode(
    const MeasurementModelContext &context) {
  if (context.debug_force_vel_jacobian == "eskf") {
    return VelJacobianMode::kEskf;
  }
  if (context.debug_force_vel_jacobian == "inekf") {
    return VelJacobianMode::kInEkf;
  }
  if (UseInEkf(context)) {
    return VelJacobianMode::kInEkf;
  }
  return VelJacobianMode::kEskf;
}

void TransformAdditiveCoreJacobianToInEkf(MatrixXd &H,
                                          const Matrix3d &C_bn) {
  if (H.cols() < StateIdx::kAtt + 3) {
    return;
  }
  MatrixXd H_pos = H.block(0, StateIdx::kPos, H.rows(), 3).eval();
  MatrixXd H_vel = H.block(0, StateIdx::kVel, H.rows(), 3).eval();
  MatrixXd H_att = H.block(0, StateIdx::kAtt, H.rows(), 3).eval();

  H.block(0, StateIdx::kPos, H.rows(), 3) = H_pos * C_bn;
  H.block(0, StateIdx::kVel, H.rows(), 3) = H_vel * C_bn;
  H.block(0, StateIdx::kAtt, H.rows(), 3) = H_att * (-C_bn);
}

}  // namespace measurement_model_detail

MeasurementLinearization BuildZuptMeasurement(
    const ZuptMeasurementInput &input) {
  MeasurementLinearization model =
      measurement_model_detail::MakeMeasurementLinearization(
          input.context, 3, "ZUPT", "NED");
  const auto frame =
      measurement_model_detail::BuildMeasurementFrameContext(input.state);

  model.y = -frame.v_ned;
  model.H.block<3, 3>(0, StateIdx::kVel) = Matrix3d::Identity();
  model.R = Matrix3d::Identity() * (input.sigma_zupt * input.sigma_zupt);
  return model;
}

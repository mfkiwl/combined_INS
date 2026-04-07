#include "navigation/measurement_model.h"

using namespace std;
using namespace Eigen;

MeasurementLinearization BuildUwbMeasurement(
    const UwbMeasurementInput &input) {
  const int num_anchors = static_cast<int>(input.anchors.rows());
  MeasurementLinearization model =
      measurement_model_detail::MakeMeasurementLinearization(
          input.context, num_anchors, "UWB", "ECEF");

  VectorXd h = VectorXd::Zero(num_anchors);
  for (int i = 0; i < num_anchors; ++i) {
    const Vector3d r = input.state.p - input.anchors.row(i).transpose();
    const double dist = r.norm();
    h(i) = dist;
    if (dist > 1e-9) {
      model.H.block<1, 3>(i, StateIdx::kPos) = r.transpose() / dist;
    }
  }

  model.y = input.z - h;
  model.R =
      MatrixXd::Identity(num_anchors, num_anchors) *
      (input.sigma_uwb * input.sigma_uwb);
  return model;
}

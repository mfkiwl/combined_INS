#include "app/fusion.h"

FusionResult RunFusion(const FusionOptions &options, const Dataset &dataset,
                       const State &x0,
                       const Matrix<double, kStateDim, kStateDim> &P0,
                       FusionDebugCapture *debug_capture) {
  return RunFusionRuntime(options, dataset, x0, P0, debug_capture).result;
}

// Semantic contracts for filter conventions and state block catalog.
#pragma once

#include <map>
#include <string>

#include "navigation/state_defs.h"

using namespace std;

enum class FilterFlavor {
  kStandardEskf,
  kInEkf,
};

enum class ResidualConvention {
  kMeasurementMinusPrediction,
};

enum class ImuErrorConvention {
  kTrueMinusNominal,
};

enum class AttitudeErrorConvention {
  kStandardEskfRightError,
  kInvariantBodyError,
};

struct FilterSemantics {
  FilterFlavor flavor = FilterFlavor::kStandardEskf;
  ResidualConvention residual_convention =
      ResidualConvention::kMeasurementMinusPrediction;
  ImuErrorConvention imu_error_convention =
      ImuErrorConvention::kTrueMinusNominal;
  AttitudeErrorConvention attitude_error_convention =
      AttitudeErrorConvention::kStandardEskfRightError;
  bool additive_imu_error_injection = true;
};

struct StateBlockInfo {
  string name;
  int start = 0;
  int size = 0;
};

FilterSemantics BuildStandardEskfSemantics();
FilterSemantics BuildInEkfSemantics();
FilterSemantics BuildFilterSemanticsFromInEkfConfig(const InEkfConfig &cfg);
FilterSemantics BuildProcessSemanticsFromInEkfConfig(const InEkfConfig &cfg);

map<string, StateBlockInfo> BuildDefaultStateBlockCatalog();

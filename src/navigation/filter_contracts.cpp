#include "navigation/filter_contracts.h"

namespace {

StateBlockInfo MakeBlockInfo(const string &name, int start, int size) {
  StateBlockInfo info;
  info.name = name;
  info.start = start;
  info.size = size;
  return info;
}

}  // namespace

FilterSemantics BuildStandardEskfSemantics() {
  FilterSemantics semantics;
  semantics.flavor = FilterFlavor::kStandardEskf;
  semantics.residual_convention =
      ResidualConvention::kMeasurementMinusPrediction;
  semantics.imu_error_convention = ImuErrorConvention::kTrueMinusNominal;
  semantics.attitude_error_convention =
      AttitudeErrorConvention::kStandardEskfRightError;
  semantics.additive_imu_error_injection = true;
  return semantics;
}

FilterSemantics BuildInEkfSemantics() {
  FilterSemantics semantics = BuildStandardEskfSemantics();
  semantics.flavor = FilterFlavor::kInEkf;
  semantics.attitude_error_convention =
      AttitudeErrorConvention::kInvariantBodyError;
  return semantics;
}

FilterSemantics BuildFilterSemanticsFromInEkfConfig(const InEkfConfig &cfg) {
  if (!cfg.enabled) {
    return BuildStandardEskfSemantics();
  }
  return BuildInEkfSemantics();
}

FilterSemantics BuildProcessSemanticsFromInEkfConfig(const InEkfConfig &cfg) {
  if (cfg.debug_force_process_model == "eskf") {
    return BuildStandardEskfSemantics();
  }
  if (cfg.debug_force_process_model == "inekf") {
    return BuildInEkfSemantics();
  }
  if (cfg.enabled) {
    return BuildInEkfSemantics();
  }
  return BuildStandardEskfSemantics();
}

map<string, StateBlockInfo> BuildDefaultStateBlockCatalog() {
  map<string, StateBlockInfo> blocks;
  blocks.emplace("pos", MakeBlockInfo("pos", StateIdx::kPos, 3));
  blocks.emplace("vel", MakeBlockInfo("vel", StateIdx::kVel, 3));
  blocks.emplace("att", MakeBlockInfo("att", StateIdx::kAtt, 3));
  blocks.emplace("ba", MakeBlockInfo("ba", StateIdx::kBa, 3));
  blocks.emplace("bg", MakeBlockInfo("bg", StateIdx::kBg, 3));
  blocks.emplace("sg", MakeBlockInfo("sg", StateIdx::kSg, 3));
  blocks.emplace("sa", MakeBlockInfo("sa", StateIdx::kSa, 3));
  blocks.emplace("odo_scale",
                 MakeBlockInfo("odo_scale", StateIdx::kOdoScale, 1));
  blocks.emplace("mounting_roll",
                 MakeBlockInfo("mounting_roll", StateIdx::kMountRoll, 1));
  blocks.emplace("mounting_pitch",
                 MakeBlockInfo("mounting_pitch", StateIdx::kMountPitch, 1));
  blocks.emplace("mounting_yaw",
                 MakeBlockInfo("mounting_yaw", StateIdx::kMountYaw, 1));
  blocks.emplace("lever", MakeBlockInfo("lever", StateIdx::kLever, 3));
  blocks.emplace("gnss_lever",
                 MakeBlockInfo("gnss_lever", StateIdx::kGnssLever, 3));
  return blocks;
}

# Unified INS/GNSS Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the INS/GNSS integrated-navigation path so ESKF/InEKF semantics, process/measurement equations, runtime orchestration, and algorithm documentation are explicit, testable, and restorable without losing the recovered baseline behavior scale.

**Architecture:** Introduce a semantic-contract layer, separate nominal/process/measurement builders from the Kalman execution core, then split runtime scheduling out of `pipeline_fusion.cpp`. Preserve behavior by anchoring each phase to the recovered sparse-15s baseline and expanding regression coverage before migrating the next layer.

**Tech Stack:** C++17, Eigen, yaml-cpp, existing CMake targets (`eskf_fusion`, `regression_checks`, `jacobian_audit`, `odo_nhc_bgz_jacobian_fd`), markdown documentation under `docs/`.

---

## File Structure

### Create

- `include/navigation/state_defs.h`
- `include/navigation/filter_contracts.h`
- `include/navigation/nominal_propagation.h`
- `include/navigation/process_model.h`
- `include/navigation/measurement_model.h`
- `include/navigation/filter_engine.h`
- `src/navigation/filter_contracts.cpp`
- `src/navigation/nominal_propagation.cpp`
- `src/navigation/process_model_standard_eskf.cpp`
- `src/navigation/process_model_inekf.cpp`
- `src/navigation/process_noise_mapping.cpp`
- `src/navigation/measurement_model_common.cpp`
- `src/navigation/measurement_model_gnss.cpp`
- `src/navigation/measurement_model_road.cpp`
- `src/navigation/measurement_model_uwb.cpp`
- `src/navigation/filter_engine.cpp`
- `src/navigation/filter_reset.cpp`
- `src/app/fusion_runtime.cpp`
- `src/app/fusion_scheduler.cpp`
- `src/app/fusion_update_runner.cpp`
- `src/app/fusion_phase_controls.cpp`
- `src/app/fusion_diagnostics_runtime.cpp`
- `docs/algorithm/ins_gnss_filter_algorithm.md`

### Modify

- `CMakeLists.txt`
- `include/core/eskf.h`
- `include/core/uwb.h`
- `include/app/fusion.h`
- `src/core/eskf_engine.cpp`
- `src/core/ins_mech.cpp`
- `src/core/measurement_models_uwb.cpp`
- `src/app/pipeline_fusion.cpp`
- `src/app/initialization.cpp`
- `src/app/config.cpp`
- `src/app/diagnostics.cpp`
- `apps/regression_checks_main.cpp`
- `apps/jacobian_audit_main.cpp`
- `apps/odo_nhc_bgz_jacobian_fd_main.cpp`

### Keep Intact Unless Forced

- `src/app/dataset_loader.cpp`
- `src/app/evaluation.cpp`
- `apps/eskf_fusion_main.cpp`
- `apps/data_converter_main.cpp`
- `apps/uwb_generator_main.cpp`

---

### Task 1: Create Semantic Contract Layer

**Files:**
- Create: `include/navigation/state_defs.h`
- Create: `include/navigation/filter_contracts.h`
- Create: `src/navigation/filter_contracts.cpp`
- Modify: `include/core/eskf.h`
- Modify: `include/core/uwb.h`
- Modify: `include/app/fusion.h`
- Test: `apps/regression_checks_main.cpp`

- [ ] **Step 1: Write failing contract tests**

Add tests to `apps/regression_checks_main.cpp` that fail until the new semantics are explicit:

```cpp
void TestStandardEskfSemanticContract() {
  FilterSemantics semantics = BuildStandardEskfSemantics();
  Expect(semantics.residual_convention ==
             ResidualConvention::kMeasurementMinusPrediction,
         "standard ESKF residual must be y = z - h");
  Expect(semantics.imu_error_convention ==
             ImuErrorConvention::kTrueMinusNominal,
         "IMU bias/scale contract must be delta_b = b_true - b_hat");
  Expect(semantics.attitude_error_convention ==
             AttitudeErrorConvention::kStandardEskfRightError,
         "standard ESKF attitude contract must match q_true ~= q_hat ⊗ Exp(-phi)");
}

void TestStateBlockCatalogMatchesLegacyStateIndices() {
  auto blocks = BuildDefaultStateBlockCatalog();
  Expect(blocks.at("ba").start == StateIdx::kBa, "ba block index drifted");
  Expect(blocks.at("bg").start == StateIdx::kBg, "bg block index drifted");
  Expect(blocks.at("sg").start == StateIdx::kSg, "sg block index drifted");
  Expect(blocks.at("sa").start == StateIdx::kSa, "sa block index drifted");
}
```

- [ ] **Step 2: Run the contract tests to verify failure**

Run:

```powershell
cmake --build build --config Release --target regression_checks
build\Release\regression_checks.exe
```

Expected: FAIL because `FilterSemantics` and the new builders do not exist yet.

- [ ] **Step 3: Implement the semantic-contract layer**

Create explicit semantic types and keep a compatibility bridge in `include/core/eskf.h`:

```cpp
enum class FilterFlavor {
  kStandardEskf,
  kHybridInEkf,
  kTrueInEkf,
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
```

Keep `State`, `NoiseParams`, and `StateIdx` reachable from old includes through forwarding includes or aliases so the first migration phase does not break the whole tree.

- [ ] **Step 4: Run contract tests to verify pass**

Run:

```powershell
cmake --build build --config Release --target regression_checks
build\Release\regression_checks.exe
```

Expected: PASS, with the new contract tests included in the output.

- [ ] **Step 5: Commit**

```powershell
git add include/navigation/state_defs.h include/navigation/filter_contracts.h src/navigation/filter_contracts.cpp include/core/eskf.h include/core/uwb.h include/app/fusion.h apps/regression_checks_main.cpp CMakeLists.txt
git commit -m "refactor: introduce navigation semantic contracts"
```

### Task 2: Extract Nominal Propagation and Process Builders

**Files:**
- Create: `include/navigation/nominal_propagation.h`
- Create: `include/navigation/process_model.h`
- Create: `src/navigation/nominal_propagation.cpp`
- Create: `src/navigation/process_model_standard_eskf.cpp`
- Create: `src/navigation/process_model_inekf.cpp`
- Create: `src/navigation/process_noise_mapping.cpp`
- Modify: `src/core/ins_mech.cpp`
- Modify: `include/core/eskf.h`
- Modify: `apps/regression_checks_main.cpp`
- Test: `apps/regression_checks_main.cpp`

- [ ] **Step 1: Write failing process-builder tests**

Extend `apps/regression_checks_main.cpp` with builder-level tests:

```cpp
void TestStandardEskfProcessBuilderMatchesFiniteDifference() {
  auto semantics = BuildStandardEskfSemantics();
  auto analytic = BuildProcessLinearization(input, semantics);
  auto numeric = BuildFiniteDifferenceProcessLinearization(input, semantics);
  ExpectMaxAbsNear(analytic.F_vba, numeric.F_vba, 5.0e-6,
                   "F_v,ba drifted from finite-difference reference");
  ExpectMaxAbsNear(analytic.F_phibg, numeric.F_phibg, 1.2e-1,
                   "F_phi,bg drifted from finite-difference reference");
}
```

- [ ] **Step 2: Run the tests to verify failure**

Run:

```powershell
cmake --build build --config Release --target regression_checks
build\Release\regression_checks.exe
```

Expected: FAIL because `BuildProcessLinearization()` and the new propagation builders do not exist.

- [ ] **Step 3: Implement nominal propagation and process builders**

Move mechanization and process logic behind explicit builder types:

```cpp
struct ProcessModelInput {
  State nominal;
  ImuData imu_prev;
  ImuData imu_curr;
  NoiseParams noise;
  FilterSemantics semantics;
};

struct ProcessLinearization {
  PropagationResult propagation;
  Matrix<double, kStateDim, kStateDim> Phi;
  Matrix<double, kStateDim, kStateDim> Qd;
  Matrix<double, kStateDim, kStateDim> F;
};

ProcessLinearization BuildStandardEskfProcessLinearization(
    const ProcessModelInput& input);
ProcessLinearization BuildInEkfProcessLinearization(
    const ProcessModelInput& input);
```

Keep `InsMech::Propagate()` and `InsMech::BuildProcessModel()` as thin compatibility wrappers in this task, forwarding to the new builders instead of owning the math directly.

- [ ] **Step 4: Re-run process tests**

Run:

```powershell
cmake --build build --config Release --target regression_checks
build\Release\regression_checks.exe
```

Expected: PASS, including the existing `F_v,ba/F_v,sa/F_phi,bg/F_phi,sg` checks.

- [ ] **Step 5: Commit**

```powershell
git add include/navigation/nominal_propagation.h include/navigation/process_model.h src/navigation/nominal_propagation.cpp src/navigation/process_model_standard_eskf.cpp src/navigation/process_model_inekf.cpp src/navigation/process_noise_mapping.cpp src/core/ins_mech.cpp apps/regression_checks_main.cpp CMakeLists.txt
git commit -m "refactor: isolate nominal propagation and process builders"
```

### Task 3: Unify Measurement-Model Construction

**Files:**
- Create: `include/navigation/measurement_model.h`
- Create: `src/navigation/measurement_model_common.cpp`
- Create: `src/navigation/measurement_model_gnss.cpp`
- Create: `src/navigation/measurement_model_road.cpp`
- Create: `src/navigation/measurement_model_uwb.cpp`
- Modify: `src/core/measurement_models_uwb.cpp`
- Modify: `include/core/uwb.h`
- Modify: `apps/jacobian_audit_main.cpp`
- Modify: `apps/odo_nhc_bgz_jacobian_fd_main.cpp`
- Modify: `apps/regression_checks_main.cpp`
- Test: `apps/jacobian_audit_main.cpp`
- Test: `apps/odo_nhc_bgz_jacobian_fd_main.cpp`

- [ ] **Step 1: Write failing measurement-contract tests**

Add tests that verify the new unified return type and residual contract:

```cpp
void TestGnssMeasurementContractUsesMeasurementMinusPrediction() {
  auto semantics = BuildStandardEskfSemantics();
  MeasurementLinearization meas =
      BuildGnssPositionMeasurement(input, semantics);
  Expect(meas.residual_convention ==
             ResidualConvention::kMeasurementMinusPrediction,
         "GNSS measurement residual convention drifted");
}
```

Also update `jacobian_audit_main.cpp` and `odo_nhc_bgz_jacobian_fd_main.cpp` to compile against a temporary stub `MeasurementLinearization` path so they fail until the new builders are wired.

- [ ] **Step 2: Run the audits to verify failure**

Run:

```powershell
cmake --build build --config Release --target jacobian_audit odo_nhc_bgz_jacobian_fd regression_checks
build\Release\regression_checks.exe
```

Expected: build or runtime failure because the new measurement interface is not fully implemented.

- [ ] **Step 3: Implement unified measurement builders**

Create a common interface:

```cpp
struct MeasurementLinearization {
  VectorXd y;
  MatrixXd H;
  MatrixXd R;
  string model_name;
  string frame_tag;
  ResidualConvention residual_convention =
      ResidualConvention::kMeasurementMinusPrediction;
};

MeasurementLinearization BuildGnssPositionMeasurement(
    const GnssPositionInput& input, const FilterSemantics& semantics);
MeasurementLinearization BuildGnssVelocityMeasurement(
    const GnssVelocityInput& input, const FilterSemantics& semantics);
MeasurementLinearization BuildOdoMeasurement(
    const OdoInput& input, const FilterSemantics& semantics);
MeasurementLinearization BuildNhcMeasurement(
    const NhcInput& input, const FilterSemantics& semantics);
MeasurementLinearization BuildUwbMeasurement(
    const UwbInput& input, const FilterSemantics& semantics);
```

Keep `MeasModels::Compute*Model()` as compatibility wrappers that return the old `UwbModel` / `VelConstraintModel` shapes while delegating to the new builders.

- [ ] **Step 4: Run Jacobian and measurement tests**

Run:

```powershell
cmake --build build --config Release --target jacobian_audit odo_nhc_bgz_jacobian_fd regression_checks
build\Release\regression_checks.exe
build\Release\jacobian_audit.exe --outdir output\_tmp_jacobian_refactor_check
build\Release\odo_nhc_bgz_jacobian_fd.exe --outdir output\_tmp_odo_nhc_fd_refactor_check
```

Expected:

- `regression_checks: PASS`
- audit executables complete and write `summary.md` files under the temporary output directories.

- [ ] **Step 5: Commit**

```powershell
git add include/navigation/measurement_model.h src/navigation/measurement_model_common.cpp src/navigation/measurement_model_gnss.cpp src/navigation/measurement_model_road.cpp src/navigation/measurement_model_uwb.cpp src/core/measurement_models_uwb.cpp include/core/uwb.h apps/jacobian_audit_main.cpp apps/odo_nhc_bgz_jacobian_fd_main.cpp apps/regression_checks_main.cpp CMakeLists.txt
git commit -m "refactor: unify measurement-model builders"
```

### Task 4: Slim the Filter Execution Core

**Files:**
- Create: `include/navigation/filter_engine.h`
- Create: `src/navigation/filter_engine.cpp`
- Create: `src/navigation/filter_reset.cpp`
- Modify: `src/core/eskf_engine.cpp`
- Modify: `include/core/eskf.h`
- Modify: `apps/regression_checks_main.cpp`
- Test: `apps/regression_checks_main.cpp`

- [ ] **Step 1: Write failing execution-core tests**

Add focused tests around injection and reset:

```cpp
void TestFilterEngineUsesAdditiveImuErrorInjection() {
  NavigationFilterEngine engine(noise, BuildStandardEskfSemantics());
  engine.Initialize(state, P0);
  auto meas = BuildIdentityInjectionMeasurement(dx_truth);
  VectorXd dx_est;
  bool updated = engine.Correct(meas, &dx_est);
  Expect(updated, "correction should succeed");
  ExpectNearVec(engine.state().bg, state.bg + dx_truth.segment<3>(StateIdx::kBg),
                1.0e-12, "bg injection sign regressed");
}
```

- [ ] **Step 2: Run the tests to verify failure**

Run:

```powershell
cmake --build build --config Release --target regression_checks
build\Release\regression_checks.exe
```

Expected: FAIL because `NavigationFilterEngine` and the new reset modules do not exist.

- [ ] **Step 3: Implement the new execution core**

Introduce a flavor-aware but semantics-light engine:

```cpp
class NavigationFilterEngine {
 public:
  explicit NavigationFilterEngine(const NoiseParams& noise,
                                  const FilterSemantics& semantics);
  void Initialize(const State& state,
                  const Matrix<double, kStateDim, kStateDim>& P0);
  bool Predict(const ProcessLinearization& process);
  bool Correct(const MeasurementLinearization& measurement,
               VectorXd* dx_out = nullptr);
};
```

Update `src/core/eskf_engine.cpp` to become a thin adapter or retire it entirely after callers are migrated.

- [ ] **Step 4: Run execution-core tests**

Run:

```powershell
cmake --build build --config Release --target regression_checks
build\Release\regression_checks.exe
```

Expected: PASS, including existing round-trip and reset checks.

- [ ] **Step 5: Commit**

```powershell
git add include/navigation/filter_engine.h src/navigation/filter_engine.cpp src/navigation/filter_reset.cpp src/core/eskf_engine.cpp include/core/eskf.h apps/regression_checks_main.cpp CMakeLists.txt
git commit -m "refactor: isolate kalman execution core"
```

### Task 5: Split Runtime Orchestration Out of `pipeline_fusion.cpp`

**Files:**
- Create: `src/app/fusion_runtime.cpp`
- Create: `src/app/fusion_scheduler.cpp`
- Create: `src/app/fusion_update_runner.cpp`
- Create: `src/app/fusion_phase_controls.cpp`
- Create: `src/app/fusion_diagnostics_runtime.cpp`
- Modify: `src/app/pipeline_fusion.cpp`
- Modify: `include/app/fusion.h`
- Modify: `src/app/diagnostics.cpp`
- Modify: `src/app/config.cpp`
- Modify: `src/app/initialization.cpp`
- Test: `apps/regression_checks_main.cpp`
- Test: `apps/eskf_fusion_main.cpp`

- [ ] **Step 1: Write a failing runtime smoke test**

Add a regression smoke that uses the exact GNSS-splitting path through the refactored runtime entrypoint:

```cpp
void TestFusionRuntimeStillUpdatesGnssAtGnssTimestamp() {
  FusionRuntime runtime = BuildFusionRuntime(options);
  auto stats = runtime.RunTimingAudit(dataset);
  Expect(stats.gnss_split_count > 0, "exact GNSS splitting disappeared");
  Expect(stats.align_prev_count == 0, "GNSS aligned to previous IMU unexpectedly");
}
```

- [ ] **Step 2: Run the smoke test to verify failure**

Run:

```powershell
cmake --build build --config Release --target regression_checks eskf_fusion
build\Release\regression_checks.exe
```

Expected: FAIL because `FusionRuntime` and the new orchestration entrypoints do not exist.

- [ ] **Step 3: Implement the runtime split**

Move scheduling and orchestration responsibilities into focused units:

```cpp
struct FusionRuntimeContext {
  FusionOptions options;
  Dataset dataset;
  NavigationFilterEngine engine;
};

bool RunPredictStep(FusionRuntimeContext& ctx, size_t imu_idx);
bool RunMeasurementWindow(FusionRuntimeContext& ctx, size_t imu_idx);
void SyncRuntimePhaseControls(FusionRuntimeContext& ctx, double t_now);
```

Leave `RunFusion()` in `include/app/fusion.h` as the public compatibility entrypoint, but make it delegate into the new runtime units.

- [ ] **Step 4: Run runtime smoke checks**

Run:

```powershell
cmake --build build --config Release --target regression_checks eskf_fusion
build\Release\regression_checks.exe
build\Release\eskf_fusion.exe --config config_data2_baseline_eskf.yaml
```

Expected:

- `regression_checks: PASS`
- `eskf_fusion` completes and writes the configured outputs without crashing.

- [ ] **Step 5: Commit**

```powershell
git add src/app/fusion_runtime.cpp src/app/fusion_scheduler.cpp src/app/fusion_update_runner.cpp src/app/fusion_phase_controls.cpp src/app/fusion_diagnostics_runtime.cpp src/app/pipeline_fusion.cpp include/app/fusion.h src/app/diagnostics.cpp src/app/config.cpp src/app/initialization.cpp apps/regression_checks_main.cpp CMakeLists.txt
git commit -m "refactor: split fusion runtime orchestration"
```

### Task 6: Write the Canonical Algorithm Document

**Files:**
- Create: `docs/algorithm/ins_gnss_filter_algorithm.md`
- Modify: `docs/README.md`
- Test: `docs/algorithm/ins_gnss_filter_algorithm.md`

- [ ] **Step 1: Write the document skeleton before filling equations**

Create the document with the required top-level sections:

```markdown
# INS/GNSS Filter Algorithm

## 1. Pipeline Order
## 2. State Vector and Block Map
## 3. Coordinate Frames and Rotation Notation
## 4. Nominal Propagation
## 5. Standard ESKF Error-State Model
## 6. InEKF Error-State Model
## 7. Process Model Blocks
## 8. Measurement Models
## 9. Residual Convention
## 10. Injection and Reset
## 11. Noise Definitions
## 12. Differences vs KF-GINS
## 13. Code Mapping Table
## 14. Fragile Sign Checklist
```

- [ ] **Step 2: Validate the document against code while it is still incomplete**

Run:

```powershell
Select-String -Path docs\algorithm\ins_gnss_filter_algorithm.md -Pattern "Fragile Sign Checklist","Differences vs KF-GINS","Residual Convention"
```

Expected: all required headings exist before the equations are filled in.

- [ ] **Step 3: Fill in the final code-aligned content**

Document the actual contracts, not generic textbook formulas. Each fragile sign-sensitive item must reference both the semantic contract and the implementing file/function.

- [ ] **Step 4: Link the document from the docs index**

Update `docs/README.md` so the algorithm document is discoverable from the repo docs landing page.

- [ ] **Step 5: Sanity-check the finished document**

Run:

```powershell
Get-Content docs\algorithm\ins_gnss_filter_algorithm.md
```

Expected: the document renders as a coherent, code-restoration reference and includes a code mapping table.

- [ ] **Step 6: Commit**

```powershell
git add docs/algorithm/ins_gnss_filter_algorithm.md docs/README.md
git commit -m "docs: add canonical ins gnss filter algorithm reference"
```

### Task 7: Run Acceptance Replays and Write Migration Notes

**Files:**
- Modify: `docs/algorithm/ins_gnss_filter_algorithm.md`
- Modify: `docs/README.md`
- Modify: `walkthrough.md`
- Test: `output/data5_kf_sparse15s_cmp_r3/` baseline comparison

- [ ] **Step 1: Build all relevant targets**

Run:

```powershell
cmake --build build --config Release --target eskf_fusion regression_checks jacobian_audit odo_nhc_bgz_jacobian_fd
```

Expected: build succeeds with no compile errors.

- [ ] **Step 2: Run contract and Jacobian checks**

Run:

```powershell
build\Release\regression_checks.exe
build\Release\jacobian_audit.exe --outdir output\_tmp_refactor_jacobian_audit
build\Release\odo_nhc_bgz_jacobian_fd.exe --outdir output\_tmp_refactor_odo_nhc_fd
```

Expected:

- `regression_checks: PASS`
- both audit tools complete and produce summary artifacts.

- [ ] **Step 3: Replay the recovered sparse-15s anchor**

Run:

```powershell
build\Release\eskf_fusion.exe --config output\data5_kf_sparse15s_cmp_r3\current_solver\artifacts\cases\data5_full_ins_gnss_sparse15s_current_solver\config_data5_full_ins_gnss_sparse15s_current_solver.yaml
```

Expected: result stays in the same scale as the anchor:

- `RMSE3D` around `0.068141 m`
- `final_3d` around `0.018918 m`

Small drift is acceptable; catastrophic drift is not.

- [ ] **Step 4: Replay official acceptance cases**

Run the user-designated `data5` outage and `data2` baseline/outage cases after the structural refactor is stable.
Record exact commands and fresh metrics in `walkthrough.md`.

- [ ] **Step 5: Update migration notes**

Append a short migration section to `docs/algorithm/ins_gnss_filter_algorithm.md` documenting:

- old API to new API mapping;
- which legacy wrappers still exist;
- which files are now authoritative for semantics, process builders, measurement builders, and runtime orchestration.

- [ ] **Step 6: Commit**

```powershell
git add walkthrough.md docs/algorithm/ins_gnss_filter_algorithm.md docs/README.md
git commit -m "chore: validate refactor against acceptance baselines"
```

### Task 8: Final Cleanup and Legacy Removal

**Files:**
- Modify: `include/core/eskf.h`
- Modify: `include/core/uwb.h`
- Modify: `src/core/eskf_engine.cpp`
- Modify: `src/core/ins_mech.cpp`
- Modify: `src/core/measurement_models_uwb.cpp`
- Modify: `src/app/pipeline_fusion.cpp`
- Test: full build and acceptance commands from Tasks 1-7

- [ ] **Step 1: Remove dead compatibility code that is no longer exercised**

Delete or shrink legacy wrappers only after all migrated callers and tests are green.

- [ ] **Step 2: Run the final full verification set**

Run:

```powershell
cmake --build build --config Release --target eskf_fusion regression_checks jacobian_audit odo_nhc_bgz_jacobian_fd
build\Release\regression_checks.exe
build\Release\jacobian_audit.exe --outdir output\_tmp_final_jacobian_audit
build\Release\odo_nhc_bgz_jacobian_fd.exe --outdir output\_tmp_final_odo_nhc_fd
build\Release\eskf_fusion.exe --config output\data5_kf_sparse15s_cmp_r3\current_solver\artifacts\cases\data5_full_ins_gnss_sparse15s_current_solver\config_data5_full_ins_gnss_sparse15s_current_solver.yaml
```

Expected:

- all builds pass;
- `regression_checks: PASS`;
- audits complete;
- sparse-15s replay remains in the recovered baseline scale.

- [ ] **Step 3: Commit**

```powershell
git add include/core/eskf.h include/core/uwb.h src/core/eskf_engine.cpp src/core/ins_mech.cpp src/core/measurement_models_uwb.cpp src/app/pipeline_fusion.cpp
git commit -m "refactor: remove legacy estimator glue after migration"
```

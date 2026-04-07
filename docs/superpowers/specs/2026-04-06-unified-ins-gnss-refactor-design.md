# Unified INS/GNSS ESKF-InEKF Refactor Design

**Date:** `2026-04-06`  
**Status:** `Draft for user review`  
**Scope:** `INS/GNSS integrated-navigation path, shared ESKF/InEKF semantics, runtime orchestration, algorithm documentation`

## 1. Goal

Refactor the current INS/GNSS combined-navigation implementation so that:

- filter semantics are explicit and centralized rather than implicit and distributed;
- the standard ESKF and InEKF paths share a unified vocabulary and contract surface;
- process-model and measurement-model equations are structurally separated from runtime scheduling and experiment logic;
- future accidental sign or convention edits are caught by design, tests, and documentation rather than by long experiment forensics;
- post-refactor behavior stays in the same performance scale as the recovered sparse-15s baseline:
  `RMSE3D ~= 0.068141 m`, `final_3d ~= 0.018918 m`.

This is an aggressive cleanup. Internal interfaces may change. Small numerical drift is acceptable, but recovered-baseline behavior must remain in the same order of magnitude.

## 2. Problem Statement

The current codebase has recovered the main INS/GNSS sign issue, but the root cause was structural rather than local:

- state semantics, error-state semantics, residual direction, and error injection are spread across multiple files;
- standard ESKF and InEKF logic are interleaved in large functions, making it easy to change one convention while silently violating another;
- runtime orchestration in `src/app/pipeline_fusion.cpp` mixes scheduling, measurement timing, ablation controls, and filter mathematics;
- critical sign-sensitive terms such as `F_v,ba`, `F_v,sa`, `F_phi,bg`, `F_phi,sg`, and IMU nominal error injection depend on implicit contracts not represented as first-class code entities;
- the project lacks a code-aligned algorithm reference that could be used to restore correct equations after accidental edits.

The current recovered sign contract is evidence-backed and must be treated as the semantic anchor for refactoring:

- residual convention: `y = z - h(x_hat)`
- IMU bias/scale error convention: `delta_b = b_true - b_hat`
- standard ESKF attitude error convention: `q_true ~= q_hat ⊗ Exp(-phi_e)`
- IMU nominal error injection: additive, `state.ba/bg/sg/sa += d*`
- standard ESKF process-model signs:
  - `F_v,ba < 0`
  - `F_v,sa < 0`
  - `F_phi,bg > 0`
  - `F_phi,sg > 0`

These contracts must become explicit architecture, not just comments.

## 3. Non-Goals

This refactor does not aim to:

- redesign estimator theory beyond clarifying the already selected ESKF/InEKF contracts;
- optimize runtime performance as a primary objective;
- change the fundamental state dimension (`kStateDim = 31`) during the first refactor phase;
- retune process or measurement noise as part of structural cleanup;
- merge all experiment scripts and outputs into a new experiment framework.

If later work changes the mathematical model itself, that must happen after this refactor stabilizes.

## 4. Design Principles

### 4.1 Semantic First

Any sign-sensitive or convention-sensitive logic must be represented as an explicit contract in code.
No critical convention should exist only as a comment inside a large implementation function.

### 4.2 Separate Meaning From Execution

The code must distinguish between:

- what the state and error mean;
- how process and measurement equations are built;
- when the runtime chooses to execute a given prediction or update.

### 4.3 Shared Vocabulary Across ESKF and InEKF

ESKF and InEKF must not be treated as entirely separate systems.
They share state storage, measurement entrypoints, diagnostics, and experiment runtime.
The code should centralize what is shared, and isolate only the genuine mathematical branch points.

### 4.4 Equation Builders Must Be Testable

Process-model and measurement-model builders must be small enough to test with:

- round-trip checks;
- finite-difference checks;
- sign-contract assertions;
- end-to-end acceptance replays.

### 4.5 Documentation Must Be Restorative

The algorithm document must be precise enough that a future developer can recover the correct code after accidental edits without re-running a full research investigation.

## 5. Proposed Architecture

The refactored architecture is organized into five layers.

### 5.1 Layer A: Semantic Contracts

**Purpose:** define what the estimator is estimating and what conventions apply.

**New files**

- `include/navigation/state_defs.h`
- `include/navigation/filter_contracts.h`
- `src/navigation/filter_contracts.cpp` if needed for helpers/string conversion

**Responsibilities**

- nominal state definition;
- state-index layout and named state blocks;
- coordinate-frame declarations and transform-direction conventions;
- error-state conventions for standard ESKF, hybrid InEKF, and true InEKF;
- residual convention;
- nominal error injection convention;
- covariance reset convention;
- documented branch points between filter flavors.

**Key types**

- `enum class FilterFlavor { StandardEskf, HybridInEkf, TrueInEkf }`
- `enum class ResidualConvention { MeasurementMinusPrediction }`
- `enum class ImuErrorConvention { TrueMinusNominal }`
- `enum class AttitudeErrorConvention { StandardEskfRightError, InEkfInvariantError }`
- `struct FilterSemantics`
- `struct StateBlockInfo`

**Expected outcome**

Any process-model or measurement-model builder receives an explicit semantic contract instead of assuming hidden conventions.

### 5.2 Layer B: Nominal Navigation and Process Models

**Purpose:** build nominal propagation and process equations without runtime orchestration noise.

**New files**

- `include/navigation/nominal_propagation.h`
- `include/navigation/process_model.h`
- `src/navigation/nominal_propagation.cpp`
- `src/navigation/process_model_standard_eskf.cpp`
- `src/navigation/process_model_inekf.cpp`
- `src/navigation/process_noise_mapping.cpp`

**Responsibilities**

- IMU correction and mechanization;
- nominal propagation in the selected navigation representation;
- continuous-time `F` construction for each filter flavor;
- discrete-time `Phi/Qd` construction;
- explicit noise mapping `G/Qc`;
- centralization of fragile blocks such as:
  - `F_v,ba`
  - `F_v,sa`
  - `F_phi,bg`
  - `F_phi,sg`

**Refactor rule**

`InsMech::Propagate` and `InsMech::BuildProcessModel` must no longer remain as a single mixed implementation with flavor branches scattered throughout a large function.
They should become thin adapters to flavor-specific builders or be fully replaced.

### 5.3 Layer C: Measurement Models

**Purpose:** unify measurement construction around a single explicit model contract.

**New files**

- `include/navigation/measurement_model.h`
- `src/navigation/measurement_model_common.cpp`
- `src/navigation/measurement_model_gnss.cpp`
- `src/navigation/measurement_model_odo.cpp`
- `src/navigation/measurement_model_nhc.cpp`
- `src/navigation/measurement_model_uwb.cpp`

**Common return object**

Each measurement builder returns a single structure, for example:

```cpp
struct MeasurementLinearization {
  VectorXd y;
  MatrixXd H;
  MatrixXd R;
  string model_name;
  string frame_tag;
  ResidualConvention residual_convention;
};
```

**Responsibilities**

- build `h(x)`-aligned residuals using the unified residual contract;
- build `H` under the correct filter flavor semantics;
- attach enough metadata for diagnostics and audit logging;
- eliminate duplicated measurement semantics spread across app and core layers.

**Special attention**

GNSS position and velocity models must explicitly state:

- whether they use ECEF or NED intermediate linearization;
- which attitude error contract they assume;
- how lever-arm Jacobians map to the current state blocks;
- how their Jacobian sign differs from the `KF-GINS` implementation because of residual convention differences.

### 5.4 Layer D: Filter Execution Core

**Purpose:** own the Kalman predict/correct/update mechanics only.

**New files**

- `include/navigation/filter_engine.h`
- `src/navigation/filter_engine.cpp`
- optionally `include/navigation/filter_reset.h`
- optionally `src/navigation/filter_reset.cpp`

**Responsibilities**

- maintain state/covariance storage;
- execute `predict`;
- execute `correct`;
- apply Joseph covariance update;
- apply error injection;
- apply reset;
- apply state masks and covariance floors;
- expose debug snapshots.

**Refactor rule**

The execution core must not build GNSS/ODO/NHC/UWB Jacobians itself.
It consumes `MeasurementLinearization` and process-model outputs from other layers.

### 5.5 Layer E: Runtime Orchestration

**Purpose:** keep experiment/runtime controls out of filter math.

**New files**

- `src/app/fusion_runtime.cpp`
- `src/app/fusion_scheduler.cpp`
- `src/app/fusion_update_runner.cpp`
- `src/app/fusion_phase_controls.cpp`
- `src/app/fusion_diagnostics_runtime.cpp`

**Responsibilities**

- dataset stepping and IMU/GNSS alignment;
- update scheduling and exact GNSS-time splitting;
- phase controls, ablation controls, and post-GNSS behavior;
- runtime diagnostics and result recording;
- experiment-facing compatibility wrappers.

**Refactor rule**

The runtime layer decides *when* a measurement model is evaluated and *whether* an update is allowed.
It must not decide *what a state block means* or *what sign a Jacobian should carry*.

## 6. Compatibility and Interface Strategy

The user has allowed interface changes, but the refactor should still preserve a controlled migration path.

### 6.1 Stable User-Facing Behavior During Transition

During the first implementation pass:

- keep executable names unchanged;
- keep key experiment scripts runnable;
- preserve output file schemas unless a documented migration is required;
- provide compatibility wrappers from old APIs to new builders where feasible.

### 6.2 Internal API Migration

Old internal entrypoints may be replaced, but only through staged migration:

1. introduce new semantic and equation builders;
2. route old code through adapters;
3. move callers to new APIs;
4. remove dead legacy helpers after regression confirmation.

### 6.3 Config Migration

Config changes are allowed, but should follow two rules:

- avoid changing mathematical meaning silently;
- if field names or grouping change, provide a migration note in the new algorithm document and a compatibility parser where inexpensive.

## 7. Detailed Refactor Strategy by Phase

### Phase 1: Semantic Extraction

**Deliverables**

- semantic contracts files created;
- state-block catalog centralized;
- residual/injection/reset conventions made explicit;
- current code updated to reference contracts rather than implicit assumptions.

**Why first**

This phase reduces the risk of repeating sign regressions before any major file movement begins.

### Phase 2: Process-Model Isolation

**Deliverables**

- nominal propagation extracted;
- standard ESKF process builder extracted;
- InEKF process builder extracted;
- shared noise mapping isolated;
- existing callers redirected.

**Acceptance criteria**

- finite-difference checks for key `F` blocks pass;
- existing regression checks still pass.

### Phase 3: Measurement-Model Isolation

**Deliverables**

- GNSS, ODO, NHC, and UWB model builders moved behind a unified interface;
- residual direction and Jacobian semantics centralized;
- measurement debug metadata standardized.

**Acceptance criteria**

- Jacobian finite-difference checks pass for critical models;
- exact GNSS-time update behavior remains intact.

### Phase 4: Filter-Core Cleanup

**Deliverables**

- `EskfEngine` replaced or slimmed into a flavor-aware but semantics-light execution core;
- injection and reset logic isolated and directly testable;
- debug snapshots preserved or improved.

**Acceptance criteria**

- round-trip and injection/reset checks pass;
- state/covariance update path no longer depends on measurement type specifics.

### Phase 5: Runtime Orchestration Split

**Deliverables**

- `pipeline_fusion.cpp` decomposed into runtime-focused units;
- measurement timing, schedule, and ablation logic isolated from estimator math;
- runtime diagnostics remain reproducible.

**Acceptance criteria**

- no regression in GNSS exact-time splitting behavior;
- existing acceptance scripts still execute with small allowed numerical drift.

### Phase 6: Algorithm Documentation Finalization

**Deliverables**

- complete markdown algorithm document under `docs/algorithm/`;
- code-to-equation mapping table;
- migration notes from old structure to new structure.

**Acceptance criteria**

- a developer can map every fragile sign-sensitive block to a documented formula and contract;
- the document is sufficient to restore the correct code after accidental edits.

## 8. Testing Strategy

This refactor requires more than one type of test.

### 8.1 Contract Tests

Extend `apps/regression_checks_main.cpp` with tests for:

- residual convention consistency;
- additive IMU error injection consistency;
- standard ESKF attitude-error round-trip;
- InEKF error-coordinate round-trip where applicable;
- reset gamma contract consistency.

### 8.2 Finite-Difference Process Tests

Numerically validate:

- `F_v,ba`
- `F_v,sa`
- `F_phi,bg`
- `F_phi,sg`
- selected InEKF-specific process blocks.

### 8.3 Finite-Difference Measurement Tests

Numerically validate Jacobian blocks for:

- GNSS position;
- GNSS velocity if enabled;
- ODO;
- NHC;
- any shared lever-arm related terms.

### 8.4 Runtime Timing Tests

Preserve and extend checks that prove:

- GNSS updates happen at exact `t_gnss`;
- IMU interval splitting remains correct;
- runtime scheduling does not degrade update timing.

### 8.5 Acceptance Replays

Use the following as acceptance anchors:

- recovered sparse-15s comparison:
  `output/data5_kf_sparse15s_cmp_r3/`
- official data5 outage reference case;
- user-designated data2 INS/GNSS baseline;
- user-designated data2 outage case.

Acceptance rule:

- same order of magnitude as the recovered baseline;
- no return of systematic opposite-direction bias/scale evolution;
- no catastrophic regression.

## 9. Algorithm Documentation Design

The new algorithm document should live at:

- `docs/algorithm/ins_gnss_filter_algorithm.md`

It should be written as a code-restoration reference, not a general tutorial.

### Required Sections

1. system scope and pipeline order;
2. state vector definition and block map;
3. coordinate frames and rotation notation;
4. nominal INS propagation;
5. standard ESKF error-state definition;
6. InEKF error-state definition and divergence points;
7. process-model equations and block formulas;
8. measurement equations for GNSS/ODO/NHC/UWB;
9. residual convention;
10. injection and reset equations;
11. noise definitions and how `Qd` is constructed;
12. exact differences versus `KF-GINS`;
13. code mapping table from equations to source files/tests;
14. known fragile sign points and the rationale for each sign.

### Documentation Rule

Every sign-sensitive equation must be traceable to:

- a semantic contract;
- a code builder;
- a regression test.

## 10. File Migration Map

This section defines the intended source migration.

### Existing Files to Shrink or Replace

- `include/core/eskf.h`
  - split into semantic contracts, filter engine interfaces, and process-model declarations.
- `src/core/eskf_engine.cpp`
  - move execution-only parts into filter core;
  - move semantic assumptions out.
- `src/core/ins_mech.cpp`
  - split mechanization from process-model builders.
- `src/app/pipeline_fusion.cpp`
  - decompose into runtime orchestration units.

### Existing Files Expected to Remain but Change Responsibility

- `src/app/dataset_loader.cpp`
  - likely stays largely intact.
- `src/app/initialization.cpp`
  - remains, but should consume centralized semantic/state definitions.
- `src/app/config.cpp`
  - remains, but config semantics should align with new contracts.

## 11. Risks and Mitigations

### Risk 1: Structural cleanup changes behavior unintentionally

**Mitigation**

- keep acceptance anchor metrics visible throughout;
- refactor in phases;
- preserve thin adapters during migration;
- run regression checks after each phase.

### Risk 2: ESKF/InEKF shared abstraction becomes too generic and hides important differences

**Mitigation**

- centralize only semantic common ground;
- keep flavor-specific builders explicit;
- do not force one formula path to emulate all flavors through flags alone.

### Risk 3: Runtime split breaks exact GNSS alignment

**Mitigation**

- isolate GNSS-time split logic into a tested runtime unit;
- preserve dedicated timing regression tests.

### Risk 4: Documentation drifts away from code

**Mitigation**

- write code mapping tables into the document;
- require each fragile contract to be tied to a builder and a regression test.

## 12. Success Criteria

The refactor is successful when all of the following are true:

- critical semantics are represented by code contracts, not hidden comments;
- process and measurement equations are isolated from runtime orchestration;
- standard ESKF and InEKF share a unified vocabulary with explicit branch points;
- sign-sensitive Jacobian terms are covered by tests;
- acceptance replays remain in the same performance scale as the recovered baseline;
- the new markdown algorithm document is sufficient to restore the correct equations after accidental edits.

## 13. Recommended Implementation Order

1. create semantic contract layer;
2. move state definitions and contracts out of `include/core/eskf.h`;
3. isolate process-model builders;
4. isolate measurement-model builders;
5. slim filter execution core;
6. split runtime orchestration;
7. finalize algorithm document and migration notes;
8. rerun acceptance experiments.

## 14. Review Questions for the User

Before implementation, confirm:

- this spec covers the intended scope of the aggressive cleanup;
- the proposed output document path is acceptable;
- the phased migration order matches the preferred risk profile;
- no additional experiment compatibility constraints need to be preserved.

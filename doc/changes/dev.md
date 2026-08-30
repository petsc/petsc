# Changes: Development

<!---
% STYLE GUIDELINES:
% * Capitalize sentences
% * Use imperative, e.g., Add, Improve, Change, etc.
% * Don't use a period (.) at the end of entries
% * If multiple sentences are needed, use a period or semicolon to divide sentences, but not at the end of the final sentence
--->

## General

- Increase the minimum required CUDA Toolkit version to 11.4
- Change the policy so that documentation changes/fixes/additions are added to the `main` branch, not the `release` branch
- Change `petscdiff` to treat ` < ` in expected output as text that must match instead of suppressing differences from numerical output
- Change the `make branch-review` rule to `[PETSC_LLM_CLI=command] [PETSC_LLM_MODEL=modelname] make branch-review`. Add support for Gemini, Codex, OpenCode, and other LLM CLIs.
  Replace the use of `CLAUDE_OPTS` with `PETSC_LLM_CLI_OPTS` and `PETSC_LLM_MODEL`
- Add a CodeGraph skill and repository guidance so LLM coding tools automatically use an existing local PETSc CodeGraph index when navigating or reviewing source
- Add `comm` and `prefix` as initial arguments to `PetscOptionsDeprecatedNoObject()`

## Configure/Build

- Increase the minimum required Python version for `./configure` to 3.6
- Add `providesDocs` and `docsDirs` package attributes so that an external package's sources can be cloned and scanned to generate PETSc manual pages when the documentation is built
- Add interface to LIBXSMM

## Sys

- Add `PetscGetConfiguration()`
- Add `PetscObjectViewSynchronizedFromOptions()`
- Add `PetscSetMPIThreadRequiredType()`
- Add `PetscOverrideIntelMKLCPUVendorDetection()`
- Deprecate `PETSC_MPI_THREAD_REQUIRED`
- Add support for running PETSc applications as MCP servers that can be accessed by LLMs
- Add `PetscRandomAppendOptionsPrefix()` and `PetscRandomGetOptionsPrefix()`
- Add `PetscCallLAPACKInfo()` for calling LAPACK routines with an `info` argument where the caller requires an `info` value of 0 for the program to continue
- Add `PetscIntCSRView()` to inspect CSR graph

## Event Logging

- Change `-log` and `-log_all` to only take optional filename arguments. They no longer take optional boolean argument

## PetscViewer

- Add support for writing CGNS descriptors on the base node: `PetscViewerCGNSGetDescriptors()`, `PetscViewerCGNSRestoreDescriptors()`, `PetscViewerCGNSSetDescriptor()`
- Add `PetscViewerVTKWriteFn` as the typedef prototype for the `write()` function passed to `PetscViewerVTKAddField()`. This addition requires no changes to user source code

## PetscDraw


## AO


## IS


## VecScatter / PetscSF


## PF


## Vec

- Add `VecCreateSeqWithArrayAndMemType()` and `VecCreateMPIWithArrayAndMemType()` to create array-style standard, CUDA, or HIP vectors from memory of a specified `PetscMemType`
- Add `VecSetStdBasis()` API to set a vector to the i-th standard basis vector
- Change the behavior of `VecPointwiseDivide()` implementing w = x / y: if a particular `y[i]` is zero and `x[i]` is also zero, `w[i]` is set to one (before it was set to zero).
- Deprecate `-vec_view_stash` in favor of `-vec_stash_view`

## PetscSection


## PetscPartitioner


## Mat

- Add `MATPRODUCT_PtAP` support for `MATDIAGONAL` and `MATCONSTANTDIAGONAL`
- Add `MATPRODUCT_AB` support for `MATDIAGONAL` and `MATCONSTANTDIAGONAL` with any matrix type
- Add `MatSeqAIJGetKokkosView()`, `MatSeqAIJRestoreKokkosView()`, `MatSeqAIJGetKokkosViewWrite()` and `MatSeqAIJRestoreKokkosViewWrite()` to the public API
- Change `MatSeqAIJCUSPARSEGetIJ()`, `MatSeqAIJCUSPARSERestoreIJ()`, `MatSeqAIJHIPSPARSEGetIJ()` and `MatSeqAIJHIPSPARSERestoreIJ()` to return `PetscInt` indices instead of `int`
- Add `MatNormApproximate()` to compute matrix norms approximately
- Add `MatGetMultPetscSF()` to access the `PetscSF` used to communicate off-process vector entries in `MatMult()` for `MATMPIAIJ`, `MATMPIBAIJ`, `MATMPISBAIJ`, `MATMPIDENSE`, and `MATMPISELL`
- Change `MATSEQSBAIJ` behavior in `MatPermute()` and `MatDiagonalScale()` to use `MATSEQBAIJ` when the result is not necessarily symmetric
- Change `MatGetValues()` to respect the row or column orientation set with `MatSetOption(mat, MAT_ROW_ORIENTED, ...)`. This will break current code that calls
  `MatSetOption(mat, MAT_ROW_ORIENTED, PETSC_FALSE)` and uses `MatGetValues()`
- Add new `MatType` `MATSEQBAIJLIBXSMM` and `MATMPIBAIJLIBXSMM`
- Add support for `MatSetInf()` with `MATSEQDENSE` and `MATMPIDENSE`; no `MatType` implemented it before
- Add device SpMM support for `MATPRODUCT_AB` and `MATPRODUCT_AtB` with a `MATAIJKOKKOS` matrix and dense matrices; previously these products looped `MatMult()` over the columns of the dense matrix
- Change `MatCreateMAIJ()` to convert its result to `MATAIJKOKKOS` when the input matrix has that type, as is already done for `MATAIJCUSPARSE`; the `MATMAIJ` kernels read the host arrays of the input matrix directly and so miss values last updated on device

## MatCoarsen


## PC

- Add `PCGAMGSetProlongatorFilter()` and `PCGAMGGetProlongatorFilter()` to set/get the threshold for filtering the prolongator in `PCGAMG`. The threshold is relative, applies to whole fine-node/coarse-node coupling blocks while preserving the near-null space, and must be in [0,1)
- Add `PCGAMGSetProlongatorFilterScale()` and `PCGAMGGetProlongatorFilterScale()` to set/get the per-level scaling of the prolongator filter threshold in `PCGAMG`; the scale must be in [0,1]
- `PCGAMGSetThresholdScale()` now requires its argument to be in [0,1]
- `PCGAMGSetThreshold()` now requires each threshold value to be less than 1; negative values still mean keeping even zero entries in the graph. It is also now `Logically Collective` (checked in debug builds), matching `PCGAMGSetThresholdScale()`
- Add `PCAIR` and `PCPFLAREINV` manual pages, generated from the PFLARE sources when the documentation is built
- Add `PCParametersInitialize`
- Fix `PCMG` to honor `PCSetUseAmat(pc, PETSC_FALSE)` at all levels
- Add `PCMatApplyRichardson()`, `PCMatApplyRichardsonExists()`, and `PCShellSetMatApplyRichardson()`, the block analogs of `PCApplyRichardson()`, `PCApplyRichardsonExists()`, and `PCShellSetApplyRichardson()`
- Add the missing Fortran binding for `PCShellSetMatApply()`
- Remove `PCSetDiagonalScale()`, `PCGetDiagonalScale()`, `PCDiagonalScaleLeft()`, and `PCDiagonalScaleRight()`

## KSP

- Fix for `KSP` pre- and post-solve callbacks, that can now be used together with Eisenstat and Walker trick for `SNES`
- Add `KSPPreSolve()` and `KSPPostSolve()` to run the registered `KSP` pre/post solve callbacks
- Change `KSPSolve()` to run the `KSPSetPreSolve()` callback after `KSPSetUp()` and `KSPSetUpOnBlocks()` instead of before
- Add `KSPIDR` — IDR(s) Induced Dimension Reduction Krylov solver (biorthogonal variant)
- Add `KSPIDRSetS()`, `KSPIDRGetS()`, `KSPIDRSetRandom()`, `KSPIDRGetRandom()`, `KSPIDRSetCosine()`, and `KSPIDRGetCosine()`
- Deprecate `KSPMonitorResidualShort()` and `-ksp_monitor_short`, remove the `preconditioned_residual_short` monitor registry name
- Remove `-ksp_plot_eigenvalues`, `-ksp_plot_eigenvalues_explicitly`, `-ksp_plot_eigencontours` that have been deprecated since version 3.9
- Remove `KSPSetDiagonalScale()`, `KSPGetDiagonalScale()`, `KSPSetDiagonalScaleFix()`, and `KSPGetDiagonalScaleFix()`, along with the `-ksp_diagonal_scale`, `-ksp_diagonal_scale_fix`, and `-ksp_view_diagonal_scale` options
- Add native support for `KSPMatSolve()` and `KSPMatSolveTranspose()` with `KSPRICHARDSON`, which iterates on batches of or the entire block of right-hand sides instead of solving them one at a time
- Add delegation of the `KSPRICHARDSON` block iteration to `PCMatApplyRichardson()` when the `PC` provides it, falling back to `PCApplyRichardson()` on one right-hand side at a time so that the result matches `KSPSolve()`
- Change `KSPConvergedDefault()` to base the relative tolerance on the Frobenius norm of the batch of (preconditioned) right-hand sides during a `KSPMatSolve()` with a nonzero initial guess, matching `KSPSolve()`
- Change `KSPMatSolve()` and `KSPMatSolveTranspose()` to reset the residual history at the start of each solve, and of each batch when `-ksp_matsolve_batch_size` is used, as `KSPSolve()` does, unless `KSPSetResidualHistory()` was called with `reset` set to `PETSC_FALSE`
- Change `KSPRichardsonSetSelfScale()` to trigger a `KSPSetUp()` re-run when the flag changes, fixing an out-of-bounds work vector access when it was set after `KSPSetUp()`
- GMRES orthogonalization routines have been promoted to the main `KSP` level, and GMRES has been dropped from the name. The new names are `KSPOrthogonalizationSet()`, `KSPOrthogonalizationGet()`, `KSPOrthogonalizationModifiedGramSchmidt()`, `KSPOrthogonalizationClassicalGramSchmidt()`, `KSPOrthogonalizationSetCGSRefinementType()`, `KSPOrthogonalizationGetCGSRefinementType()`. Note that the signature of orthogonalization functions has changed, see `KSPOrthogonalizationFn`. The related enumeration is now `KSP_ORTHOGONALIZATION_CGS_REFINE_*` and command-line options are now `-ksp_orthogonalization (cgs|mgs)` and `-ksp_orthogonalization_cgs_refinement_type (refine_never|refine_ifneeded|refine_always)`

## SNES

- Change `SNESSetUp()` to not overwrite the NPC application context if one has previously been set on the NPC
- Change `SNESComputeJacobian()` to call the user-provided Jacobian function when a left NPC is active and the solver is not `SNESASPIN`
- Add support for nonlinear preconditioners with a `DM` different from the parent `SNES` `DM`. Calling `SNESSetNPC()` will no longer enforce default parameters on the NPC
- Change `-snes_mf` to respect an explicitly set `PC` type instead of silently overriding it with `PCNONE`; an explicitly requested `PC` that requires an assembled matrix now errors
- Deprecate `SNESMonitorDefaultShort()` and `-snes_monitor_short`
- Add `SNESFASSetUseCoarseCorrectionLineSearch()` and `-snes_fas_use_coarse_correction_linesearch` to implement the algorithm in {cite}`nash2000mgopt`.
- Add `SNESFASGetCoarseCorrectionLineSearch()`
- Change default linesearch for `SNESFAS` with `SNESFASType` of `SNES_FAS_ADDITIVE` to `SNESLINESEARCHSECANT`

## SNESLineSearch

- Deprecate `SNESLINESEARCHBASIC` in favor of `SNESLINESEARCHNONE`
- Add `SNESLineSearchViewFromOptions()`

## TS

- Add `DMTSSetIFunctionPre()`
- Add `TSDiscGradSetImplicitFormulation()`
- Expose `TSDiscGradGetX0AndXdot()` and `TSDiscGradRestoreX0AndXdot()`
- Add `TSIsImplicit()` that indicates if the `TSType` is implicit and uses `SNES` or `KSP`

## TAO

- Add `TaoGetDM()` and `TaoSetDM()`
- Deprecate `TaoMonitorDefaultShort()`, `-tao_monitor_short`, and `-tao_monitor_short_interval`
- Change the deprecated `TaoSMonitor()` and `-tao_smonitor` to use the full-precision default monitor
- Add `TaoGetConvergedReasonString()` to retrieve a human readable string describing the `TaoConvergedReason`

## TaoTerm


## PetscRegressor


## PetscDA

- Add the `PetscDALETKFLocalizationType` enum (`PETSCDA_LETKF_LOC_NONE`, `PETSCDA_LETKF_LOC_GASPARI_COHN`, `PETSCDA_LETKF_LOC_GAUSSIAN`, `PETSCDA_LETKF_LOC_BOXCAR`) selecting the LETKF localization kernel
- Add `PetscDALETKFSetLocalizationType()`, `PetscDALETKFGetLocalizationType()`, `PetscDALETKFSetLocalizationRadius()`, `PetscDALETKFGetLocalizationRadius()`, and `PetscDALETKFSetLocalizationCoordinates()`; the localization matrix is built lazily from these distance-based kernel parameters
- Add `PetscDALETKFResetLocalization()` to drop the cached localization matrix so the next analysis rebuilds it from the current kernel parameters
- Remove `PETSCDAETKF`; use `PETSCDALETKF` with `PetscDALETKFSetLocalizationType(da, PETSCDA_LETKF_LOC_NONE)` for identical behavior
- Remove `PetscDAEnsembleSetSqrtType()`, `PetscDAEnsembleGetSqrtType()`, the `PetscDASqrtType` enum (`PETSCDA_SQRT_CHOLESKY`, `PETSCDA_SQRT_EIGEN`), and the `-petscda_ensemble_sqrt_type` option; the symmetric-eigendecomposition square root is now the only path
- Remove `PetscDALETKFSetLocalization()`; use the distance-based API `PetscDALETKFSetLocalizationType()`, `PetscDALETKFSetLocalizationRadius()`, and `PetscDALETKFSetLocalizationCoordinates()` instead
- Remove `PetscDALETKFSetObsPerVertex()` and `PetscDALETKFGetObsPerVertex()`; per-vertex observation counts are now derived from the distance-based localization kernel
- Remove `PetscDALETKFGetLocalizationMatrix()`; the localization matrix is an internal cached object built lazily on the first analysis. Callers that previously supplied this matrix should switch to `PetscDALETKFSetLocalizationCoordinates()` and let the implementation build the matrix from the chosen kernel
- Change the LETKF distance-based periodicity convention: per-axis periodicity is now activated by `bd[d] > 0.0` (the period), and negative `bd[d]` now raises `PETSC_ERR_ARG_OUTOFRANGE`; previously any non-zero `bd[d]` (including negative values) enabled periodicity
- Add `PetscDAEnsembleForecastFn` typedef for the `PetscDAEnsembleForecast()` model callback
- Change the `PetscDAEnsembleForecast()` model callback signature from `(Vec, Vec, PetscCtx)` to `(Mat, PetscCtx)`; the model now receives the entire ensemble matrix and advances all members in place. Existing per-member callbacks should iterate over the columns with `MatDenseGetColumnVec()`/`MatDenseRestoreColumnVec()` (see `ShallowWaterStep2D()` in `src/ml/da/tutorials/ex4.c`)
- Change `-petscda_view` to fire at the tail of every `PetscDAEnsembleAnalysis()` call (mirroring `KSPSolve()`/`SNESSolve()`), so it now emits once per analysis cycle rather than once per run; code that wants a single end-of-run snapshot should call `PetscDAView()` explicitly after the assimilation loop

## DM

- Change `DMLabelPropagatePush()` to take a reduce operator
- Add `DMKSPSetCreateOperators()` to let the `DM` provide a pair of application specific `Mat` objects to inner `KSP` solvers.
- Fix `DMGetLocalToGlobalMapping()` for a local section carrying a chart permutation (see `DMReorderSectionSetDefault()`); the map is now indexed by the local section offsets. The local section must now be set up with `PetscSectionSetUp()` before the mapping is built, otherwise an error is raised
- Fix `DMSetLocalSection()` and `DMSetGlobalSection()` to invalidate a previously built section-derived local-to-global mapping, which was computed from the old sections; a mapping built by the `DM` implementation itself, such as by `DMDA` in `DMSetUp()`, is kept. `DMFOREST` with `p4est` shares its sections into its cached `DMPLEX` without going through those setters, and now invalidates the mapping there as well

## DMSwarm

- Add `DMSwarmProjectFields()` and `DMSwarmProjectGradientFields()`
- Add `DMSwarmSort` class
- Add `DMSwarmSortDestroy()` and `DMSwarmSortView()`
- Allow `DMSwarmCellDMSetSort()` to take in `NULL` and clear the sort
- Add `DMSwarmPreallocateMassMatrix()` and `DMSwarmFillMassMatrix()`

## DMPlex

- Add `DMPlexSetClosurePermutationLexicographic()`
- Add `DMPlexDrawCell()`
- Add `DMPlexLabelCompleteStar()`
- Add `DMPlexVecGetClosureAtDepth()`
- Add an extra communicator argument to `DMPlexFilter()` to allow extracting local meshes
- Add `DMPlexCopyFlags()`
- Add `DMPlexRebalanceSharedLabelPoints()`
- Add `DMPlexCheckLabel()` and `DMPlexReconcileLabel()`
- Change CGNS viewer to use multi-component read/write interface for better performance
- Add `DMPlexTransformOrderSupports()`
- Add `DMPlexLabelCohesiveCheck()`
- Add `DMPlexCheckOrientationLabel()`
- Change `DMPlexLabelCohesiveComplete()` to remove split argument
- Add `DM_COORD_MAP_TORUS`
- Add `DM_COORD_MAP_ROTATE`
- Add `DM_SHAPE_DIIID`
- Add `DMPlexTriangleSetAngleBound()`, `DMPlexTriangleGetAngleBound()`, `DMPlexTetgenSetRadiusEdgeBound()`, `DMPlexTetgenGetRadiusEdgeBound()`, `DMPlexTetgenSetDihedralBound()`, `DMPlexTetgenGetDihedralBound()`

## FE/FV


## DMNetwork


## DMStag


## DT

- Add `PetscWeakFormGetKeys()`

## Fortran

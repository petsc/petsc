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

## Configure/Build

- Add `providesDocs` and `docsDirs` package attributes so that an external package's sources can be cloned and scanned to generate PETSc manual pages when the documentation is built

## Sys

- Add `PetscGetConfiguration()`
- Add `PetscObjectViewSynchronizedFromOptions()`
- Add `PetscSetMPIThreadRequiredType()`
- Add `PetscOverrideIntelMKLCPUVendorDetection()`
- Deprecate `PETSC_MPI_THREAD_REQUIRED`
- Add support for running PETSc applications as MCP servers that can be accessed by LLMs
- Add `PetscRandomAppendOptionsPrefix()` and `PetscRandomGetOptionsPrefix()`
- Add `PetscCallLAPACKInfo()` for calling LAPACK routines with an `info` argument where the caller requires an `info` value of 0 for the program to continue

## Event Logging


## PetscViewer

- Add support for writing CGNS descriptors on the base node: `PetscViewerCGNSGetDescriptors()`, `PetscViewerCGNSRestoreDescriptors()`, `PetscViewerCGNSSetDescriptor()`
- Add `PetscViewerVTKWriteFn` as the typedef prototype for the `write()` function passed to `PetscViewerVTKAddField()`. This addition requires no changes to user source code

## PetscDraw


## AO


## IS


## VecScatter / PetscSF


## PF


## Vec

- Add `VecSetStdBasis()` API to set a vector to the i-th standard basis vector
- Change the behavior of `VecPointwiseDivide()` implementing w = x / y: if a particular `y[i]` is zero and `x[i]` is also zero, `w[i]` is set to one (before it was set to zero).

## PetscSection


## PetscPartitioner


## Mat

- Add `MATPRODUCT_PtAP` support for `MATDIAGONAL` and `MATCONSTANTDIAGONAL`
- Add `MATPRODUCT_AB` support for `MATDIAGONAL` and `MATCONSTANTDIAGONAL` with any matrix type
- Add `MatSeqAIJGetKokkosView()`, `MatSeqAIJRestoreKokkosView()`, `MatSeqAIJGetKokkosViewWrite()` and `MatSeqAIJRestoreKokkosViewWrite()` to the public API
- Change `MatSeqAIJCUSPARSEGetIJ()`, `MatSeqAIJCUSPARSERestoreIJ()`, `MatSeqAIJHIPSPARSEGetIJ()` and `MatSeqAIJHIPSPARSERestoreIJ()` to return `PetscInt` indices instead of `int`
- Add `MatNormApproximate()` to compute matrix norms approximately
- Add `MatGetMultPetscSF()` to access the `PetscSF` used to communicate off-process vector entries in `MatMult()` for `MATMPIAIJ`, `MATMPIBAIJ`, `MATMPISBAIJ`, `MATMPIDENSE`, and `MATMPISELL`

## MatCoarsen


## PC

- Add `PCGAMGSetProlongatorFilter()` and `PCGAMGGetProlongatorFilter()` to set/get the threshold for filtering small entries from the prolongator in `PCGAMG`
- Add `PCAIR` and `PCPFLAREINV` manual pages, generated from the PFLARE sources when the documentation is built
- Add `PCParametersInitialize`
- Fix `PCMG` to honor `PCSetUseAmat(pc, PETSC_FALSE)` at all levels

## KSP

- Fix for `KSP` pre- and post-solve callbacks, that can now be used together with Eisenstat and Walker trick for `SNES`
- Add `KSPPreSolve()` and `KSPPostSolve()` to run the registered `KSP` pre/post solve callbacks
- Add `KSPIDR` — IDR(s) Induced Dimension Reduction Krylov solver (biorthogonal variant)
- Add `KSPIDRSetS()`, `KSPIDRGetS()`, `KSPIDRSetRandom()`, `KSPIDRGetRandom()`, `KSPIDRSetCosine()`, and `KSPIDRGetCosine()`

## SNES

- Change `SNESSetUp()` to not overwrite the NPC application context if one has previously been set on the NPC
- Change `SNESComputeJacobian()` to call the user-provided Jacobian function when a left NPC is active and the solver is not `SNESASPIN`
- Add support for nonlinear preconditioners with a `DM` different from the parent `SNES` `DM`. Calling `SNESSetNPC()` will no longer enforce default parameters on the npc.

## SNESLineSearch

- Deprecate `SNESLINESEARCHBASIC` in favor of `SNESLINESEARCHNONE`

## TS

- Add `DMTSSetIFunctionPre()`
- Add `TSDiscGradSetImplicitFormulation()`
- Expose `TSDiscGradGetX0AndXdot()` and `TSDiscGradRestoreX0AndXdot()`
- Add `TSIsImplicit()` that indicates if the `TSType` is implicit and uses `SNES` or `KSP`

## TAO

- Add `TaoGetDM()` and `TaoSetDM()`

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

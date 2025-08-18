# Changes: Development

% STYLE GUIDELINES:
% * Capitalize sentences
% * Use imperative, e.g., Add, Improve, Change, etc.
% * Don't use a period (.) at the end of entries
% * If multiple sentences are needed, use a period or semicolon to divide sentences, but not at the end of the final sentence

```{rubric} General:
```

- Add `PETSCPYTHONPATH` to the generated `$PETSC_DIR/$PETSC_ARCH/lib/petsc/conf/petscvariables` file so it is available to users through the makefile system
- Add `PETSCPYTHONPATH` to the generated `$PETSC_DIR/$PETSC_ARCH/lib/pkgconfig/PETSc.pc` file so it is available to users with
  `PKG_CONFIG_PATH=$PETSC_DIR/$PETSC_ARCH/lib/pkgconfig pkg-config --variable=PETSCPYTHONPATH PETSc.pc`
- Add `EXTRA_OPTIONS_INITIAL` to the test system, which prepends options

```{rubric} Configure/Build:
```

- Change `make sphinxhtml` in the `doc` directory to be `make docs`
- Change `make docs` to put all its artifacts in `${PETSC_ARCH}-doc` instead of `doc`
- Add support for `--download-xxx`, `--with-xxx`, and `-with-xxx-dir` for Python packages that install with pip. See `packages.py`
- Change `make alletags` to `make etags`
- Deactivate Fortran bindings of SuperLU_DIST and ExodusII by default, but they can still be built with the configure options `--with-superlu_dist-fortran-bindings` and `--with-exodusii-fortran-bindings`, respectively

```{rubric} Sys:
```

- Add `PETSC_E`

- Deprecate `PetscSSEIsEnabled()`
- Add `PetscBTCopy()`

```{rubric} Event Logging:
```

```{rubric} PetscViewer:
```

- Add `PetscViewerHDF5SetCompress()` and `PetscViewerHDF5GetCompress()`

```{rubric} PetscDraw:
```

- Add `PetscDrawLGGetData()`

```{rubric} AO:
```

```{rubric} IS:
```

```{rubric} VecScatter / PetscSF:
```

```{rubric} PF:
```

```{rubric} Vec:
```

```{rubric} PetscSection:
```

```{rubric} PetscPartitioner:
```

```{rubric} Mat:
```

- Add `MatConstantDiagonalGetConstant()`
- Add `MatNullSpaceRemoveFn` type definition
- Add `MatMFFDFn`, `MatMFFDiFn`, `MatMFFDiBaseFn`, and `MatMFFDCheckhFn` type definitions
- Add `MatFDColoringFn` type definition
- Add support for `-mat_mumps_icntl_15 1` with the companion function `MatMumpsSetBlk()`

```{rubric} MatCoarsen:
```

```{rubric} PC:
```

- Add `PCMatApplyTranspose()` and `PCShellSetMatApplyTranspose()`
- Remove `PC_ApplyMultiple`
- Add `PCShellPSolveFn`
- Add `PCModifySubMatricesFn`

```{rubric} KSP:
```

- Add `MatLMVMGetLastUpdate()`
- Add `MatLMVMMultAlgorithm`, `MatLMVMSetMultAlgorithm()`, and `MatLMVMGetMultAlgorithm()`
- Add `MatLMVMSymBroydenGetPhi()` and `MatLMVMSymBroydenSetPhi()`
- Add `MatLMVMSymBadBroydenGetPsi()` and `MatLMVMSymBadBroydenSetPsi()`
- Deprecate `KSP_CONVERGED_RTOL_NORMAL` in favor of `KSP_CONVERGED_RTOL_NORMAL_EQUATIONS` and `KSP_CONVERGED_ATOL_NORMAL` in favor of `KSP_CONVERGED_ATOL_NORMAL_EQUATIONS`
- Add `KSPFlexibleSetModifyPC()` to provide a common API for setting the modification function for all flexible `KSP` methods
- Add `KSPFlexibleModifyPCFn` function prototype
- Change the function signature of the `destroy()` argument to `KSPSetConvergenceTest()` to `PetscCtxDestroyFn*`. If you provide custom destroy
  functions to `KSPSetConvergenceTest()` you must change them to expect a `void **` argument and immediately dereference the input
- Add `KSPPSolveFn`
- Change `KSPMonitorResidualDraw()` to `KSPMonitorResidualView()`
- Change `KSPMonitorTrueResidualDraw()` to `KSPMonitorTrueResidualView()`

```{rubric} SNES:
```

- Change `SNESTestJacobian()` to report the norms
- Add `SNESNormSchedule` support to `SNESKSPONLY`

```{rubric} SNESLineSearch:
```

```{rubric} TS:
```

- Add `TSSetRunSteps()` and `-ts_run_steps` for better control of restarted jobs
- Add `-ts_monitor_solution_skip_initial` to skip first call to the solution monitor
- Add `-ts_monitor_wall_clock_time` to display the elapsed wall-clock time for every step
- Change `TSDiscGradIsGonzalez()`, `TSDiscGradUseGonzalez()` to `TSDiscGradSetType()`,`TSDiscGradGetType()`

```{rubric} TAO:
```

- Add `TaoBRGNSetRegularizationType()`, `TaoBRGNGetRegularizationType()`

```{rubric} PetscRegressor:
```

- Add new component to support regression and classification machine learning tasks: [](ch_regressor)
- Add `PetscRegressor` type `PETSCREGRESSORLINEAR` for solving linear regression problems with optional regularization

```{rubric} DM/DA:
```

- Add `DMHasBound()`, `DM_BC_LOWER_BOUND` and `DM_BC_LOWER_BOUND`
- Add `DMSetCellCoordinateField()`
- Add ``localized`` argument to `DMSetCoordinateDisc()` and `DMCreateAffineCoordinates_Internal()`
- Add `DMCreateGradientMatrix()`

```{rubric} DMSwarm:
```

- Add `DMSwarmProjectFields()` and `DMSwarmProjectGradientFields()`

```{rubric} DMPlex:
```

- Add `DMPlexGetTransform()`, `DMPlexSetTransform()`, `DMPlexGetSaveTransform()`, and `DMPlexSetSaveTransform()`
- Add `DMPlexGetCoordinateMap()` and `DMPlexSetCoordinateMap()`
- Add `DMPlexTransformCohesiveExtrudeGetUnsplit()`
- Add `DMFieldCreateDefaultFaceQuadrature()`
- Rename `DMPlexComputeResidual_Internal()` to `DMPlexComputeResidualForKey()`
- Rename `DMPlexComputeJacobian_Internal()` to `DMPlexComputeJacobianByKey()`
- Rename `DMPlexComputeJacobian_Action_Internal()` to `DMPlexComputeJacobianActionByKey()`
- Rename `DMPlexComputeResidual_Hybrid_Internal()` to `DMPlexComputeResidualHybridByKey()`
- Rename `DMPlexComputeJacobian_Hybrid_Internal()` to `DMPlexComputeJacobianHybridByKey()`
- Add `DMPlexInsertBounds()`
- Change argument order for `DMPlexComputeBdResidualSingle()` and `DMPlexComputeBdJacobianSingle()` to match domain functions
- Add `DMPlexComputeBdResidualSingleByKey()` and `DMPlexComputeBdJacobianSingleByLabel()`
- Add ``localized`` argument to `DMPlexCreateCoordinateSpace()`
- Remove ``coordFunc`` argument from `DMPlexCreateCoordinateSpace()`
- Change `DMPlexExtrude()` to take a label argument
- Rename `DMPlexVecGetOrientedClosure_Internal()` to `DMPlexVecGetOrientedClosure()`

```{rubric} FE/FV:
```

- Add `PetscFEExpandFaceQuadrature()`
- Add `PetscFECreateBrokenElement()`
- Change `PetscFEIntegrateJacobian()` signature to allow rectangular operators

```{rubric} DMNetwork:
```

```{rubric} DMStag:
```

```{rubric} DT:
```

- Deprecate `PetscSimplePointFunc` in favor of `PetscSimplePointFn *`
- Deprecate `PetscPointFunc` in favor of `PetscPointFn *`
- Deprecate `PetscPointJac` in favor of `PetscPointJacFn *`
- Deprecate `PetscBdPointFunc` in favor of `PetscBdPointFn *`
- Deprecate `PetscBdPointJac` in favor of `PetscBdPointJacFn *`
- Deprecate `PetscRiemannFunc` in favor of `PetscRiemannFn *`
- Deprecate `PetscProbFunc` in favor of `PetscProbFn *`
- Add `PetscDTCreateQuadratureByCell()`

```{rubric} Fortran:
```

- Add `PetscObjectNullify()`

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


```{rubric} Configure/Build:
```

```{rubric} Sys:
```

- Deprecate `PetscSSEIsEnabled()`

```{rubric} Event Logging:
```

```{rubric} PetscViewer:
```

- Add `PetscViewerHDF5SetCompress()` and `PetscViewerHDF5GetCompress()`

```{rubric} PetscDraw:
```

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

```{rubric} MatCoarsen:
```

```{rubric} PC:
```

- Add `PCMatApplyTranspose()`
- Remove `PC_ApplyMultiple`

```{rubric} KSP:
```

- Add `MatLMVMGetLastUpdate()`
- Add `MatLMVMMultAlgorithm`, `MatLMVMSetMultAlgorithm()`, and `MatLMVMGetMultAlgorithm()`
- Add `MatLMVMSymBroydenGetPhi()` and `MatLMVMSymBroydenSetPhi()`
- Add `MatLMVMSymBadBroydenGetPsi()` and `MatLMVMSymBadBroydenSetPsi()`
- Deprecate `KSP_CONVERGED_RTOL_NORMAL` in favor of `KSP_CONVERGED_RTOL_NORMAL_EQUATIONS` and `KSP_CONVERGED_ATOL_NORMAL` in favor of `KSP_CONVERGED_ATOL_NORMAL_EQUATIONS`

```{rubric} SNES:
```

```{rubric} SNESLineSearch:
```

```{rubric} TS:
```

```{rubric} TAO:
```

- Add ``TaoBRGNSetRegularizationType()``, ``TaoBRGNGetRegularizationType()``

```{rubric} PetscRegressor:
```

- Add new component to support regression and classification machine learning tasks: [](ch_regressor)
- Add `PetscRegressor` type `PETSCREGRESSORLINEAR` for solving linear regression problems with optional regularization

```{rubric} DM/DA:
```

- Add `DMHasBound()`, `DM_BC_LOWER_BOUND` and `DM_BC_LOWER_BOUND`


```{rubric} DMSwarm:
```

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

```{rubric} FE/FV:
```

- Add `PetscFEExpandFaceQuadrature()`

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

```{rubric} Fortran:
```

# Changes: Development

% STYLE GUIDELINES:
% * Capitalize sentences
% * Use imperative, e.g., Add, Improve, Change, etc.
% * Don't use a period (.) at the end of entries
% * If multiple sentences are needed, use a period or semicolon to divide sentences, but not at the end of the final sentence

```{rubric} General:
```

- Change behavior of `-options_left` when set to `true`: it no longer triggers a call to `PetscOptionsView()`

```{rubric} Configure/Build:
```

- Make `SYCL` a `devicePackage`, i.e., builds `--with-sycl` now have `PETSC_HAVE_DEVICE` defined
- Add the option `--with-devicelanguage` to compile `PetscDevice` code using either a C or C++ compiler
- Add `Caliper`, an instrumentation and performance profiling library that can be used to profile `Hypre`.

```{rubric} Sys:
```

- Add `PetscCallHYPRE()` to check HYPRE error codes and print error messages on failure


```{rubric} Event Logging:
```

- Add two approaches for GPU energy monitoring:  `-log_view_gpu_energy` and `-log_view_gpu_energy_meter`
- Add API `PetscLogGpuEnergy()`, `PetscLogGpuEnergyMeter()`, `PetscLogGpuEnergyMeterBegin()` and `PetscLogGpuEnergyMeterEnd()` for GPU energy monitoring

```{rubric} PetscViewer:
```

-  Change the final argument of `PetscViewerGLVisSetFields()` to `PetscCtxDestroyFn *`. This means the destroy function must dereference the argument before operating on it

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

- Change the `destroy()` function argument of `MatShellSetMatProductOperation()` to type `PetscCtxDestroyFn *`. This means the destroy function must dereference the argument before operating on it
- Remove `MatMissingDiagonal()`. Developers should use `MatGetDiagonalMarkers_SeqXXX()` when the functionality is needed
- Change `MatSetOption(A, MAT_HERMITIAN, PETSC_TRUE)` for `MatSBAIJ` to no longer automatically set the option `MAT_SYMMETRIC` to `PETSC_FALSE`. It is now the duty of the user to call `MatSetOption(A, MAT_SYMMETRIC, PETSC_FALSE)` if a `MatSBAIJ` is Hermitian but not symmetric

```{rubric} MatCoarsen:
```

```{rubric} PC:
```

- Add multi-precision support for MUMPS. One could use `-pc_precision <single, double>` to set the precision to be used by MUMPS, which can be different from `PetscScalar`'s precision
- Add support for MUMPS out-of-core facility with the option `-mat_mumps_ooc_tmpdir <dir>` and new functions `MatMumpsSetOocTmpDir()`, `MatMumpsGetOocTmpDir()`

```{rubric} KSP:
```

- Remove `KSPHPDDMPrecision` in favor of `PetscPrecision`

```{rubric} SNES:
```

- Change the `destroy()` function argument of `SNESSetConvergenceTest()` to type `PetscCtxDestroyFn *`. This means the destroy function must dereference the argument before operating on it
- Add `SNESSetObjectiveDomainError()`
- Change `SNES_DIVERGED_FNORM_NAN` to `SNES_DIVERGED_FUNCTION_NANORINF`
- Add `SNES_DIVERGED_OBJECTIVE_NANORINF`
- Add `SNES_DIVERGED_OBJECTIVE_DOMAIN`
- Add developer functions `SNESCheckFunctionDomainError()`, `SNESLineSearchCheckFunctionDomainError()`, `SNESCheckObjectiveDomainError()`, `SNESLineSearchCheckObjectiveDomainError()`, `SNESCheckJacobianDomainError()`, and `SNESLineSearchCheckJacobianDomainError()`

```{rubric} SNESLineSearch:
```

```{rubric} TS:
```

- Add `TSPseudoComputeFunction()` to get nonlinear residual while avoiding recalculation if possible
- Remove unused `TSPseudoVerifyTimeStepDefault()`
- Remove `TSPseudoComputeTimeStep()` and `TSPseudoVerifyTimeStep()`
- Change the `destroy()` function argument of `TSTrajectorySetTransform()` to type `PetscCtxDestroyFn *`. This means the destroy function must dereference the argument before operating on it
- Correct option `-ts_max_reject` to `-ts_max_step_rejections`
- Correct option `-ts_dt` to `-ts_time_step`

```{rubric} TAO:
```

```{rubric} PetscRegressor:
```

```{rubric} DM/DA:
```

-  Change the final argument of `DMShellSetDestroyContext()` to `PetscCtxDestroyFn *`. This means the destroy function must dereference the argument before operating on it

```{rubric} DMSwarm:
```

```{rubric} DMPlex:
```

- Add `DMPlexVecGetClosureAtDepth()`
- Add an extra communicator argument to `DMPlexFilter()` to allow extracting local meshes
- Add `DMPlexGetLETKFLocalizationMatrix` to compute localization weight matrix for LETKF

```{rubric} FE/FV:
```

```{rubric} DMNetwork:
```

```{rubric} DMStag:
```

```{rubric} DT:
```

```{rubric} Fortran:
```

- Replace `./configure` option `--with-mpi-f90module-visibility` with `--with-mpi-ftn-module=<mpi or mpi_f08>`
- Add `PETSC_INT_KIND` and `PETSC_MPIINT_KIND`
- Fortran code should now use `MPIU_Comm` instead of `MPI_Comm`, and similarly for other MPI types, see section "Fortran and MPI" in the users guide


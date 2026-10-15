# Changes: Development

% STYLE GUIDELINES:
% * Capitalize sentences
% * Use imperative, e.g., Add, Improve, Change, etc.
% * Don't use a period (.) at the end of entries
% * If multiple sentences are needed, use a period or semicolon to divide sentences, but not at the end of the final sentence

```{rubric} General:
```

```{rubric} Configure/Build:
```

- Add `--download-cutest` and `--download-sifdecode` for the CUTEst optimization testing environment and SIF problem decoder

- Add Meson package builds, `--download-meson`, and `--download-package-meson-arguments` for additional Meson setup arguments

- Add `--download-ninja` and `--with-ninja-exec` for the Ninja backend used by Meson package builds

- Add `--with-meson-exec` to select an existing Meson executable

```{rubric} Sys:
```

- Add `PetscCommCreateNonempty()` and `PetscCommDestroyNonempty()` for external libraries, such as ParMETIS and PT-SCOTCH, that require every process of their communicator to own data

```{rubric} Event Logging:
```

- Change the `CpuToGpu Count` and `GpuToCpu Count` columns of `-log_view` to report the maximum over the processes instead of the average, which rounded to 0 whenever the total number of copies was less than half the number of processes; the size columns remain averages

```{rubric} PetscViewer:
```

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

- Add a device implementation of `VecSqrtAbs()` for `VECKOKKOS`; previously it copied the vector to the host

```{rubric} PetscSection:
```

```{rubric} PetscPartitioner:
```

```{rubric} Mat:
```

- Add device implementations of `MatNorm()` with `NORM_1`, `NORM_FROBENIUS`, and `NORM_INFINITY` for `MATAIJKOKKOS`; previously all norms copied the matrix values to the host
- Add `-mat_spd` to set `MAT_SPD` from the options database in `MatSetFromOptions()`
- Change `MatSetOption()` to error when a symmetry-related option contradicts an already known property
- Add support for `MatZeroRows()` and `MatZeroRowsColumns()` with a nonzero diagonal value on a `MATIS` whose local matrix does not implement `MatSetValues()`, such as `MATSHELL`
- Change `MatSetValuesCOO()` for `MATAIJ` and its device subtypes to generate an error when the nonzero pattern has changed since `MatSetPreallocationCOO()`, for example because `MatSetValues()` inserted a new nonzero; values were previously written into wrong locations in such a sequence of events
- Change `MatAXPY()` with `DIFFERENT_NONZERO_PATTERN` for `MATSEQAIJ` and `MATMPIAIJ` to add in place when the nonzeros of `X` are a subset of those of `Y`, keeping the nonzero pattern and nonzero state of `Y`, as well as the options that survive assembly, such as `MAT_SYMMETRY_ETERNAL`
- Add `MatGetRowSumAbs()` for `MATHYPRE`, on the host and the device, and device implementations of `MatGetRowSumAbs()` for `MATAIJCUSPARSE`, `MATAIJHIPSPARSE`, and `MATAIJKOKKOS`; previously the AIJ device types copied the matrix values to the host
- Change `MatSetRandom()` to no longer assemble at the interface; `MATOP_SETRANDOM` implementations must leave the matrix assembled with increased state

```{rubric} MatCoarsen:
```

```{rubric} PC:
```

- Add `PCHPDDMSetScaling()` to supply a diagonal scaling `Vec` for the deflation space set with `PCHPDDMSetDeflationMat()`. If the fine level is a `PCASM`, its default `PCASMType` is set to `PC_ASM_WEIGHTED`, in which case the scaling is forwarded accordingly

```{rubric} KSP:
```

- Add `KSPCGTRNormType`, `KSPCGSetTRNormType()`, and `KSPCGGetTRNormType()` to select the trust-region norm in `KSPCG`, `KSPSTCG`, `KSPNASH`, and `KSPGLTR`
- Deprecate `KSPCGGetNormD()` in favor of `KSPCGGetTRSolutionNorm()` and `-ksp_cg_dtype` in favor of `-ksp_cg_tr_norm_type`
- Implement `KSPSTCG` as a specialization of `KSPCG`, sharing trust-region norms, feasible Euclidean initial guesses, and objective targets
- Add `-ksp_cg_radius radius` and `KSPCGGetTRSolutionNorm()` support to `KSPCG`

```{rubric} SNES:
```

```{rubric} SNESLineSearch:
```

```{rubric} TS:
```

```{rubric} TAO:
```

- Add support for CUTEst unconstrained problems in TAO tutorials and tests
- Map `TAOSNES` convergence and divergence reasons with the public `TaoConvergedReasonFromSNES()`, and add `TaoSNESGetSNES()` to access the underlying solver

```{rubric} TaoTerm:
```

```{rubric} PetscRegressor:
```

```{rubric} PetscDA:
```

```{rubric} DM:
```

```{rubric} DMSwarm:
```

```{rubric} DMPlex:
```

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

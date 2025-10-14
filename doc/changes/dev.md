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

- Make `SYCL` a `devicePackage`, i.e., builds `--with-sycl` now have `PETSC_HAVE_DEVICE` defined

```{rubric} Sys:
```

```{rubric} Event Logging:
```

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

```{rubric} PetscSection:
```

```{rubric} PetscPartitioner:
```

```{rubric} Mat:
```

```{rubric} MatCoarsen:
```

```{rubric} PC:
```

- Add multi-precision support for MUMPS. One could use `-pc_precision <single, double>` to set the precision to be used by MUMPS, which can be different from `PetscScalar`'s precision

```{rubric} KSP:
```

- Remove `KSPHPDDMPrecision` in favor of `PetscPrecision`

```{rubric} SNES:
```

```{rubric} SNESLineSearch:
```

```{rubric} TS:
```

```{rubric} TAO:
```

```{rubric} PetscRegressor:
```

```{rubric} DM/DA:
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

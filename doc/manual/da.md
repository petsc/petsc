(ch_da)=

# PetscDA: Data Assimilation

PETSc's `PetscDA` object coordinates data assimilation (DA) workflows.

This is new code, please independently verify all results you obtain using it.

Some planned work for `PetscDA` is available as GitLab Issue #1882

## Ensemble-based Data Assimilation

Currently `PetscDA` supports one ensemble-based assimilator: `PETSCDALETKF`. It implements the Local Ensemble
Transform Kalman Filter, with a `none`-localization mode that reduces to the classic global ETKF.
The framework is extensible to other assimilation techniques.

- {any}`sec_da_letkf`

The `PetscDA` object owns the ensemble `Mat`, the observation operator, and the registered forecast/analysis callbacks, so assimilation algorithms work against a single API regardless of how the ensemble is partitioned across MPI ranks or which `Mat`/`Vec` backend stores it.

(sec_da_ensemble_lifecycle)=

## Lifecycle overview

A typical assimilation cycle alternates between forecast propagation and statistical analysis:

1. Initialize a `PetscDA` context and configure ensemble sizes and data structures.
2. Populate the ensemble state vectors and optional observation-error descriptions.
3. Advance each ensemble member with a model operator supplied by the application.
4. Combine forecasts with observations through `PetscDAEnsembleAnalysis()` to produce the posterior ensemble.
5. Repeat until the desired simulation horizon is complete, optionally extracting diagnostics after each phase.

Throughout this loop the `PetscDA` object abstracts the global vectors, scatters, and reductions needed to compute ensemble means, anomalies, and square-root transforms.

Here we introduce a simple example to demonstrate `PetscDA` usage with the `PETSCDALETKF` on the Lorenz-96 model.
Please read {any}`sec_da_impls` for more in-depth discussion of the available implementations.

(da-ex1)=
:::{admonition} Listing: `src/ml/da/tutorials/ex1.c`
```{literalinclude} /../src/ml/da/tutorials/ex1.c
:start-at: int main
:end-at: PetscFinalize
:append: return 0;}
```
:::

(sec_da_creating)=

## Creating a PetscDA context

Create, configure, and destroy a `PetscDA` object with the standard PETSc object lifecycle:

```c
PetscDA da;
PetscCall(PetscDACreate(PETSC_COMM_WORLD, &da));
PetscCall(PetscDASetType(da, PETSCDALETKF));
PetscCall(PetscDASetSizes(da, state_size, obs_size));
PetscCall(PetscDAEnsembleSetSize(da, ensemble_size));
PetscCall(PetscDASetFromOptions(da));
PetscCall(PetscDASetUp(da));

/* ... data assimilation loop ... */

PetscCall(PetscDADestroy(&da));
```

To create a `PetscDA` instance, call `PetscDACreate()`:

```c
PetscDACreate(MPI_Comm comm, PetscDA *da);
```

To choose an implementation type, call

```c
PetscDASetType(PetscDA da, PetscDAType type);
```

or use the command-line option `-petscda_type name`; details regarding the
available implementations are presented in {any}`sec_da_impls`.

`PetscDASetSizes()` records the global state dimension and the number of observations:

```c
PetscDASetSizes(PetscDA da, PetscInt state_size, PetscInt obs_size)
```

For MPI-parallel runs where the state or observation vectors are distributed,
the local partition sizes can be set explicitly with

```c
PetscDASetLocalSizes(PetscDA da, PetscInt local_state_size, PetscInt local_obs_size);
```

Pass `PETSC_DECIDE` for either argument to let PETSc choose the partition automatically.

`PetscDAEnsembleSetSize()` records the number of ensemble members requested:

```c
PetscDAEnsembleSetSize(PetscDA da, PetscInt ensemble_size)
```

After having set these options, call `PetscDASetFromOptions()` to apply any
command-line overrides, then `PetscDASetUp()` to allocate internal storage:

```c
PetscDASetFromOptions(PetscDA da);
PetscDASetUp(PetscDA da);
```

Finally, after the user is done using the DA object, destroy it with

```c
PetscDADestroy(PetscDA *da);
```

(sec_da_ndof)=

## Degrees of freedom per grid point

When the state vector is laid out on a structured grid, the number of physical
degrees of freedom per grid point can be recorded with

```c
PetscDASetNDOF(PetscDA da, PetscInt ndof);
PetscDAGetNDOF(PetscDA da, PetscInt *ndof);
```

This metadata is used by some implementations and viewers; the default is `1`.

(sec_da_ensemble)=

## Managing ensembles

`PetscDA` stores ensemble members as PETSc `Vec` objects and exposes convenience
helpers to access them safely.

Individual ensemble members can be read or overwritten using a get/restore
ownership pattern analogous to `MatDenseGetColumnVec`:

```c
/* Read-only view of member i; must be paired with Restore */
PetscDAEnsembleGetMember(PetscDA da, PetscInt i, Vec *member);
PetscDAEnsembleRestoreMember(PetscDA da, PetscInt i, Vec *member);

/* Inject an externally created vector into slot i */
PetscDAEnsembleSetMember(PetscDA da, PetscInt i, Vec member);
```

`PetscDAEnsembleGetMember()` / `PetscDAEnsembleRestoreMember()` map ensemble
indices to `Vec` handles that participate in PETSc's reference counting.
`PetscDAEnsembleSetMember()` lets applications inject externally created vectors
into specific slots, which is useful when importing state snapshots from disk or
another solver component.

Ensemble statistics are computed with:

```c
/* Form the sample mean across all members */
PetscDAEnsembleComputeMean(PetscDA da, Vec mean);

/* Return a tall-and-skinny Mat whose columns are the mean-subtracted,
   normalized anomalies X = (E - mean) / sqrt(m-1) */
PetscDAEnsembleComputeAnomalies(PetscDA da, Vec mean, Mat *anomalies);
```

`PetscDAEnsembleComputeAnomalies()` returns a dense `Mat` whose columns are the
mean-subtracted ensemble anomalies. Many square-root filters use this matrix to
construct low-rank covariance factorizations. Pass `NULL` for `mean` to have the
function compute it internally.

(sec_da_obs_error)=

## Observation error

Observation-error variances (or more general descriptions) are supplied through
`PetscDASetObsErrorVariance()`. The associated vector is assumed to follow the
global observation ordering; `PetscDAGetObsErrorVariance()` returns the stored
object for later inspection or reuse.

```c
PetscDASetObsErrorVariance(PetscDA da, Vec obs_error_var);
PetscDAGetObsErrorVariance(PetscDA da, Vec *obs_error_var);
```

(sec_da_analysis)=

## Analysis step

`PetscDAEnsembleAnalysis()` performs the assimilation step given an observation vector and a linear observation operator matrix:

```c
/* observation - Vec of length P (number of observations) */
/* H           - Mat of size P x N (observation operator mapping state to observation space) */
PetscCall(PetscDAEnsembleAnalysis(da, observation, H));
```

`H` is a `Mat` (typically sparse AIJ) that maps the N-dimensional state vector to the P-dimensional observation space: `y ≈ H*x`. `PetscDAAnalysis()` handles all ensemble reductions,
gain computations, and posterior updates.

(sec_da_model)=

## Forecast step

`PetscDAEnsembleForecast()` wraps the forecast step. The user supplies a function that advances the entire ensemble matrix in one call:

Calling sequence for model:

- `ensemble` - the ensemble matrix whose columns are the states to be advanced in place
- `ctx` - the context for the model function

A model that can advance the whole ensemble at once (e.g. a vectorized RHS, a Kokkos-resident propagator, or a single time integrator over a stacked state) writes directly to the columns of `ensemble`. A model that can only advance one state at a time, such as a `TS`-driven step, loops over the columns itself. Use `MatDenseGetColumnVec()` / `MatDenseRestoreColumnVec()` (the read-write pair) for the common case where the model reads the current column as an initial condition and writes the advanced state back; reach for `MatDenseGetColumnVecRead()` / `MatDenseGetColumnVecWrite()` only when the kernel is truly read-only or write-only (e.g. a write-only resampler):

```c
/* Prototype for model forecast M(X) */
PetscErrorCode ModelForecast(Mat ensemble, PetscCtx ctx) {
  PetscInt n, j;

  PetscFunctionBeginUser;
  PetscCall(MatGetSize(ensemble, NULL, &n));
  for (j = 0; j < n; j++) {
    Vec col;

    PetscCall(MatDenseGetColumnVec(ensemble, j, &col));
    /* Advance col by dt (e.g., step a TS object) */
    PetscCall(MatDenseRestoreColumnVec(ensemble, j, &col));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscCall(PetscDAEnsembleForecast(da, ModelForecast, ctx));
```

No `MatAssemblyBegin()`/`MatAssemblyEnd()` call is needed after `MatDenseRestoreColumnVec()`: the dense `Get`/`Restore` pair is itself the assembled write path, and the matrix stays assembled across it.

`ModelForecast` can call PETSc time integrators ({any}`ch_ts`), nonlinear solvers ({any}`ch_snes`), or bespoke device kernels.

(sec_da_inflation)=

## Inflation

Covariance inflation counteracts ensemble collapse by artificially widening the
prior spread before the analysis step. The inflation factor $\rho \ge 1$ scales
each anomaly so that the effective prior covariance becomes $\rho^2 P^f$.
Set and retrieve the factor with

```c
PetscDAEnsembleSetInflation(PetscDA da, PetscReal inflation);
PetscDAEnsembleGetInflation(PetscDA da, PetscReal *inflation);
```

or at runtime with `-petscda_ensemble_inflation value` (default: `1.0`, i.e. no inflation).

(sec_da_impls)=

## Implementations

The only built-in `PetscDA` implementation is the Local Ensemble Transform Kalman
Filter (`PETSCDALETKF`, selected with `-petscda_type letkf`). Custom types can be
registered with `PetscDARegister()` and selected at runtime with `-petscda_type name`.

(sec_da_letkf)=

### LETKF

The Local ETKF (`PETSCDALETKF`, `-petscda_type letkf`) is the default implementation.
It performs the analysis update locally around each grid point, enabling scalable
assimilation on large domains by avoiding the global ensemble covariance matrix.
With `-petscda_letkf_localization_type none` it reduces to the classic global ETKF
(Algorithm 6.4 in {cite}`da2016`), a deterministic square-root update that avoids
stochastic perturbations.

The reduced-space T-matrix is factored via a symmetric eigendecomposition
$T = V D V^T$, so the square root used in the ensemble transform is the symmetric
$T^{-1/2} = V D^{-1/2} V^T$. The symmetric form minimizes the rotation of the prior
ensemble (preserving member continuity across analysis cycles) and is the only square
root that is consistent across overlapping local domains under localization.

LETKF-specific configuration:

```c
/* Distance-based localization: pick a kernel, set the radius, supply
   per-dimension state coordinates and the observation operator H.
   The localization matrix Q is built lazily on the first analysis. */
PetscDALETKFSetLocalizationType(PetscDA da, PetscDALETKFLocalizationType type);
PetscDALETKFSetLocalizationRadius(PetscDA da, PetscReal radius);
PetscDALETKFSetLocalizationCoordinates(PetscDA da, Vec xyz[], PetscReal bd[], Mat H);
```

The built-in `-petscda_letkf_localization_type` values `gaspari_cohn`, `gaussian`, and `boxcar` are available in
every PETSc build; the `none` type disables localization and is mathematically
equivalent to global ETKF. The localization matrix Q is built on the device
matching the observation-error covariance matrix `R` (set via `PetscDASetObsErrorVariance()`):
a Kokkos backend is used when `R` has type `MATAIJKOKKOS`, otherwise a CPU
analysis path is used. Select the kernel at runtime with
`-petscda_letkf_localization_type (none|gaspari_cohn|gaussian|boxcar)`.

```{note}
The CPU analysis path is currently single-rank only. The unlocalized fast path
(`PETSCDA_LETKF_LOC_NONE`) factors the m × m T matrix with LAPACK on each
rank's local slice, and the per-vertex CPU path lacks the cross-rank
observation scatter (which is built only by the Kokkos backend). For
multi-rank runs configure PETSc with `--download-kokkos-kernels` and use a
localized kernel; otherwise restrict the run to a single MPI rank.
```

(sec_da_options)=

## Command-line options

The `PetscDA` object obeys standard PETSc options parsing. Commonly used switches include:

- `-petscda_type name`                – select a registered `PetscDA` implementation (`letkf`).
- `-petscda_ensemble_inflation value` – set the covariance inflation factor (default: `1.0`).
- `-petscda_view`                     – inspect ensemble metadata and internal sizes.

Because `PetscDA` participates in the PETSc object registry, any prefix applied with `PetscDASetOptionsPrefix()` scopes these options.

(sec_da_viewing)=

## Viewing and monitoring

`PetscDAView()` and `PetscDAViewFromOptions()` expose ensemble sizing, observation dimensions, and implementation-specific diagnostics
. Views can be directed to ASCII, HDF5, or custom `PetscViewer` targets, enabling lightweight instrumentation of assimilation experiments.
For advanced profiling, the `PetscDA` package registers with PETSc's logging infrastructure via `PetscDAInitializePackage()`/`PetscDAFinalizePackage()`,
so standard `-log_view` outputs will include assimilation breakdown.

(sec_da_further_reading)=

## Further reading

- {any}`ch_ts` discusses PETSc time integrators that can supply the forecast operator passed to `PetscDAForecast()`.
- {any}`ch_vectors` documents vector assembly and parallel data management for the state and observation spaces.
- {any}`ch_snes` outlines nonlinear solvers that often participate in observation or model operators.
- {any}`ch_dmbase` provides background on distributed mesh infrastructure that can coexist with PetscDA-managed ensembles.

```{eval-rst}
.. bibliography:: /petsc.bib
    :filter: docname in docnames
```

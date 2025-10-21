(ch_da)=

# PetscDA: Data Assimilation

PETSc's `PetscDA` object coordinates data assimilation (DA) workflows.

This is new code, please independently verify all results you obtain using it.

Some planned work for `PetscDA` is available as GitLab Issue #1882

## Ensemble-based Data Assimilation

Currently `PetscDA` only supports ensemble-based data assimilation with two `PetscDAType`: `PETSCDAETKF` and `PETSCDALETKF`. These
focus on ensemble transform Kalman filter (ETKF)-style updates but are extensible to other assimilation techniques.

- {any}`sec_da_etkf`
- {any}`sec_da_letkf`

These centralize ensemble storage, observational metadata, and user-defined forecast/analysis operators so that algorithms can run independently of the MPI layout or the vector/matrix backends.

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
PetscCall(PetscDASetType(da, PETSCDAETKF));
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

or use the command-line option `-petscda_type <name>`; details regarding the
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

`PetscDAEnsembleForecast()` wraps the forecast step. The user supplies a function that advances a single ensemble member:

Calling sequence for model:

- `input` - the vector to be evolved, forecasted, time-stepped, or otherwise advanced
- `output` - the forecast, evolved, or time-stepped result
- `ctx` - the context for the model function

```c
/* Prototype for model forecast M(x) */
PetscErrorCode ModelForecast(Vec input, Vec output, PetscCtx ctx) {
  /* Advance input by dt to produce output */
  /* (e.g., step a TS object) */
  return PETSC_SUCCESS;
}

PetscCall(PetscDAEnsembleForecast(da, ModelForecast, ctx));
```

`ModelForecast` can call PETSc time integrators ({any}`ch_ts`), nonlinear solvers ({any}`ch_snes`), or bespoke device kernels.
The `PetscDA` layer orchestrates calls across the entire ensemble, issuing them in rank-local loops while ensuring that ownership and recycling semantics remain correct.

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

or at runtime with `-petscda_ensemble_inflation <value>` (default: `1.0`, i.e. no inflation).

(sec_da_impls)=

## Implementations

The available `PetscDA` implementations are listed in Table {any}`tab-dadefaults`.
Custom types can be registered with `PetscDARegister()` and selected at runtime
with `-petscda_type <name>`.

```{eval-rst}
.. list-table:: PETSc Data Assimilation Methods
   :name: tab-dadefaults
   :header-rows: 1

   * - Method
     - PetscDAType
     - Options Name
   * - Ensemble Transform Kalman Filter
     - ``PETSCDAETKF``
     - ``etkf``
   * - Local Ensemble Transform Kalman Filter
     - ``PETSCDALETKF``
     - ``letkf``
```

(sec_da_etkf)=

### ETKF

The built-in square-root ETKF (`PETSCDAETKF`, `-petscda_type etkf`) is the
default implementation. It implements Algorithm 6.4 in {cite}`da2016` using a
deterministic square-root update that avoids stochastic perturbations.

The ETKF supports two factorization strategies for the reduced-space T-matrix:

```c
PetscDAEnsembleSetSqrtType(PetscDA da, PetscDASqrtType type);
PetscDAEnsembleGetSqrtType(PetscDA da, PetscDASqrtType *type);
```

```{eval-rst}
.. list-table:: ETKF square-root types
   :name: tab-dasqrttypes
   :header-rows: 1

   * - ``PetscDASqrtType``
     - Options string
     - Notes
   * - ``PETSCDA_SQRT_CHOLESKY``
     - ``cholesky``
     - O(n³/3); preferred when the reduced-space matrix is positive definite
   * - ``PETSCDA_SQRT_EIGEN``
     - ``eigen``
     - More robust for semi-definite matrices; handles small negative eigenvalues from round-off
```

Select at runtime with `-petscda_ensemble_sqrt_type {cholesky,eigen}` (default: `eigen`).

(sec_da_letkf)=

### LETKF

The Local ETKF (`PETSCDALETKF`, `-petscda_type letkf`) performs the analysis
update locally around each grid point, enabling scalable assimilation on large
domains by avoiding the global ensemble covariance matrix. LETKF-specific
configuration:

```c
/* Number of observations associated with each grid vertex (default: 9) */
PetscDALETKFSetObsPerVertex(PetscDA da, PetscInt n_obs_vertex);
PetscDALETKFGetObsPerVertex(PetscDA da, PetscInt *n_obs_vertex);

/* Localization weight matrix Q (N x P) and observation operator matrix H (P x N) */
PetscDALETKFSetLocalization(PetscDA da, Mat Q, Mat H);
```

Set the observation count at runtime with
`-petscda_letkf_obs_per_vertex <n>` (default: `9`).

(sec_da_options)=

## Command-line options

The `PetscDA` object obeys standard PETSc options parsing. Commonly used switches include:

- `-petscda_type <name>`                         – select a registered `PetscDA` implementation (`etkf`, `letkf`).
- `-petscda_ensemble_inflation <value>`          – set the covariance inflation factor (default: `1.0`).
- `-petscda_ensemble_sqrt_type {cholesky,eigen}` – select the T-matrix square-root algorithm for ETKF (default: `eigen`).
- `-petscda_letkf_obs_per_vertex <n>`            – set the number of observations per grid vertex for LETKF (default: `9`).
- `-petscda_view`                                – inspect ensemble metadata and internal sizes.

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

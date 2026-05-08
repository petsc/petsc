#pragma once

#include <petsc/private/daimpl.h>
#include <petsc/private/daensembleimpl.h>

typedef struct {
  PetscDA_Ensemble en; /* MUST stay first: shared ensemble code casts da->data as (PetscDA_Ensemble *) */
  Vec              mean;
  Vec              y_mean;
  Vec              delta_scaled;
  Vec              w;
  Vec              r_inv_sqrt;
  Mat              Z;
  Mat              S;
  Mat              T_sqrt;
  Mat              w_ones;
  /* Localization matrix (n_grid x n_observations_total), variable nnz per row;
     built lazily on first analysis and refreshed when Q_dirty is set. */
  Mat                          Q;
  PetscDALETKFLocalizationType type;                /* Localization kernel type */
  PetscReal                    localization_radius; /* Cutoff half-width for built-in kernels */

  /* Cached inputs for lazy Q construction (built-in kernels only) */
  Vec       coord_xyz[3]; /* Coordinate vectors for grid points (per dimension) */
  PetscReal coord_bd[3];  /* Periodic-domain extents (0 = non-periodic) */
  Mat       coord_H;      /* Observation operator used to map coordinates to observation locations */
  PetscBool Q_dirty;      /* Q must be (re)built before next analysis */

  PetscInt max_nnz_per_row; /* Cached max nnz across all rows of Q (global) */
  PetscInt min_nnz_per_row; /* Cached min nnz across all rows of Q (global) */
  PetscInt n_grid;          /* Number of grid points (n_grid = state_size / da->ndof) */
  PetscInt batch_size;      /* Batch size for GPU processing */

  /* Localization support for MPI */
  IS         obs_is_local;    /* Indices of observations needed by this process */
  VecScatter obs_scat;        /* Scatter context for observations */
  Vec        obs_work;        /* Local work vector for observations */
  Vec        y_mean_work;     /* Local work vector for y_mean */
  Vec        r_inv_sqrt_work; /* Local work vector for r_inv_sqrt */
  Mat        Z_work;          /* Local work matrix for Z (SeqDense) */
  PetscHMapI obs_g2l;         /* Map global observation index to local index in obs_work */

  /* Device-side CSR view of Q (Kokkos Views cast to void*; backend reinterprets) */
  void *Q_device_i; /* Row pointers (length n_grid + 1) */
  void *Q_device_j; /* Column indices, LOCAL into obs_work (not global obs indices) */
  void *Q_device_a; /* Nonzero values */

  /* Persistent solver handles and workspace */
  void *solver_handle; /* cusolverDnHandle_t / rocblas_handle / sycl::queue* */
  void *eigen_work;    /* EigenWorkspace* */
} PetscDA_LETKF;

PETSC_INTERN const char *const PetscDALETKFLocalizationTypes[];

PETSC_INTERN PetscErrorCode PetscDALETKFCreateLocalizationMat(PetscDALETKFLocalizationType, PetscReal, Vec[], PetscReal[], Mat, Mat *);
PETSC_INTERN PetscErrorCode PetscDALETKFGatherObsBbox(PetscInt, Vec[], PetscReal[], PetscReal, Mat, Vec[], PetscInt *, PetscInt **, PetscReal **);
PETSC_INTERN PetscErrorCode PetscDALETKFSetupObsScatter(PetscDA_LETKF *, Mat);
PETSC_INTERN PetscErrorCode PetscDALETKFDestroyObsScatter(PetscDA_LETKF *);
PETSC_INTERN PetscErrorCode PetscDALETKFBroadcastWeightVector(Vec, PetscInt, Mat);
PETSC_INTERN PetscErrorCode PetscDALETKFLocalAnalysis(PetscDA, PetscDA_LETKF *, PetscInt, PetscInt, Mat, Vec, Mat, Vec, Vec);
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
PETSC_INTERN PetscErrorCode PetscDALETKFCreateLocalizationMat_Kokkos(PetscDALETKFLocalizationType, PetscReal, Vec[], PetscReal[], Mat, Mat *);
PETSC_INTERN PetscErrorCode PetscDALETKFLocalAnalysis_Kokkos(PetscDA, PetscDA_LETKF *, PetscInt, PetscInt, Mat, Vec, Mat, Vec, Vec);
PETSC_INTERN PetscErrorCode PetscDALETKFGlobalAnalysis_Kokkos(PetscDA, PetscDA_LETKF *, PetscInt, Mat, Vec);
PETSC_INTERN PetscErrorCode PetscDALETKFSetupLocalization_Kokkos(PetscDA_LETKF *);
PETSC_INTERN PetscErrorCode PetscDALETKFDestroyLocalization_Kokkos(PetscDA_LETKF *);
#endif

#pragma once

#include <petscda.h>
#include <petsc/private/petscimpl.h>
#include <petsc/private/hashmapi.h>

PETSC_EXTERN PetscClassId PETSCDA_CLASSID;

typedef struct _PetscDAOps *PetscDAOps;
struct _PetscDAOps {
  PetscErrorCode (*destroy)(PetscDA);
  PetscErrorCode (*setup)(PetscDA);
  PetscErrorCode (*view)(PetscDA, PetscViewer);
  PetscErrorCode (*setfromoptions)(PetscDA, PetscOptionItems *);
};

struct _p_PetscDA {
  PETSCHEADER(struct _PetscDAOps);
  PetscInt obs_size;         /* Observation vector dimension (p) */
  PetscInt local_obs_size;   /* Local observation vector dimension */
  PetscInt state_size;       /* State vector dimension (n) */
  PetscInt local_state_size; /* Local state vector dimension */
  Vec      obs_error_var;    /* Observation error variance (diagonal of R), length p */
  Mat      R;                /* Observation error covariance matrix (p x p) */
  PetscInt ndof;             /* Number of degrees of freedom per grid point; must be the same for all grid points */
  void    *data;
};

/* data common to all the ensemble based PetscDAType */
typedef struct {
  PetscErrorCode (*analysis)(PetscDA, Vec, Mat);
  PetscErrorCode (*forecast)(PetscDA, PetscErrorCode (*)(Vec, Vec, PetscCtx), PetscCtx);
  PetscInt  size;      /* Number of ensemble members (m) */
  Mat       ensemble;  /* Ensemble matrix (n x m) */
  PetscReal inflation; /* Inflation factor */

  /* Algorithm state */
  PetscBool assembled; /* Is the PetscDA object assembled/ready */

  /* T-matrix factorization data (shared across implementations) */
  PetscDASqrtType sqrt_type;       /* Square root factorization type */
  Mat             V;               /* Eigen vectors (LAPACK column-major storage) */
  Mat             L_cholesky;      /* Lower triangular Cholesky factor */
  Vec             sqrt_eigen_vals; /* Square root of eigen values */
  Mat             I_StS;           /* T = I + S^T * S matrix */
} PetscDA_Ensemble;

/* Internal utility functions shared across PetscDA implementations */
PETSC_INTERN PetscErrorCode PetscDASymmetricEigenSqrt_Private(Mat, Mat *);

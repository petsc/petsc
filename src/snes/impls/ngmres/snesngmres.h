#pragma once

#include <petsc/private/snesimpl.h>

/*  Data structure for the Nonlinear GMRES method.  */
typedef struct {
  /* Solver parameters and counters */
  PetscInt    msize;            /* maximum size of krylov space */
  PetscInt    restart_it;       /* number of iterations the restart conditions persist before restart */
  PetscViewer monitor;          /* debugging output for NGMRES */
  PetscInt    restart_periodic; /* number of iterations to restart after */

  SNESNGMRESRestartType restart_type;
  SNESNGMRESSelectType  select_type;

  /* History and subspace data */
  Vec       *Fdot;   /* residual history -- length msize */
  Vec       *Xdot;   /* solution history -- length msize */
  PetscReal *fnorms; /* the residual norm history  */
  PetscReal *xnorms; /* the solution norm history */

  /* General minimization problem context */
  PetscScalar *h;    /* the constraint matrix */
  PetscScalar *beta; /* rhs for the minimization problem */
  PetscScalar *xi;   /* the dot-product of the current and previous res. */

  /* Line searches */
  SNESLineSearch additive_linesearch; /* Line search for the additive variant */
  PetscReal      descent_ls_test;     /* restart GMRES when X_A -X_M is not a descent direction */

  /* Selection constants */
  PetscBool candidate;       /* use candidate storage approach */
  PetscBool approxfunc;      /* approximate the function rather than recomputing it */
  PetscReal gammaA;          /* Criterion A residual tolerance */
  PetscReal epsilonB;        /* Criterion B difference tolerance */
  PetscReal deltaB;          /* Criterion B residual tolerance */
  PetscReal gammaC;          /* Restart residual tolerance */
  PetscBool restart_fm_rise; /* Restart on F_M residual increase */

  PetscReal andersonBeta; /* Relaxation parameter for Anderson Mixing */

  // Window of stored iterates that the least-squares problem uses
  PetscInt  window_size;         /* maximum number of stored iterates in the window, at most msize */
  PetscInt  window_size_initial; /* initial window size of the adaptive strategy */
  PetscBool adaptive;            /* grow the window when theta predicts the residual reduction */
  PetscReal adaptive_tol;        /* tolerance on |theta - ||F_{k+1}|| / ||F_k||| that grows the window */
  PetscReal lsnorm;              /* least-squares objective, the norm of the minimized combination of residuals */
  PetscReal theta;               /* least-squares objective divided by the residual norm ||F_k|| of the current iterate */
  PetscReal gamma;               /* least-squares objective divided by the residual norm of the candidate solution */
  PetscInt *window;              /* history indices of the iterates in the window, in increasing order */
  Vec      *Fwindow;             /* array of references to the residuals in the window, not new Vec objects */
  Vec      *Xwindow;             /* array of references to the solutions in the window, not new Vec objects */

  /* Least squares minimization solve context */
  PetscScalar *q;     /* the matrix formed as q_ij = (rdot_i, rdot_j) */
  PetscBLASInt m;     /* matrix dimension */
  PetscBLASInt n;     /* matrix dimension */
  PetscBLASInt nrhs;  /* the number of right-hand sides */
  PetscBLASInt lda;   /* the padded matrix dimension */
  PetscBLASInt ldb;   /* the padded vector dimension */
  PetscReal   *s;     /* the singular values */
  PetscReal    rcond; /* the exit condition */
  PetscBLASInt rank;  /* the effective rank */
  PetscScalar *work;  /* the work vector */
  PetscReal   *rwork; /* the real work vector used for complex */
  PetscBLASInt lwork; /* the size of the work vector */

  PetscBool setup_called; /* indicates whether SNESSetUp_NGMRES() has been called  */
} SNES_NGMRES;

#define H(i, j) ngmres->h[(i) * ngmres->msize + (j)]
#define Q(i, j) ngmres->q[(i) * ngmres->msize + (j)]

/* private functions that are shared components of the methods */
PETSC_INTERN PetscErrorCode SNESNGMRESUpdateSubspace_Private(SNES, PetscInt, PetscInt, Vec, PetscReal, Vec);
PETSC_INTERN PetscErrorCode SNESNGMRESFormCombinedSolution_Private(SNES, PetscInt, PetscInt, Vec, Vec, PetscReal, Vec, Vec, Vec);
PETSC_INTERN PetscErrorCode SNESNGMRESNorms_Private(SNES, PetscInt, Vec, Vec, Vec, Vec, Vec, Vec, Vec, PetscReal *, PetscReal *, PetscReal *, PetscReal *, PetscReal *, PetscReal *, PetscReal *, PetscReal *);
PETSC_INTERN PetscErrorCode SNESNGMRESSelect_Private(SNES, PetscInt, Vec, Vec, PetscReal, PetscReal, PetscReal, PetscReal, Vec, Vec, PetscReal, PetscReal, PetscReal, PetscReal, PetscReal, PetscReal, PetscReal, Vec, Vec, Vec, PetscReal *, PetscReal *, PetscReal *);
PETSC_INTERN PetscErrorCode SNESNGMRESSelectRestart_Private(SNES, PetscInt, PetscReal, PetscReal, PetscReal, PetscReal, PetscReal, PetscReal, PetscBool *);

PETSC_INTERN PetscErrorCode SNESNGMRESGetAdditiveLineSearch_Private(SNES, SNESLineSearch *);
PETSC_INTERN PetscErrorCode SNESDestroy_NGMRES(SNES);
PETSC_INTERN PetscErrorCode SNESReset_NGMRES(SNES);
PETSC_INTERN PetscErrorCode SNESSetUp_NGMRES(SNES);
PETSC_INTERN PetscErrorCode SNESView_NGMRES(SNES, PetscViewer);

#ifndef _SNESNGMRES_H
#define _SNESNGMRES_H

#include <petsc-private/snesimpl.h>

/*  Data structure for the Nonlinear GMRES method.  */
typedef struct {

  /* Solver parameters and counters */
  PetscInt     msize;          /* maximum size of krylov space */
  PetscInt     restart_it;     /* number of iterations the restart conditions persist before restart */
  PetscViewer  monitor;        /* debugging output for NGMRES */
  PetscInt    restart_periodic;/* number of iterations to restart after */

  SNESNGMRESRestartType   restart_type;
  SNESNGMRESSelectType    select_type;

  /* History and subspace data */
  Vec          *Fdot;          /* residual history -- length msize */
  Vec          *Xdot;          /* solution history -- length msize */
  PetscReal    *fnorms;        /* the residual norm history  */
  PetscReal    *xnorms;        /* the solution norm history */

  /* General minimization problem context */
  PetscScalar  *h;             /* the constraint matrix */
  PetscScalar  *beta;          /* rhs for the minimization problem */
  PetscScalar  *xi;            /* the dot-product of the current and previous res. */

  /* Line searches */
  SNESLineSearch   additive_linesearch; /* Line search for the additive variant */

  /* Selection constants */
  PetscBool    anderson;       /* use anderson-mixing approach */
  PetscBool    singlereduction;/* use a single reduction (with more local work) for tolerance selection */
  PetscReal    gammaA;         /* Criterion A residual tolerance */
  PetscReal    epsilonB;       /* Criterion B difference tolerance */
  PetscReal    deltaB;         /* Criterion B residual tolerance */
  PetscReal    gammaC;         /* Restart residual tolerance */

  /* Least squares minimization solve context */
  PetscScalar  *q;             /* the matrix formed as q_ij = (rdot_i, rdot_j) */
  PetscBLASInt m;              /* matrix dimension */
  PetscBLASInt n;              /* matrix dimension */
  PetscBLASInt nrhs;           /* the number of right hand sides */
  PetscBLASInt lda;            /* the padded matrix dimension */
  PetscBLASInt ldb;            /* the padded vector dimension */
  PetscReal    *s;             /* the singular values */
  PetscReal    rcond;          /* the exit condition */
  PetscBLASInt rank;           /* the effective rank */
  PetscScalar  *work;          /* the work vector */
  PetscReal    *rwork;         /* the real work vector used for complex */
  PetscBLASInt lwork;          /* the size of the work vector */
  PetscBLASInt info;           /* the output condition */

  PetscBool    setup_called;    /* indicates whether SNESSetUp_NGMRES() has been called  */
} SNES_NGMRES;

#define H(i,j)  ngmres->h[i*ngmres->msize + j]
#define Q(i,j)  ngmres->q[i*ngmres->msize + j]

#endif

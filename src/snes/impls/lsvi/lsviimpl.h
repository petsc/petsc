/* 
   Private context for semismooth newton method with line search for solving
   system of mixed complementarity equations
 */

#ifndef __SNES_LSVI_H
#define __SNES_LSVI_H
#include "private/snesimpl.h"

#define PETSC_LSVI_INF  1.0e20
#define PETSC_LSVI_EPS  DBL_EPSILON

typedef struct {
  PetscErrorCode           (*LineSearch)(SNES,void*,Vec,Vec,Vec,Vec,Vec,PetscReal,PetscReal,PetscReal*,PetscReal*,PetscTruth*);
  void                     *lsP;                              /* user-defined line-search context (optional) */
  /* --------------- Parameters used by line search method ----------------- */
  PetscReal                alpha;		                                                   /* used to determine sufficient reduction */
  PetscReal                maxstep;                                                          /* maximum step size */
  PetscReal                minlambda;                                                        /* determines smallest line search lambda used */
  PetscErrorCode           (*precheckstep)(SNES,Vec,Vec,void*,PetscTruth*);                  /* step-checking routine (optional) */
  void                     *precheck;                                                        /* user-defined step-checking context (optional) */
  PetscErrorCode           (*postcheckstep)(SNES,Vec,Vec,Vec,void*,PetscTruth*,PetscTruth*); /* step-checking routine (optional) */
  void                     *postcheck;                                                       /* user-defined step-checking context (optional) */
  PetscViewerASCIIMonitor  monitor;

  /* ------------------ Semismooth algorithm stuff ------------------------------ */
  Vec                      phi;                      /* pointer to semismooth function */
  PetscErrorCode           (*computessfunction)(SNES,Vec); /* Semismooth function evaluation routine */
  PetscScalar              psi;                                        /* Merit function */
  PetscErrorCode           (*computemeritfunction)(SNES,PetscScalar*); /* function to compute merit function */
  Vec                      dpsi;          /* Gradient of merit function */
  PetscErrorCode           (*computemeritfunctiongradient)(SNES,Vec);
  Mat                      Bsubd;         /* B sub-differential matrix */
  Vec                      Da;            /* B sub-differential work vector (diag perturbation) */
  Vec                      Db;            /* B sub-differential work vector (row scaling) */
  Vec                      xl;            /* lower bound on variables */
  Vec                      xu;            /* upper bound on variables */

  PetscScalar             norm_d;         /* two norm of the descent direction */
  /* Parameters for checking sufficient descent conditions satisfied */
  PetscReal             rho;
  PetscReal             delta;
} SNES_LSVI;

#endif


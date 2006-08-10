
#ifndef __TSIMPL_H
#define __TSIMPL_H

#include "petscts.h"

/*
    Timesteping context. 
      General case: U_t = F(t,U) <-- the right-hand-side function
      Linear  case: U_t = A(t) U <-- the right-hand-side matrix
      Linear (no time) case: U_t = A U <-- the right-hand-side matrix
*/

/*
     Maximum number of monitors you can run with a single TS
*/
#define MAXTSMONITORS 5 

struct _TSOps {
  PetscErrorCode (*rhsmatrix)(TS, PetscReal, Mat *, Mat *, MatStructure *, void *);
  PetscErrorCode (*lhsmatrix)(TS, PetscReal, Mat *, Mat *, MatStructure *, void *);
  PetscErrorCode (*rhsfunction)(TS, PetscReal, Vec, Vec, void *);
  PetscErrorCode (*rhsjacobian)(TS, PetscReal, Vec, Mat *, Mat *, MatStructure *, void *);
  PetscErrorCode (*prestep)(TS);
  PetscErrorCode (*update)(TS, PetscReal, PetscReal *);
  PetscErrorCode (*poststep)(TS);
  PetscErrorCode (*reform)(TS);
  PetscErrorCode (*reallocate)(TS);
  PetscErrorCode (*setup)(TS);
  PetscErrorCode (*step)(TS,PetscInt *, PetscReal *);
  PetscErrorCode (*setfromoptions)(TS);
  PetscErrorCode (*destroy)(TS);
  PetscErrorCode (*view)(TS, PetscViewer);
};

struct _p_TS {
  PETSCHEADER(struct _TSOps);
  TSProblemType problem_type;
  Vec           vec_sol, vec_sol_always;

  /* ---------------- User (or PETSc) Provided stuff ---------------------*/
  PetscErrorCode (*monitor[MAXTSMONITORS])(TS,PetscInt,PetscReal,Vec,void*); /* returns control to user after */
  PetscErrorCode (*mdestroy[MAXTSMONITORS])(void*);                
  void *monitorcontext[MAXTSMONITORS];                 /* residual calculation, allows user */
  PetscInt  numbermonitors;                                 /* to, for instance, print residual norm, etc. */

  /* Identifies this as a grid TS structure */
  PetscTruth *isExplicit;                            /* Indicates which fields have explicit time dependence */
  PetscInt   *Iindex;                                /* The index of the identity for each time dependent field */

  /* ---------------------Linear Iteration---------------------------------*/
  KSP ksp;
  Mat A, B;                                         /* user provided matrix and preconditioner */
  Mat Alhs, Blhs;                    /* user provided left hand side matrix and preconditioner */

  /* ---------------------Nonlinear Iteration------------------------------*/
  SNES  snes;
  void *funP;
  void *jacP,*jacPlhs;
  void *bcP;


  /* --- Data that is unique to each particular solver --- */
  PetscInt setupcalled;            /* true if setup has been called */
  void     *data;                   /* implementationspecific data */
  void     *user;                   /* user context */

  /* ------------------  Parameters -------------------------------------- */
  PetscInt  max_steps;              /* max number of steps */
  PetscReal max_time;               /* max time allowed */
  PetscReal time_step;              /* current time increment */
  PetscReal time_step_old;          /* previous time increment */
  PetscReal initial_time_step;      /* initial time increment */
  PetscInt  steps;                  /* steps taken so far */
  PetscReal ptime;                  /* time taken so far */
  PetscInt  linear_its;             /* total number of linear solver iterations */
  PetscInt  nonlinear_its;          /* total number of nonlinear solver iterations */

  /* ------------------- Default work-area management ------------------ */
  PetscInt nwork;              
  Vec      *work;
};

EXTERN PetscErrorCode TSMonitor(TS,PetscInt,PetscReal,Vec);
EXTERN PetscErrorCode TSScaleShiftMatrices(TS,Mat,Mat,MatStructure);

extern PetscEvent    TS_Step, TS_PseudoComputeTimeStep, TS_FunctionEval, TS_JacobianEval;

#endif

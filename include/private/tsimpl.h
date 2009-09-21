
#ifndef __TSIMPL_H
#define __TSIMPL_H

#include "petscts.h"

/*
    Timesteping context.
      General DAE: F(t,U,U_t) = 0, required Jacobian is G'(U) where G(U) = F(t,U,U0+a*U)
      General ODE: U_t = F(t,U) <-- the right-hand-side function
      Linear  ODE: U_t = A(t) U <-- the right-hand-side matrix
      Linear (no time) ODE: U_t = A U <-- the right-hand-side matrix
*/

/*
     Maximum number of monitors you can run with a single TS
*/
#define MAXTSMONITORS 5 

struct _TSOps {
  PetscErrorCode (*rhsmatrix)(TS,PetscReal,Mat*,Mat*,MatStructure*,void*);
  PetscErrorCode (*lhsmatrix)(TS,PetscReal,Mat*,Mat*,MatStructure*,void*);
  PetscErrorCode (*rhsfunction)(TS,PetscReal,Vec,Vec,void*);
  PetscErrorCode (*rhsjacobian)(TS,PetscReal,Vec,Mat*,Mat*,MatStructure*,void*);
  PetscErrorCode (*ifunction)(TS,PetscReal,Vec,Vec,Vec,void*);
  PetscErrorCode (*ijacobian)(TS,PetscReal,Vec,Vec,PetscReal,Mat*,Mat*,MatStructure*,void*);
  PetscErrorCode (*prestep)(TS);
  PetscErrorCode (*poststep)(TS);
  PetscErrorCode (*setup)(TS);
  PetscErrorCode (*step)(TS,PetscInt*,PetscReal*);
  PetscErrorCode (*setfromoptions)(TS);
  PetscErrorCode (*destroy)(TS);
  PetscErrorCode (*view)(TS,PetscViewer);
};

struct _p_TS {
  PETSCHEADER(struct _TSOps);
  TSProblemType problem_type;
  Vec           vec_sol,vec_sol_always;

  /* ---------------- User (or PETSc) Provided stuff ---------------------*/
  PetscErrorCode (*monitor[MAXTSMONITORS])(TS,PetscInt,PetscReal,Vec,void*); /* returns control to user after */
  PetscErrorCode (*mdestroy[MAXTSMONITORS])(void*);                
  void *monitorcontext[MAXTSMONITORS];                 /* residual calculation, allows user */
  PetscInt  numbermonitors;                                 /* to, for instance, print residual norm, etc. */

  /* ---------------------Linear Iteration---------------------------------*/
  KSP ksp;
  Mat A,B;           /* internel matrix and preconditioner used for KSPSolve() */
  Mat Arhs,Alhs;     /* user provided right/left hand side matrix and preconditioner */
  MatStructure matflg; /* flag indicating the matrix structure of Arhs and Alhs */

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

extern PetscLogEvent TS_Step, TS_PseudoComputeTimeStep, TS_FunctionEval, TS_JacobianEval;

#endif

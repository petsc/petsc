
#ifndef __TSIMPL_H
#define __TSIMPL_H

#include <petscts.h>

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

typedef struct _TSOps *TSOps;

struct _TSOps {
  PetscErrorCode (*snesfunction)(SNES,Vec,Vec,TS);
  PetscErrorCode (*snesjacobian)(SNES,Vec,Mat*,Mat*,MatStructure*,TS);
  PetscErrorCode (*prestep)(TS);
  PetscErrorCode (*poststep)(TS);
  PetscErrorCode (*setup)(TS);
  PetscErrorCode (*step)(TS);
  PetscErrorCode (*solve)(TS);
  PetscErrorCode (*setfromoptions)(TS);
  PetscErrorCode (*destroy)(TS);
  PetscErrorCode (*view)(TS,PetscViewer);
  PetscErrorCode (*reset)(TS);
};

struct _TSUserOps {
  PetscErrorCode (*rhsfunction)(TS,PetscReal,Vec,Vec,void*);
  PetscErrorCode (*rhsjacobian)(TS,PetscReal,Vec,Mat*,Mat*,MatStructure*,void*);
  PetscErrorCode (*ifunction)(TS,PetscReal,Vec,Vec,Vec,void*);
  PetscErrorCode (*ijacobian)(TS,PetscReal,Vec,Vec,PetscReal,Mat*,Mat*,MatStructure*,void*);
};

struct _p_TS {
  PETSCHEADER(struct _TSOps);

  struct _TSUserOps *userops;
  DM            dm;
  TSProblemType problem_type;
  Vec           vec_sol;

  /* ---------------- User (or PETSc) Provided stuff ---------------------*/
  PetscErrorCode (*monitor[MAXTSMONITORS])(TS,PetscInt,PetscReal,Vec,void*); /* returns control to user after */
  PetscErrorCode (*mdestroy[MAXTSMONITORS])(void**);                
  void *monitorcontext[MAXTSMONITORS];                 /* residual calculation, allows user */
  PetscInt  numbermonitors;                                 /* to, for instance, print residual norm, etc. */

  /* ---------------------- IMEX support ---------------------------------*/
  /* These extra slots are only used when the user provides both Implicit and RHS */
  Mat Arhs;     /* Right hand side matrix */
  Mat Brhs;     /* Right hand side preconditioning matrix */
  Vec Frhs;     /* Right hand side function value */

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
  PetscReal time_step;              /* current/completed time increment */
  PetscReal next_time_step;         /* expected next time step (but may end up being different, e.g. if the step is rejected) */
  PetscReal initial_time_step;      /* initial time increment */
  PetscInt  steps;                  /* steps taken so far */
  PetscReal ptime;                  /* time at the start of the current step (stage time is internal if it exists) */
  PetscInt  linear_its;             /* total number of linear solver iterations */
  PetscInt  nonlinear_its;          /* total number of nonlinear solver iterations */

  PetscInt num_snes_failures;
  PetscInt max_snes_failures;
  TSConvergedReason reason;
  PetscBool errorifstepfailed;
  PetscBool exact_final_time;
  PetscInt reject,max_reject;

  /* ------------------- Default work-area management ------------------ */
  PetscInt nwork;              
  Vec      *work;
};

extern PetscErrorCode TSMonitor(TS,PetscInt,PetscReal,Vec);

extern PetscLogEvent TS_Step, TS_PseudoComputeTimeStep, TS_FunctionEval, TS_JacobianEval;

#endif


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
  PetscErrorCode (*interpolate)(TS,PetscReal,Vec);
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

  /* This is a general caching scheme to avoid recomputing the Jacobian at a place that has been previously been evaluated.
   * The present use case is that TSComputeRHSFunctionLinear() evaluates the Jacobian once and we don't want it to be immeditely re-evaluated.
   */
  struct {
    PetscReal time;             /* The time at which the matrices were last evaluated */
    Vec X;                      /* Solution vector at which the Jacobian was last evaluated */
    PetscInt Xstate;            /* State of the solution vector */
    MatStructure mstructure;    /* The structure returned */
  } rhsjacobian;

  struct {
    PetscReal time;             /* The time at which the matrices were last evaluated */
    Vec X;                      /* Solution vector at which the Jacobian was last evaluated */
    Vec Xdot;                   /* Time derivative of the state vector at which the Jacobian was last evaluated */
    PetscInt Xstate;            /* State of the solution vector */
    PetscInt Xdotstate;         /* State of the solution vector */
    MatStructure mstructure;    /* The structure returned */
    PetscReal shift;            /* The derivative of the lhs wrt to Xdot */
    PetscBool imex;             /* Flag of the method if it was started as an imex method */
  } ijacobian;

  /* ---------------------Nonlinear Iteration------------------------------*/
  SNES  snes;
  void *funP;
  void *jacP;


  /* --- Data that is unique to each particular solver --- */
  PetscInt setupcalled;             /* true if setup has been called */
  void     *data;                   /* implementationspecific data */
  void     *user;                   /* user context */

  /* ------------------  Parameters -------------------------------------- */
  PetscInt  max_steps;              /* max number of steps */
  PetscReal max_time;               /* max time allowed */
  PetscReal time_step;              /* current/completed time increment */
  PetscReal time_step_prev;         /* previous time step  */
  PetscInt  steps;                  /* steps taken so far */
  PetscReal ptime;                  /* time at the start of the current step (stage time is internal if it exists) */
  PetscInt  linear_its;             /* total number of linear solver iterations */
  PetscInt  nonlinear_its;          /* total number of nonlinear solver iterations */

  PetscInt num_snes_failures;
  PetscInt max_snes_failures;
  TSConvergedReason reason;
  PetscBool errorifstepfailed;
  PetscInt  exact_final_time;   /* PETSC_DECIDE, PETSC_TRUE, or PETSC_FALSE */
  PetscBool retain_stages;
  PetscInt reject,max_reject;

  /* ------------------- Default work-area management ------------------ */
  PetscInt nwork;
  Vec      *work;
};


extern PetscLogEvent TS_Step, TS_PseudoComputeTimeStep, TS_FunctionEval, TS_JacobianEval;

#endif

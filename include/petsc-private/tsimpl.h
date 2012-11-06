
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
  PetscErrorCode (*prestage)(TS,PetscReal);
  PetscErrorCode (*poststep)(TS);
  PetscErrorCode (*setup)(TS);
  PetscErrorCode (*step)(TS);
  PetscErrorCode (*solve)(TS);
  PetscErrorCode (*interpolate)(TS,PetscReal,Vec);
  PetscErrorCode (*evaluatestep)(TS,PetscInt,Vec,PetscBool*);
  PetscErrorCode (*setfromoptions)(TS);
  PetscErrorCode (*destroy)(TS);
  PetscErrorCode (*view)(TS,PetscViewer);
  PetscErrorCode (*reset)(TS);
  PetscErrorCode (*linearstability)(TS,PetscReal,PetscReal,PetscReal*,PetscReal*);
};

struct _TSUserOps {
  PetscErrorCode (*rhsfunction)(TS,PetscReal,Vec,Vec,void*);
  PetscErrorCode (*rhsjacobian)(TS,PetscReal,Vec,Mat*,Mat*,MatStructure*,void*);
  PetscErrorCode (*ifunction)(TS,PetscReal,Vec,Vec,Vec,void*);
  PetscErrorCode (*ijacobian)(TS,PetscReal,Vec,Vec,PetscReal,Mat*,Mat*,MatStructure*,void*);
};

struct _p_TS {
  PETSCHEADER(struct _TSOps);
  DM            dm;
  TSProblemType problem_type;
  Vec           vec_sol;
  TSAdapt       adapt;

  /* ---------------- User (or PETSc) Provided stuff ---------------------*/
  PetscErrorCode (*monitor[MAXTSMONITORS])(TS,PetscInt,PetscReal,Vec,void*); /* returns control to user after */
  PetscErrorCode (*monitordestroy[MAXTSMONITORS])(void**);
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

  /* --- Data that is unique to each particular solver --- */
  PetscInt setupcalled;             /* true if setup has been called */
  void     *data;                   /* implementationspecific data */
  void     *user;                   /* user context */

  /* ------------------  Parameters -------------------------------------- */
  PetscInt  max_steps;              /* max number of steps */
  PetscReal max_time;               /* max time allowed */
  PetscReal time_step;              /* current/completed time increment */
  PetscReal time_step_prev;         /* previous time step  */

  /*
     these are temporary to support increasing the time step if nonlinear solver convergence remains good
     and time_step was previously cut due to failed nonlinear solver
  */
  PetscReal time_step_orig;            /* original time step requested by user */
  PetscInt  time_steps_since_decrease; /* number of timesteps since timestep was decreased due to lack of convergence */
  /* ----------------------------------------------------------------------------------------------------------------*/

  PetscInt  steps;                  /* steps taken so far */
  PetscReal ptime;                  /* time at the start of the current step (stage time is internal if it exists) */
  PetscInt  ksp_its;                /* total number of linear solver iterations */
  PetscInt  snes_its;               /* total number of nonlinear solver iterations */

  PetscInt num_snes_failures;
  PetscInt max_snes_failures;
  TSConvergedReason reason;
  PetscBool errorifstepfailed;
  PetscInt  exact_final_time;   /* PETSC_DECIDE, PETSC_TRUE, or PETSC_FALSE */
  PetscBool retain_stages;
  PetscInt reject,max_reject;

  PetscReal atol,rtol;          /* Relative and absolute tolerance for local truncation error */
  Vec       vatol,vrtol;        /* Relative and absolute tolerance in vector form */
  PetscReal cfltime,cfltime_local;

  /* ------------------- Default work-area management ------------------ */
  PetscInt nwork;
  Vec      *work;
};

struct _TSAdaptOps {
  PetscErrorCode (*choose)(TSAdapt,TS,PetscReal,PetscInt*,PetscReal*,PetscBool*,PetscReal*);
  PetscErrorCode (*checkstage)(TSAdapt,TS,PetscBool*);
  PetscErrorCode (*destroy)(TSAdapt);
  PetscErrorCode (*view)(TSAdapt,PetscViewer);
  PetscErrorCode (*setfromoptions)(TSAdapt);
};

struct _p_TSAdapt {
  PETSCHEADER(struct _TSAdaptOps);
  void *data;
  struct {
    PetscInt   n;                /* number of candidate schemes, including the one currently in use */
    PetscBool  inuse_set;        /* the current scheme has been set */
    const char *name[16];        /* name of the scheme */
    PetscInt   order[16];        /* classical order of each scheme */
    PetscInt   stageorder[16];   /* stage order of each scheme */
    PetscReal  ccfl[16];         /* stability limit relative to explicit Euler */
    PetscReal  cost[16];         /* relative measure of the amount of work required for each scheme */
  } candidates;
  PetscReal   dt_min,dt_max;
  PetscReal   scale_solve_failed; /* Scale step by this factor if solver (linear or nonlinear) fails. */
  PetscViewer monitor;
};

typedef struct _n_TSDM *TSDM;
struct _n_TSDM {
  TSRHSFunction rhsfunction;
  TSRHSJacobian rhsjacobian;

  TSIFunction ifunction;
  TSIJacobian ijacobian;

  TSSolutionFunction solution;

  void *rhsfunctionctx;
  void *rhsjacobianctx;

  void *ifunctionctx;
  void *ijacobianctx;

  void *solutionctx;


  /* This context/destroy pair allows implementation-specific routines such as DMDA local functions. */
  PetscErrorCode (*destroy)(TSDM);
  void *data;

  /* This is NOT reference counted. The SNES that originally created this context is cached here to implement copy-on-write.
   * Fields in the TSDM should only be written if the SNES matches originalsnes.
   */
  DM originaldm;
};

PETSC_EXTERN PetscErrorCode DMTSGetContext(DM,TSDM*);
PETSC_EXTERN PetscErrorCode DMTSGetContextWrite(DM,TSDM*);
PETSC_EXTERN PetscErrorCode DMTSCopyContext(DM,DM);
PETSC_EXTERN PetscErrorCode DMTSDuplicateContext(DM,DM);
PETSC_EXTERN PetscErrorCode DMTSSetUpLegacy(DM);



PETSC_EXTERN PetscLogEvent TS_Step, TS_PseudoComputeTimeStep, TS_FunctionEval, TS_JacobianEval;

typedef enum {TS_STEP_INCOMPLETE, /* vec_sol, ptime, etc point to beginning of step */
              TS_STEP_PENDING,    /* vec_sol advanced, but step has not been accepted yet */
              TS_STEP_COMPLETE    /* step accepted and ptime, steps, etc have been advanced */
} TSStepStatus;

#endif

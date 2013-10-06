
#ifndef __TSIMPL_H
#define __TSIMPL_H

#include <petscts.h>
#include <petsc-private/petscimpl.h>

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
  PetscErrorCode (*load)(TS,PetscViewer);
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

  PetscErrorCode (*prestep)(TS);
  PetscErrorCode (*prestage)(TS,PetscReal);
  PetscErrorCode (*poststep)(TS);

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
    PetscObjectState Xstate;    /* State of the solution vector */
    MatStructure mstructure;    /* The structure returned */
    /* Flag to unshift Jacobian before calling the IJacobian or RHSJacobian functions.  This is useful
     * if the user would like to reuse (part of) the Jacobian from the last evaluation. */
    PetscBool reuse;
    PetscReal scale,shift;
  } rhsjacobian;

  struct {
    PetscReal shift;            /* The derivative of the lhs wrt to Xdot */
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
  PetscReal solvetime;              /* time at the conclusion of TSSolve() */
  PetscInt  ksp_its;                /* total number of linear solver iterations */
  PetscInt  snes_its;               /* total number of nonlinear solver iterations */

  PetscInt num_snes_failures;
  PetscInt max_snes_failures;
  TSConvergedReason reason;
  TSEquationType equation_type;
  PetscBool errorifstepfailed;
  TSExactFinalTimeOption  exact_final_time;
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
  PetscErrorCode (*load)(TSAdapt,PetscViewer);
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

typedef struct _p_DMTS *DMTS;
typedef struct _DMTSOps *DMTSOps;
struct _DMTSOps {
  TSRHSFunction rhsfunction;
  TSRHSJacobian rhsjacobian;

  TSIFunction ifunction;
  PetscErrorCode (*ifunctionview)(void*,PetscViewer);
  PetscErrorCode (*ifunctionload)(void**,PetscViewer);

  TSIJacobian ijacobian;
  PetscErrorCode (*ijacobianview)(void*,PetscViewer);
  PetscErrorCode (*ijacobianload)(void**,PetscViewer);

  TSSolutionFunction solution;
  PetscErrorCode (*forcing)(TS,PetscReal,Vec,void*);

  PetscErrorCode (*destroy)(DMTS);
  PetscErrorCode (*duplicate)(DMTS,DMTS);
};

struct _p_DMTS {
  PETSCHEADER(struct _DMTSOps);
  void *rhsfunctionctx;
  void *rhsjacobianctx;

  void *ifunctionctx;
  void *ijacobianctx;

  void *solutionctx;
  void *forcingctx;

  void *data;

  /* This is NOT reference counted. The DM on which this context was first created is cached here to implement one-way
   * copy-on-write. When DMGetDMTSWrite() sees a request using a different DM, it makes a copy. Thus, if a user
   * only interacts directly with one level, e.g., using TSSetIFunction(), then coarse levels of a multilevel item
   * integrator are built, then the user changes the routine with another call to TSSetIFunction(), it automatically
   * propagates to all the levels. If instead, they get out a specific level and set the function on that level,
   * subsequent changes to the original level will no longer propagate to that level.
   */
  DM originaldm;
};

PETSC_EXTERN PetscErrorCode DMGetDMTS(DM,DMTS*);
PETSC_EXTERN PetscErrorCode DMGetDMTSWrite(DM,DMTS*);
PETSC_EXTERN PetscErrorCode DMCopyDMTS(DM,DM);
PETSC_EXTERN PetscErrorCode DMTSView(DMTS,PetscViewer);
PETSC_EXTERN PetscErrorCode DMTSLoad(DMTS,PetscViewer);


PETSC_EXTERN PetscLogEvent TS_Step, TS_PseudoComputeTimeStep, TS_FunctionEval, TS_JacobianEval;

typedef enum {TS_STEP_INCOMPLETE, /* vec_sol, ptime, etc point to beginning of step */
              TS_STEP_PENDING,    /* vec_sol advanced, but step has not been accepted yet */
              TS_STEP_COMPLETE    /* step accepted and ptime, steps, etc have been advanced */
} TSStepStatus;

#endif

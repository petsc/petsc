#ifndef __TSIMPL_H
#define __TSIMPL_H

#include <petscts.h>
#include <petsc/private/petscimpl.h>

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
#define MAXTSMONITORS 10

PETSC_EXTERN PetscBool TSRegisterAllCalled;
PETSC_EXTERN PetscErrorCode TSRegisterAll(void);
PETSC_EXTERN PetscErrorCode TSAdaptRegisterAll(void);

PETSC_EXTERN PetscErrorCode TSRKRegisterAll(void);
PETSC_EXTERN PetscErrorCode TSARKIMEXRegisterAll(void);
PETSC_EXTERN PetscErrorCode TSRosWRegisterAll(void);
PETSC_EXTERN PetscErrorCode TSGLRegisterAll(void);
PETSC_EXTERN PetscErrorCode TSGLAdaptRegisterAll(void);

typedef struct _TSOps *TSOps;

struct _TSOps {
  PetscErrorCode (*snesfunction)(SNES,Vec,Vec,TS);
  PetscErrorCode (*snesjacobian)(SNES,Vec,Mat,Mat,TS);
  PetscErrorCode (*setup)(TS);
  PetscErrorCode (*step)(TS);
  PetscErrorCode (*solve)(TS);
  PetscErrorCode (*interpolate)(TS,PetscReal,Vec);
  PetscErrorCode (*evaluatewlte)(TS,NormType,PetscInt*,PetscReal*);
  PetscErrorCode (*evaluatestep)(TS,PetscInt,Vec,PetscBool*);
  PetscErrorCode (*setfromoptions)(PetscOptionItems*,TS);
  PetscErrorCode (*destroy)(TS);
  PetscErrorCode (*view)(TS,PetscViewer);
  PetscErrorCode (*reset)(TS);
  PetscErrorCode (*linearstability)(TS,PetscReal,PetscReal,PetscReal*,PetscReal*);
  PetscErrorCode (*load)(TS,PetscViewer);
  PetscErrorCode (*rollback)(TS);
  PetscErrorCode (*getstages)(TS,PetscInt*,Vec**);
  PetscErrorCode (*adjointstep)(TS);
  PetscErrorCode (*adjointsetup)(TS);
  PetscErrorCode (*adjointintegral)(TS);
  PetscErrorCode (*forwardintegral)(TS);
};

/*
   TSEvent - Abstract object to handle event monitoring
*/
typedef struct _n_TSEvent *TSEvent;

typedef struct _TSTrajectoryOps *TSTrajectoryOps;

struct _TSTrajectoryOps {
  PetscErrorCode (*view)(TSTrajectory,PetscViewer);
  PetscErrorCode (*destroy)(TSTrajectory);
  PetscErrorCode (*set)(TSTrajectory,TS,PetscInt,PetscReal,Vec);
  PetscErrorCode (*get)(TSTrajectory,TS,PetscInt,PetscReal*);
  PetscErrorCode (*setfromoptions)(PetscOptionItems*,TSTrajectory);
  PetscErrorCode (*setup)(TSTrajectory,TS);
};

struct _p_TSTrajectory {
  PETSCHEADER(struct _TSTrajectoryOps);
  PetscInt setupcalled;             /* true if setup has been called */
  PetscInt recomps;                 /* counter for recomputations in the adjoint run */
  PetscInt diskreads,diskwrites;    /* counters for disk checkpoint reads and writes */
  void *data;
};

struct _p_TS {
  PETSCHEADER(struct _TSOps);
  TSProblemType  problem_type;
  TSEquationType equation_type;

  DM             dm;
  Vec            vec_sol; /* solution vector in first and second order equations */
  Vec            vec_dot; /* time derivative vector in second order equations */
  TSAdapt        adapt;
  TSEvent        event;

  /* ---------------- User (or PETSc) Provided stuff ---------------------*/
  PetscErrorCode (*monitor[MAXTSMONITORS])(TS,PetscInt,PetscReal,Vec,void*);
  PetscErrorCode (*monitordestroy[MAXTSMONITORS])(void**);
  void *monitorcontext[MAXTSMONITORS];
  PetscInt  numbermonitors;
  PetscErrorCode (*adjointmonitor[MAXTSMONITORS])(TS,PetscInt,PetscReal,Vec,PetscInt,Vec*,Vec*,void*);
  PetscErrorCode (*adjointmonitordestroy[MAXTSMONITORS])(void**);
  void *adjointmonitorcontext[MAXTSMONITORS];
  PetscInt  numberadjointmonitors;

  PetscErrorCode (*prestep)(TS);
  PetscErrorCode (*prestage)(TS,PetscReal);
  PetscErrorCode (*poststage)(TS,PetscReal,PetscInt,Vec*);
  PetscErrorCode (*poststep)(TS);
  PetscErrorCode (*functiondomainerror)(TS,PetscReal,Vec,PetscBool*);

  /* ---------------------- Sensitivity Analysis support -----------------*/
  TSTrajectory trajectory;          /* All solutions are kept here for the entire time integration process */
  Vec       *vecs_sensi;            /* one vector for each cost function */
  Vec       *vecs_sensip;
  PetscInt  numcost;                /* number of cost functions */
  Vec       vec_costintegral;
  PetscInt  adjointsetupcalled;
  PetscInt  adjoint_max_steps;
  PetscBool adjoint_solve;          /* immediately call TSAdjointSolve() after TSSolve() is complete */
  PetscBool costintegralfwd;        /* cost integral is evaluated in the forward run if true */
  Vec       vec_costintegrand;      /* workspace for Adjoint computations */
  Mat       Jacp;
  void      *rhsjacobianpctx;
  void      *costintegrandctx;
  Vec       *vecs_drdy;
  Vec       *vecs_drdp;

  PetscErrorCode (*rhsjacobianp)(TS,PetscReal,Vec,Mat,void*);
  PetscErrorCode (*costintegrand)(TS,PetscReal,Vec,Vec,void*);
  PetscErrorCode (*drdyfunction)(TS,PetscReal,Vec,Vec*,void*);
  PetscErrorCode (*drdpfunction)(TS,PetscReal,Vec,Vec*,void*);

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

  /* --------------------Nonlinear Iteration------------------------------*/
  SNES     snes;
  PetscInt ksp_its;                /* total number of linear solver iterations */
  PetscInt snes_its;               /* total number of nonlinear solver iterations */
  PetscInt num_snes_failures;
  PetscInt max_snes_failures;

  /* --- Data that is unique to each particular solver --- */
  PetscInt setupcalled;             /* true if setup has been called */
  void     *data;                   /* implementationspecific data */
  void     *user;                   /* user context */

  /* ------------------  Parameters -------------------------------------- */
  PetscInt  max_steps;              /* max number of steps */
  PetscReal max_time;               /* max time allowed */

  /* --------------------------------------------------------------------- */

  PetscBool steprollback;           /* flag to indicate that the step was rolled back */
  PetscBool steprestart;            /* flag to indicate that the timestepper has to discard any history and restart */
  PetscInt  steps;                  /* steps taken so far in latest call to TSSolve() */
  PetscInt  total_steps;            /* steps taken in all calls to TSSolve() since the TS was created or since TSSetUp() was called */
  PetscReal ptime;                  /* time at the start of the current step (stage time is internal if it exists) */
  PetscReal time_step;              /* current time increment */
  PetscReal ptime_prev;             /* time at the start of the previous step */
  PetscReal ptime_prev_rollback;    /* time at the start of the 2nd previous step to recover from rollback */
  PetscReal solvetime;              /* time at the conclusion of TSSolve() */

  TSConvergedReason reason;
  PetscBool errorifstepfailed;
  PetscInt  reject,max_reject;
  TSExactFinalTimeOption exact_final_time;

  PetscReal atol,rtol;              /* Relative and absolute tolerance for local truncation error */
  Vec       vatol,vrtol;            /* Relative and absolute tolerance in vector form */
  PetscReal cfltime,cfltime_local;

  /* ------------------- Default work-area management ------------------ */
  PetscInt nwork;
  Vec      *work;
};

struct _TSAdaptOps {
  PetscErrorCode (*choose)(TSAdapt,TS,PetscReal,PetscInt*,PetscReal*,PetscBool*,PetscReal*);
  PetscErrorCode (*destroy)(TSAdapt);
  PetscErrorCode (*reset)(TSAdapt);
  PetscErrorCode (*view)(TSAdapt,PetscViewer);
  PetscErrorCode (*setfromoptions)(PetscOptionItems*,TSAdapt);
  PetscErrorCode (*load)(TSAdapt,PetscViewer);
};

struct _p_TSAdapt {
  PETSCHEADER(struct _TSAdaptOps);
  void *data;
  PetscErrorCode (*checkstage)(TSAdapt,TS,PetscReal,Vec,PetscBool*);
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
  NormType    wnormtype;
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

  TSI2Function i2function;
  TSI2Jacobian i2jacobian;

  TSSolutionFunction solution;
  TSForcingFunction  forcing;

  PetscErrorCode (*destroy)(DMTS);
  PetscErrorCode (*duplicate)(DMTS,DMTS);
};

struct _p_DMTS {
  PETSCHEADER(struct _DMTSOps);
  void *rhsfunctionctx;
  void *rhsjacobianctx;

  void *ifunctionctx;
  void *ijacobianctx;

  void *i2functionctx;
  void *i2jacobianctx;

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
PETSC_EXTERN PetscErrorCode DMTSCopy(DMTS,DMTS);

typedef enum {TSEVENT_NONE,TSEVENT_LOCATED_INTERVAL,TSEVENT_PROCESSING,TSEVENT_ZERO,TSEVENT_RESET_NEXTSTEP} TSEventStatus;

struct _n_TSEvent {
  PetscScalar    *fvalue;          /* value of event function at the end of the step*/
  PetscScalar    *fvalue_prev;     /* value of event function at start of the step (left end-point of event interval) */
  PetscReal       ptime_prev;      /* time at step start (left end-point of event interval) */
  PetscReal       ptime_end;       /* end time of step (when an event interval is detected, ptime_end is fixed to the time at step end during event processing) */
  PetscReal       ptime_right;     /* time on the right end-point of the event interval */
  PetscScalar    *fvalue_right;    /* value of event function at the right end-point of the event interval */
  PetscInt       *side;            /* Used for detecting repetition of end-point, -1 => left, +1 => right */
  PetscReal       timestep_prev;   /* previous time step */
  PetscReal       timestep_orig;   /* initial time step */
  PetscBool      *zerocrossing;    /* Flag to signal zero crossing detection */
  PetscErrorCode  (*eventhandler)(TS,PetscReal,Vec,PetscScalar*,void*); /* User event handler function */
  PetscErrorCode  (*postevent)(TS,PetscInt,PetscInt[],PetscReal,Vec,PetscBool,void*); /* User post event function */
  void           *ctx;              /* User context for event handler and post even functions */
  PetscInt       *direction;        /* Zero crossing direction: 1 -> Going positive, -1 -> Going negative, 0 -> Any */
  PetscBool      *terminate;        /* 1 -> Terminate time stepping, 0 -> continue */
  PetscInt        nevents;          /* Number of events to handle */
  PetscInt        nevents_zero;     /* Number of event zero detected */
  PetscInt       *events_zero;      /* List of events that have reached zero */
  PetscReal      *vtol;             /* Vector tolerances for event zero check */
  TSEventStatus   status;           /* Event status */
  PetscInt        iterctr;          /* Iteration counter */
  PetscViewer     monitor;
  /* Struct to record the events */
  struct {
    PetscInt  ctr;        /* recorder counter */
    PetscReal *time;      /* Event times */
    PetscInt  *stepnum;   /* Step numbers */
    PetscInt  *nevents;   /* Number of events occuring at the event times */
    PetscInt  **eventidx; /* Local indices of the events in the event list */
  } recorder;
  PetscInt  recsize; /* Size of recorder stack */
};

PETSC_EXTERN PetscErrorCode TSEventInitialize(TSEvent,TS,PetscReal,Vec);
PETSC_EXTERN PetscErrorCode TSEventDestroy(TSEvent*);
PETSC_EXTERN PetscErrorCode TSEventHandler(TS);
PETSC_EXTERN PetscErrorCode TSAdjointEventHandler(TS);

PETSC_EXTERN PetscLogEvent TS_AdjointStep, TS_Step, TS_PseudoComputeTimeStep, TS_FunctionEval, TS_JacobianEval;

typedef enum {TS_STEP_INCOMPLETE, /* vec_sol, ptime, etc point to beginning of step */
              TS_STEP_PENDING,    /* vec_sol advanced, but step has not been accepted yet */
              TS_STEP_COMPLETE    /* step accepted and ptime, steps, etc have been advanced */
} TSStepStatus;

struct _n_TSMonitorLGCtx {
  PetscDrawLG    lg;
  PetscInt       howoften;  /* when > 0 uses step % howoften, when negative only final solution plotted */
  PetscInt       ksp_its,snes_its;
  char           **names;
  char           **displaynames;
  PetscInt       ndisplayvariables;
  PetscInt       *displayvariables;
  PetscReal      *displayvalues;
  PetscErrorCode (*transform)(void*,Vec,Vec*);
  PetscErrorCode (*transformdestroy)(void*);
  void           *transformctx;
};

struct _n_TSMonitorEnvelopeCtx {
  Vec max,min;
};

PETSC_EXTERN PetscLogEvent TSTrajectory_Set, TSTrajectory_Get, TSTrajectory_DiskWrite, TSTrajectory_DiskRead;

#endif

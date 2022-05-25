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
PETSC_EXTERN PetscErrorCode TSMPRKRegisterAll(void);
PETSC_EXTERN PetscErrorCode TSARKIMEXRegisterAll(void);
PETSC_EXTERN PetscErrorCode TSRosWRegisterAll(void);
PETSC_EXTERN PetscErrorCode TSGLLERegisterAll(void);
PETSC_EXTERN PetscErrorCode TSGLLEAdaptRegisterAll(void);
PETSC_EXTERN PetscErrorCode TSIRKRegisterAll(void);

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
  PetscErrorCode (*getstages)(TS,PetscInt*,Vec*[]);
  PetscErrorCode (*adjointstep)(TS);
  PetscErrorCode (*adjointsetup)(TS);
  PetscErrorCode (*adjointreset)(TS);
  PetscErrorCode (*adjointintegral)(TS);
  PetscErrorCode (*forwardsetup)(TS);
  PetscErrorCode (*forwardreset)(TS);
  PetscErrorCode (*forwardstep)(TS);
  PetscErrorCode (*forwardintegral)(TS);
  PetscErrorCode (*forwardgetstages)(TS,PetscInt*,Mat*[]);
  PetscErrorCode (*getsolutioncomponents)(TS,PetscInt*,Vec*);
  PetscErrorCode (*getauxsolution)(TS,Vec*);
  PetscErrorCode (*gettimeerror)(TS,PetscInt,Vec*);
  PetscErrorCode (*settimeerror)(TS,Vec);
  PetscErrorCode (*startingmethod) (TS);
  PetscErrorCode (*initcondition)(TS,Vec);
  PetscErrorCode (*exacterror)(TS,Vec,Vec);
};

/*
   TSEvent - Abstract object to handle event monitoring
*/
typedef struct _n_TSEvent *TSEvent;

typedef struct _TSTrajectoryOps *TSTrajectoryOps;

struct _TSTrajectoryOps {
  PetscErrorCode (*view)(TSTrajectory,PetscViewer);
  PetscErrorCode (*reset)(TSTrajectory);
  PetscErrorCode (*destroy)(TSTrajectory);
  PetscErrorCode (*set)(TSTrajectory,TS,PetscInt,PetscReal,Vec);
  PetscErrorCode (*get)(TSTrajectory,TS,PetscInt,PetscReal*);
  PetscErrorCode (*setfromoptions)(PetscOptionItems*,TSTrajectory);
  PetscErrorCode (*setup)(TSTrajectory,TS);
};

/* TSHistory is an helper object that allows inquiring
   the TSTrajectory by time and not by the step number only */
typedef struct _n_TSHistory* TSHistory;

struct _p_TSTrajectory {
  PETSCHEADER(struct _TSTrajectoryOps);
  TSHistory tsh;        /* associates times to unique step ids */
  /* stores necessary data to reconstruct states and derivatives via Lagrangian interpolation */
  struct {
    PetscInt    order;  /* interpolation order */
    Vec         *W;     /* work vectors */
    PetscScalar *L;     /* workspace for Lagrange basis */
    PetscReal   *T;     /* Lagrange times (stored) */
    Vec         *WW;    /* just an array of pointers */
    PetscBool   *TT;    /* workspace for Lagrange */
    PetscReal   *TW;    /* Lagrange times (workspace) */

    /* caching */
    PetscBool caching;
    struct {
      PetscObjectId    id;
      PetscObjectState state;
      PetscReal        time;
      PetscInt         step;
    } Ucached;
    struct {
      PetscObjectId    id;
      PetscObjectState state;
      PetscReal        time;
      PetscInt         step;
    } Udotcached;
  } lag;
  Vec            U,Udot;                  /* used by TSTrajectory{Get|Restore}UpdatedHistoryVecs */
  PetscBool      usehistory;              /* whether to use TSHistory */
  PetscBool      solution_only;           /* whether we dump just the solution or also the stages */
  PetscBool      adjoint_solve_mode;      /* whether we will use the Trajectory inside a TSAdjointSolve() or not */
  PetscViewer    monitor;
  PetscInt       setupcalled;             /* true if setup has been called */
  PetscInt       recomps;                 /* counter for recomputations in the adjoint run */
  PetscInt       diskreads,diskwrites;    /* counters for disk checkpoint reads and writes */
  char           **names;                 /* the name of each variable; each process has only the local names */
  PetscBool      keepfiles;               /* keep the files generated during the run after the run is complete */
  char           *dirname,*filetemplate;  /* directory name and file name template for disk checkpoints */
  char           *dirfiletemplate;        /* complete directory and file name template for disk checkpoints */
  PetscErrorCode (*transform)(void*,Vec,Vec*);
  PetscErrorCode (*transformdestroy)(void*);
  void*          transformctx;
  void           *data;
};

typedef struct _TS_RHSSplitLink *TS_RHSSplitLink;
struct _TS_RHSSplitLink {
  TS              ts;
  char            *splitname;
  IS              is;
  TS_RHSSplitLink next;
  PetscLogEvent   event;
};

typedef struct _TS_TimeSpan *TSTimeSpan;
struct _TS_TimeSpan {
    PetscInt  num_span_times; /* number of time points */
    PetscReal *span_times;    /* array of the time span */
    PetscReal reltol;         /* relative tolerance for span point detection */
    PetscReal abstol;         /* absolute tolerance for span point detection */
    PetscInt  spanctr;        /* counter of the time points that have been reached */
    Vec       *vecs_sol;      /* array of the solutions at the specified time points */
};

struct _p_TS {
  PETSCHEADER(struct _TSOps);
  TSProblemType  problem_type;
  TSEquationType equation_type;

  DM             dm;
  Vec            vec_sol; /* solution vector in first and second order equations */
  Vec            vec_dot; /* time derivative vector in second order equations */
  TSAdapt        adapt;
  TSAdaptType    default_adapt_type;
  TSEvent        event;

  /* ---------------- User (or PETSc) Provided stuff ---------------------*/
  PetscErrorCode (*monitor[MAXTSMONITORS])(TS,PetscInt,PetscReal,Vec,void*);
  PetscErrorCode (*monitordestroy[MAXTSMONITORS])(void**);
  void            *monitorcontext[MAXTSMONITORS];
  PetscInt         numbermonitors;
  PetscErrorCode (*adjointmonitor[MAXTSMONITORS])(TS,PetscInt,PetscReal,Vec,PetscInt,Vec*,Vec*,void*);
  PetscErrorCode (*adjointmonitordestroy[MAXTSMONITORS])(void**);
  void            *adjointmonitorcontext[MAXTSMONITORS];
  PetscInt         numberadjointmonitors;
  PetscInt         monitorFrequency; /* Number of timesteps between monitor output */

  PetscErrorCode (*prestep)(TS);
  PetscErrorCode (*prestage)(TS,PetscReal);
  PetscErrorCode (*poststage)(TS,PetscReal,PetscInt,Vec*);
  PetscErrorCode (*postevaluate)(TS);
  PetscErrorCode (*poststep)(TS);
  PetscErrorCode (*functiondomainerror)(TS,PetscReal,Vec,PetscBool*);

  /* ---------------------- Sensitivity Analysis support -----------------*/
  TSTrajectory trajectory;          /* All solutions are kept here for the entire time integration process */
  Vec       *vecs_sensi;            /* one vector for each cost function */
  Vec       *vecs_sensip;
  PetscInt  numcost;                /* number of cost functions */
  Vec       vec_costintegral;
  PetscInt  adjointsetupcalled;
  PetscInt  adjoint_steps;
  PetscInt  adjoint_max_steps;
  PetscBool adjoint_solve;          /* immediately call TSAdjointSolve() after TSSolve() is complete */
  PetscBool costintegralfwd;        /* cost integral is evaluated in the forward run if true */
  Vec       vec_costintegrand;      /* workspace for Adjoint computations */
  Mat       Jacp,Jacprhs;
  void      *ijacobianpctx,*rhsjacobianpctx;
  void      *costintegrandctx;
  Vec       *vecs_drdu;
  Vec       *vecs_drdp;
  Vec       vec_drdu_col,vec_drdp_col;

  /* first-order adjoint */
  PetscErrorCode (*rhsjacobianp)(TS,PetscReal,Vec,Mat,void*);
  PetscErrorCode (*ijacobianp)(TS,PetscReal,Vec,Vec,PetscReal,Mat,void*);
  PetscErrorCode (*costintegrand)(TS,PetscReal,Vec,Vec,void*);
  PetscErrorCode (*drdufunction)(TS,PetscReal,Vec,Vec*,void*);
  PetscErrorCode (*drdpfunction)(TS,PetscReal,Vec,Vec*,void*);

  /* second-order adjoint */
  Vec *vecs_sensi2;
  Vec *vecs_sensi2p;
  Vec vec_dir; /* directional vector for optimization */
  Vec *vecs_fuu,*vecs_guu;
  Vec *vecs_fup,*vecs_gup;
  Vec *vecs_fpu,*vecs_gpu;
  Vec *vecs_fpp,*vecs_gpp;
  void *ihessianproductctx,*rhshessianproductctx;
  PetscErrorCode (*ihessianproduct_fuu)(TS,PetscReal,Vec,Vec*,Vec,Vec*,void*);
  PetscErrorCode (*ihessianproduct_fup)(TS,PetscReal,Vec,Vec*,Vec,Vec*,void*);
  PetscErrorCode (*ihessianproduct_fpu)(TS,PetscReal,Vec,Vec*,Vec,Vec*,void*);
  PetscErrorCode (*ihessianproduct_fpp)(TS,PetscReal,Vec,Vec*,Vec,Vec*,void*);
  PetscErrorCode (*rhshessianproduct_guu)(TS,PetscReal,Vec,Vec*,Vec,Vec*,void*);
  PetscErrorCode (*rhshessianproduct_gup)(TS,PetscReal,Vec,Vec*,Vec,Vec*,void*);
  PetscErrorCode (*rhshessianproduct_gpu)(TS,PetscReal,Vec,Vec*,Vec,Vec*,void*);
  PetscErrorCode (*rhshessianproduct_gpp)(TS,PetscReal,Vec,Vec*,Vec,Vec*,void*);

  /* specific to forward sensitivity analysis */
  Mat       mat_sensip;              /* matrix storing forward sensitivities */
  Vec       vec_sensip_col;          /* space for a column of the sensip matrix */
  Vec       *vecs_integral_sensip;   /* one vector for each integral */
  PetscInt  num_parameters;
  PetscInt  num_initialvalues;
  void      *vecsrhsjacobianpctx;
  PetscInt  forwardsetupcalled;
  PetscBool forward_solve;
  PetscErrorCode (*vecsrhsjacobianp)(TS,PetscReal,Vec,Vec*,void*);

  /* ---------------------- IMEX support ---------------------------------*/
  /* These extra slots are only used when the user provides both Implicit and RHS */
  Mat Arhs;     /* Right hand side matrix */
  Mat Brhs;     /* Right hand side preconditioning matrix */
  Vec Frhs;     /* Right hand side function value */

  /* This is a general caching scheme to avoid recomputing the Jacobian at a place that has been previously been evaluated.
   * The present use case is that TSComputeRHSFunctionLinear() evaluates the Jacobian once and we don't want it to be immeditely re-evaluated.
   */
  struct {
    PetscReal        time;          /* The time at which the matrices were last evaluated */
    PetscObjectId    Xid;           /* Unique ID of solution vector at which the Jacobian was last evaluated */
    PetscObjectState Xstate;        /* State of the solution vector */
    MatStructure     mstructure;    /* The structure returned */
    /* Flag to unshift Jacobian before calling the IJacobian or RHSJacobian functions.  This is useful
     * if the user would like to reuse (part of) the Jacobian from the last evaluation. */
    PetscBool        reuse;
    PetscReal        scale,shift;
  } rhsjacobian;

  struct {
    PetscReal shift;            /* The derivative of the lhs wrt to Xdot */
  } ijacobian;

  MatStructure  axpy_pattern;  /* information about the nonzero pattern of the RHS Jacobian in reference to the implicit Jacobian */
  /* --------------------Nonlinear Iteration------------------------------*/
  SNES     snes;
  PetscBool usessnes;   /* Flag set by each TSType to indicate if the type actually uses a SNES;
                           this works around the design flaw that a SNES is ALWAYS created with TS even when it is not needed.*/
  PetscInt ksp_its;                /* total number of linear solver iterations */
  PetscInt snes_its;               /* total number of nonlinear solver iterations */
  PetscInt num_snes_failures;
  PetscInt max_snes_failures;

  /* --- Logging --- */
  PetscInt ifuncs,rhsfuncs,ijacs,rhsjacs;

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
  PetscInt  steps;                  /* steps taken so far in all successive calls to TSSolve() */
  PetscReal ptime;                  /* time at the start of the current step (stage time is internal if it exists) */
  PetscReal time_step;              /* current time increment */
  PetscReal ptime_prev;             /* time at the start of the previous step */
  PetscReal ptime_prev_rollback;    /* time at the start of the 2nd previous step to recover from rollback */
  PetscReal solvetime;              /* time at the conclusion of TSSolve() */
  PetscBool stifflyaccurate;        /* flag to indicate that the method is stiffly accurate */

  TSConvergedReason reason;
  PetscBool errorifstepfailed;
  PetscInt  reject,max_reject;
  TSExactFinalTimeOption exact_final_time;

  PetscReal atol,rtol;              /* Relative and absolute tolerance for local truncation error */
  Vec       vatol,vrtol;            /* Relative and absolute tolerance in vector form */
  PetscReal cfltime,cfltime_local;

  PetscBool testjacobian;
  PetscBool testjacobiantranspose;
  /* ------------------- Default work-area management ------------------ */
  PetscInt nwork;
  Vec      *work;

  /* ---------------------- RHS splitting support ---------------------------------*/
  PetscInt        num_rhs_splits;
  TS_RHSSplitLink tsrhssplit;
  PetscBool       use_splitrhsfunction;

  /* ---------------------- Quadrature integration support ---------------------------------*/
  TS quadraturets;

  /* ---------------------- Time span support ---------------------------------*/
  TSTimeSpan tspan;
};

struct _TSAdaptOps {
  PetscErrorCode (*choose)(TSAdapt,TS,PetscReal,PetscInt*,PetscReal*,PetscBool*,PetscReal*,PetscReal*,PetscReal*);
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
  PetscBool   always_accept;
  PetscReal   safety;             /* safety factor relative to target error/stability goal */
  PetscReal   reject_safety;      /* extra safety factor if the last step was rejected */
  PetscReal   clip[2];            /* admissible time step decrease/increase factors */
  PetscReal   dt_min,dt_max;      /* admissible minimum and maximum time step */
  PetscReal   ignore_max;         /* minimum value of the solution to be considered by the adaptor */
  PetscBool   glee_use_local;     /* GLEE adaptor uses global or local error */
  PetscReal   scale_solve_failed; /* scale step by this factor if solver (linear or nonlinear) fails. */
  PetscReal   matchstepfac[2];    /* factors to control the behaviour of matchstep */
  NormType    wnormtype;
  PetscViewer monitor;
  PetscInt    timestepjustdecreased_delay; /* number of timesteps after a decrease in the timestep before the timestep can be increased */
  PetscInt    timestepjustdecreased;
  PetscReal   dt_span_cached;    /* time step before hitting a TS span time point */
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

  TSTransientVariable transientvar;

  TSSolutionFunction solution;
  TSForcingFunction  forcing;

  PetscErrorCode (*destroy)(DMTS);
  PetscErrorCode (*duplicate)(DMTS,DMTS);
};

struct _p_DMTS {
  PETSCHEADER(struct _DMTSOps);
  PetscContainer rhsfunctionctxcontainer;
  PetscContainer rhsjacobianctxcontainer;

  PetscContainer ifunctionctxcontainer;
  PetscContainer ijacobianctxcontainer;

  PetscContainer i2functionctxcontainer;
  PetscContainer i2jacobianctxcontainer;

  void *transientvarctx;

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

PETSC_EXTERN PetscErrorCode DMTSUnsetRHSFunctionContext_Internal(DM);
PETSC_EXTERN PetscErrorCode DMTSUnsetRHSJacobianContext_Internal(DM);
PETSC_EXTERN PetscErrorCode DMTSUnsetIFunctionContext_Internal(DM);
PETSC_EXTERN PetscErrorCode DMTSUnsetIJacobianContext_Internal(DM);
PETSC_EXTERN PetscErrorCode DMTSUnsetI2FunctionContext_Internal(DM);
PETSC_EXTERN PetscErrorCode DMTSUnsetI2JacobianContext_Internal(DM);

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
  PetscReal       timestep_posteventinterval;  /* time step immediately after the event interval */
  PetscReal       timestep_postevent;  /* time step immediately after the event */
  PetscReal       timestep_min;    /* Minimum time step */
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
    PetscInt  *nevents;   /* Number of events occurring at the event times */
    PetscInt  **eventidx; /* Local indices of the events in the event list */
  } recorder;
  PetscInt  recsize; /* Size of recorder stack */
  PetscInt  refct; /* reference count */
};

PETSC_EXTERN PetscErrorCode TSEventInitialize(TSEvent,TS,PetscReal,Vec);
PETSC_EXTERN PetscErrorCode TSEventDestroy(TSEvent*);
PETSC_EXTERN PetscErrorCode TSEventHandler(TS);
PETSC_EXTERN PetscErrorCode TSAdjointEventHandler(TS);

PETSC_EXTERN PetscLogEvent TS_AdjointStep;
PETSC_EXTERN PetscLogEvent TS_Step;
PETSC_EXTERN PetscLogEvent TS_PseudoComputeTimeStep;
PETSC_EXTERN PetscLogEvent TS_FunctionEval;
PETSC_EXTERN PetscLogEvent TS_JacobianEval;
PETSC_EXTERN PetscLogEvent TS_ForwardStep;

typedef enum {TS_STEP_INCOMPLETE, /* vec_sol, ptime, etc point to beginning of step */
              TS_STEP_PENDING,    /* vec_sol advanced, but step has not been accepted yet */
              TS_STEP_COMPLETE    /* step accepted and ptime, steps, etc have been advanced */
} TSStepStatus;

struct _n_TSMonitorLGCtx {
  PetscDrawLG    lg;
  PetscBool      semilogy;
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

struct _n_TSMonitorSPCtx{
  PetscDrawSP sp;
  PetscInt    howoften; /* when > 0 uses step % howoften, when negative only final solution plotted */
  PetscInt    retain;   /* Retain n points plotted to show trajectories, or -1 for all points */
  PetscBool   phase;    /* Plot in phase space rather than coordinate space */
  PetscInt    ksp_its, snes_its;
};

struct _n_TSMonitorEnvelopeCtx {
  Vec max,min;
};

/*
    Checks if the user provide a TSSetIFunction() but an explicit method is called; generate an error in that case
*/
static inline PetscErrorCode TSCheckImplicitTerm(TS ts)
{
  TSIFunction ifunction;
  DM          dm;

  PetscFunctionBegin;
  PetscCall(TSGetDM(ts,&dm));
  PetscCall(DMTSGetIFunction(dm,&ifunction,NULL));
  PetscCheck(!ifunction,PetscObjectComm((PetscObject)ts),PETSC_ERR_ARG_INCOMP,"You are attempting to use an explicit ODE integrator but provided an implicit function definition with TSSetIFunction()");
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode TSGetRHSMats_Private(TS,Mat*,Mat*);
/* this is declared here as TSHistory is not public */
PETSC_EXTERN PetscErrorCode TSAdaptHistorySetTSHistory(TSAdapt,TSHistory,PetscBool);

PETSC_INTERN PetscErrorCode TSTrajectoryReconstruct_Private(TSTrajectory,TS,PetscReal,Vec,Vec);
PETSC_INTERN PetscErrorCode TSTrajectorySetUp_Basic(TSTrajectory,TS);

PETSC_EXTERN PetscLogEvent TSTrajectory_Set;
PETSC_EXTERN PetscLogEvent TSTrajectory_Get;
PETSC_EXTERN PetscLogEvent TSTrajectory_GetVecs;
PETSC_EXTERN PetscLogEvent TSTrajectory_SetUp;
PETSC_EXTERN PetscLogEvent TSTrajectory_DiskWrite;
PETSC_EXTERN PetscLogEvent TSTrajectory_DiskRead;

struct _n_TSMonitorDrawCtx {
  PetscViewer   viewer;
  Vec           initialsolution;
  PetscBool     showinitial;
  PetscInt      howoften;  /* when > 0 uses step % howoften, when negative only final solution plotted */
  PetscBool     showtimestepandtime;
};
#endif

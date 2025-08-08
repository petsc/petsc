#pragma once

#include <petscts.h>
#include <petsc/private/petscimpl.h>

/* SUBMANSEC = TS */

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

PETSC_EXTERN PetscBool      TSRegisterAllCalled;
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
  PetscErrorCode (*snesfunction)(SNES, Vec, Vec, TS);
  PetscErrorCode (*snesjacobian)(SNES, Vec, Mat, Mat, TS);
  PetscErrorCode (*setup)(TS);
  PetscErrorCode (*step)(TS);
  PetscErrorCode (*solve)(TS);
  PetscErrorCode (*interpolate)(TS, PetscReal, Vec);
  PetscErrorCode (*evaluatewlte)(TS, NormType, PetscInt *, PetscReal *);
  PetscErrorCode (*evaluatestep)(TS, PetscInt, Vec, PetscBool *);
  PetscErrorCode (*setfromoptions)(TS, PetscOptionItems);
  PetscErrorCode (*destroy)(TS);
  PetscErrorCode (*view)(TS, PetscViewer);
  PetscErrorCode (*reset)(TS);
  PetscErrorCode (*linearstability)(TS, PetscReal, PetscReal, PetscReal *, PetscReal *);
  PetscErrorCode (*load)(TS, PetscViewer);
  PetscErrorCode (*rollback)(TS);
  PetscErrorCode (*getstages)(TS, PetscInt *, Vec *[]);
  PetscErrorCode (*adjointstep)(TS);
  PetscErrorCode (*adjointsetup)(TS);
  PetscErrorCode (*adjointreset)(TS);
  PetscErrorCode (*adjointintegral)(TS);
  PetscErrorCode (*forwardsetup)(TS);
  PetscErrorCode (*forwardreset)(TS);
  PetscErrorCode (*forwardstep)(TS);
  PetscErrorCode (*forwardintegral)(TS);
  PetscErrorCode (*forwardgetstages)(TS, PetscInt *, Mat *[]);
  PetscErrorCode (*getsolutioncomponents)(TS, PetscInt *, Vec *);
  PetscErrorCode (*getauxsolution)(TS, Vec *);
  PetscErrorCode (*gettimeerror)(TS, PetscInt, Vec *);
  PetscErrorCode (*settimeerror)(TS, Vec);
  PetscErrorCode (*startingmethod)(TS);
  PetscErrorCode (*initcondition)(TS, Vec);
  PetscErrorCode (*exacterror)(TS, Vec, Vec);
  PetscErrorCode (*resizeregister)(TS, PetscBool);
};

/*
   TSEvent - Abstract object to handle event monitoring
*/
typedef struct _n_TSEvent *TSEvent;

typedef struct _TSTrajectoryOps *TSTrajectoryOps;

struct _TSTrajectoryOps {
  PetscErrorCode (*view)(TSTrajectory, PetscViewer);
  PetscErrorCode (*reset)(TSTrajectory);
  PetscErrorCode (*destroy)(TSTrajectory);
  PetscErrorCode (*set)(TSTrajectory, TS, PetscInt, PetscReal, Vec);
  PetscErrorCode (*get)(TSTrajectory, TS, PetscInt, PetscReal *);
  PetscErrorCode (*setfromoptions)(TSTrajectory, PetscOptionItems);
  PetscErrorCode (*setup)(TSTrajectory, TS);
};

/* TSHistory is an helper object that allows inquiring
   the TSTrajectory by time and not by the step number only */
typedef struct _n_TSHistory *TSHistory;

struct _p_TSTrajectory {
  PETSCHEADER(struct _TSTrajectoryOps);
  TSHistory tsh; /* associates times to unique step ids */
  /* stores necessary data to reconstruct states and derivatives via Lagrangian interpolation */
  struct {
    PetscInt     order; /* interpolation order */
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
  Vec         U, Udot;            /* used by TSTrajectory{Get|Restore}UpdatedHistoryVecs */
  PetscBool   usehistory;         /* whether to use TSHistory */
  PetscBool   solution_only;      /* whether we dump just the solution or also the stages */
  PetscBool   adjoint_solve_mode; /* whether we will use the Trajectory inside a TSAdjointSolve() or not */
  PetscViewer monitor;
  PetscBool   setupcalled;            /* true if setup has been called */
  PetscInt    recomps;                /* counter for recomputations in the adjoint run */
  PetscInt    diskreads, diskwrites;  /* counters for disk checkpoint reads and writes */
  char      **names;                  /* the name of each variable; each process has only the local names */
  PetscBool   keepfiles;              /* keep the files generated during the run after the run is complete */
  char       *dirname, *filetemplate; /* directory name and file name template for disk checkpoints */
  char       *dirfiletemplate;        /* complete directory and file name template for disk checkpoints */
  PetscErrorCode (*transform)(void *, Vec, Vec *);
  PetscErrorCode (*transformdestroy)(void *);
  void *transformctx;
  void *data;
};

typedef struct _TS_RHSSplitLink *TS_RHSSplitLink;
struct _TS_RHSSplitLink {
  TS              ts;
  char           *splitname;
  IS              is;
  TS_RHSSplitLink next;
  PetscLogEvent   event;
};

typedef struct _TS_EvaluationTimes *TSEvaluationTimes;
struct _TS_EvaluationTimes {
  PetscInt   num_time_points; /* number of time points */
  PetscReal *time_points;     /* array of the time span */
  PetscReal  reltol;          /* relative tolerance for span point detection */
  PetscReal  abstol;          /* absolute tolerance for span point detection */
  PetscReal  worktol;         /* the ultimate tolerance (variable), maintained within a single TS time step for consistency */
  PetscInt   time_point_idx;  /* index of the time_point to be reached next */
  PetscInt   sol_idx;         /* index into sol_vecs and sol_times */
  Vec       *sol_vecs;        /* array of the solutions at the specified time points */
  PetscReal *sol_times;       /* array of times that sol_vecs was taken at */
};

struct _p_TS {
  PETSCHEADER(struct _TSOps);
  TSProblemType  problem_type;
  TSEquationType equation_type;

  DM          dm;
  Vec         vec_sol;  /* solution vector in first and second order equations */
  Vec         vec_sol0; /* solution vector at the beginning of the step */
  Vec         vec_dot;  /* time derivative vector in second order equations */
  TSAdapt     adapt;
  TSAdaptType default_adapt_type;
  TSEvent     event;

  /* ---------------- Resize ---------------------*/
  PetscBool       resizerollback;
  PetscObjectList resizetransferobjs;

  /* ---------------- User (or PETSc) Provided stuff ---------------------*/
  PetscErrorCode (*monitor[MAXTSMONITORS])(TS, PetscInt, PetscReal, Vec, void *);
  PetscCtxDestroyFn *monitordestroy[MAXTSMONITORS];
  void              *monitorcontext[MAXTSMONITORS];
  PetscInt           numbermonitors;
  PetscErrorCode (*adjointmonitor[MAXTSMONITORS])(TS, PetscInt, PetscReal, Vec, PetscInt, Vec *, Vec *, void *);
  PetscCtxDestroyFn *adjointmonitordestroy[MAXTSMONITORS];
  void              *adjointmonitorcontext[MAXTSMONITORS];
  PetscInt           numberadjointmonitors;

  PetscErrorCode (*prestep)(TS);
  PetscErrorCode (*prestage)(TS, PetscReal);
  PetscErrorCode (*poststage)(TS, PetscReal, PetscInt, Vec *);
  PetscErrorCode (*postevaluate)(TS);
  PetscErrorCode (*poststep)(TS);
  PetscErrorCode (*functiondomainerror)(TS, PetscReal, Vec, PetscBool *);
  PetscErrorCode (*resizesetup)(TS, PetscInt, PetscReal, Vec, PetscBool *, void *);
  PetscErrorCode (*resizetransfer)(TS, PetscInt, Vec[], Vec[], void *);
  void *resizectx;

  /* ---------------------- Sensitivity Analysis support -----------------*/
  TSTrajectory trajectory; /* All solutions are kept here for the entire time integration process */
  Vec         *vecs_sensi; /* one vector for each cost function */
  Vec         *vecs_sensip;
  PetscInt     numcost; /* number of cost functions */
  Vec          vec_costintegral;
  PetscBool    adjointsetupcalled;
  PetscInt     adjoint_steps;
  PetscInt     adjoint_max_steps;
  PetscBool    adjoint_solve;     /* immediately call TSAdjointSolve() after TSSolve() is complete */
  PetscBool    costintegralfwd;   /* cost integral is evaluated in the forward run if true */
  Vec          vec_costintegrand; /* workspace for Adjoint computations */
  Mat          Jacp, Jacprhs;
  void        *ijacobianpctx, *rhsjacobianpctx;
  void        *costintegrandctx;
  Vec         *vecs_drdu;
  Vec         *vecs_drdp;
  Vec          vec_drdu_col, vec_drdp_col;

  /* first-order adjoint */
  PetscErrorCode (*rhsjacobianp)(TS, PetscReal, Vec, Mat, void *);
  PetscErrorCode (*ijacobianp)(TS, PetscReal, Vec, Vec, PetscReal, Mat, void *);
  PetscErrorCode (*costintegrand)(TS, PetscReal, Vec, Vec, void *);
  PetscErrorCode (*drdufunction)(TS, PetscReal, Vec, Vec *, void *);
  PetscErrorCode (*drdpfunction)(TS, PetscReal, Vec, Vec *, void *);

  /* second-order adjoint */
  Vec  *vecs_sensi2;
  Vec  *vecs_sensi2p;
  Vec   vec_dir; /* directional vector for optimization */
  Vec  *vecs_fuu, *vecs_guu;
  Vec  *vecs_fup, *vecs_gup;
  Vec  *vecs_fpu, *vecs_gpu;
  Vec  *vecs_fpp, *vecs_gpp;
  void *ihessianproductctx, *rhshessianproductctx;
  PetscErrorCode (*ihessianproduct_fuu)(TS, PetscReal, Vec, Vec *, Vec, Vec *, void *);
  PetscErrorCode (*ihessianproduct_fup)(TS, PetscReal, Vec, Vec *, Vec, Vec *, void *);
  PetscErrorCode (*ihessianproduct_fpu)(TS, PetscReal, Vec, Vec *, Vec, Vec *, void *);
  PetscErrorCode (*ihessianproduct_fpp)(TS, PetscReal, Vec, Vec *, Vec, Vec *, void *);
  PetscErrorCode (*rhshessianproduct_guu)(TS, PetscReal, Vec, Vec *, Vec, Vec *, void *);
  PetscErrorCode (*rhshessianproduct_gup)(TS, PetscReal, Vec, Vec *, Vec, Vec *, void *);
  PetscErrorCode (*rhshessianproduct_gpu)(TS, PetscReal, Vec, Vec *, Vec, Vec *, void *);
  PetscErrorCode (*rhshessianproduct_gpp)(TS, PetscReal, Vec, Vec *, Vec, Vec *, void *);

  /* specific to forward sensitivity analysis */
  Mat       mat_sensip;           /* matrix storing forward sensitivities */
  Vec       vec_sensip_col;       /* space for a column of the sensip matrix */
  Vec      *vecs_integral_sensip; /* one vector for each integral */
  PetscInt  num_parameters;
  PetscInt  num_initialvalues;
  void     *vecsrhsjacobianpctx;
  PetscBool forwardsetupcalled;
  PetscBool forward_solve;
  PetscErrorCode (*vecsrhsjacobianp)(TS, PetscReal, Vec, Vec *, void *);

  /* ---------------------- IMEX support ---------------------------------*/
  /* These extra slots are only used when the user provides both Implicit and RHS */
  Mat Arhs; /* Right hand side matrix */
  Mat Brhs; /* Right hand side matrix used to construct the preconditioner */
  Vec Frhs; /* Right hand side function value */

  /* This is a general caching scheme to avoid recomputing the Jacobian at a place that has been previously been evaluated.
   * The present use case is that TSComputeRHSFunctionLinear() evaluates the Jacobian once and we don't want it to be immeditely re-evaluated.
   */
  struct {
    PetscReal        time;       /* The time at which the matrices were last evaluated */
    PetscObjectId    Xid;        /* Unique ID of solution vector at which the Jacobian was last evaluated */
    PetscObjectState Xstate;     /* State of the solution vector */
    MatStructure     mstructure; /* The structure returned */
    /* Flag to unshift Jacobian before calling the IJacobian or RHSJacobian functions.  This is useful
     * if the user would like to reuse (part of) the Jacobian from the last evaluation. */
    PetscBool reuse;
    PetscReal scale, shift;
  } rhsjacobian;

  struct {
    PetscReal shift; /* The derivative of the lhs wrt to Xdot */
  } ijacobian;

  MatStructure axpy_pattern; /* information about the nonzero pattern of the RHS Jacobian in reference to the implicit Jacobian */
  /* --------------------Nonlinear Iteration------------------------------*/
  SNES      snes;
  PetscBool usessnes; /* Flag set by each TSType to indicate if the type actually uses a SNES;
                           this works around the design flaw that a SNES is ALWAYS created with TS even when it is not needed.*/
  PetscInt  ksp_its;  /* total number of linear solver iterations */
  PetscInt  snes_its; /* total number of nonlinear solver iterations */
  PetscInt  num_snes_failures;
  PetscInt  max_snes_failures;

  /* --- Logging --- */
  PetscInt ifuncs, rhsfuncs, ijacs, rhsjacs;

  /* --- Data that is unique to each particular solver --- */
  PetscBool setupcalled; /* true if setup has been called */
  void     *data;        /* implementationspecific data */
  void     *ctx;         /* user context */

  PetscBool steprollback;        /* flag to indicate that the step was rolled back */
  PetscBool steprestart;         /* flag to indicate that the timestepper has to discard any history and restart */
  PetscBool stepresize;          /* flag to indicate that the discretization was resized */
  PetscInt  steps;               /* steps taken so far in all successive calls to TSSolve() */
  PetscReal ptime;               /* time at the start of the current step (stage time is internal if it exists) */
  PetscReal time_step;           /* current time increment */
  PetscReal time_step0;          /* proposed time increment at the beginning of the step */
  PetscReal ptime_prev;          /* time at the start of the previous step */
  PetscReal ptime_prev_rollback; /* time at the start of the 2nd previous step to recover from rollback */
  PetscReal solvetime;           /* time at the conclusion of TSSolve() */
  PetscBool stifflyaccurate;     /* flag to indicate that the method is stiffly accurate */

  TSConvergedReason      reason;
  PetscBool              errorifstepfailed;
  PetscInt               reject, max_reject;
  TSExactFinalTimeOption exact_final_time;

  PetscObjectParameterDeclare(PetscReal, rtol); /* Relative and absolute tolerance for local truncation error */
  PetscObjectParameterDeclare(PetscReal, atol);
  PetscObjectParameterDeclare(PetscReal, max_time); /* max time allowed */
  PetscObjectParameterDeclare(PetscInt, max_steps); /* maximum time-step number to execute until (possibly with nonzero starting value) */
  PetscObjectParameterDeclare(PetscInt, run_steps); /* maximum number of time steps for TSSolve to take on each call */
  Vec       vatol, vrtol;                           /* Relative and absolute tolerance in vector form */
  PetscReal cfltime, cfltime_local;
  PetscInt  start_step; /* step number at start of current run */

  PetscBool testjacobian;
  PetscBool testjacobiantranspose;
  /* ------------------- Default work-area management ------------------ */
  PetscInt nwork;
  Vec     *work;

  /* ---------------------- RHS splitting support ---------------------------------*/
  PetscInt        num_rhs_splits;
  TS_RHSSplitLink tsrhssplit;
  PetscBool       use_splitrhsfunction;
  SNES            snesrhssplit;

  /* ---------------------- Quadrature integration support ---------------------------------*/
  TS quadraturets;

  /* ---------------------- Time span support ---------------------------------*/
  TSEvaluationTimes eval_times;
};

struct _TSAdaptOps {
  PetscErrorCode (*choose)(TSAdapt, TS, PetscReal, PetscInt *, PetscReal *, PetscBool *, PetscReal *, PetscReal *, PetscReal *);
  PetscErrorCode (*destroy)(TSAdapt);
  PetscErrorCode (*reset)(TSAdapt);
  PetscErrorCode (*view)(TSAdapt, PetscViewer);
  PetscErrorCode (*setfromoptions)(TSAdapt, PetscOptionItems);
  PetscErrorCode (*load)(TSAdapt, PetscViewer);
};

struct _p_TSAdapt {
  PETSCHEADER(struct _TSAdaptOps);
  void *data;
  PetscErrorCode (*checkstage)(TSAdapt, TS, PetscReal, Vec, PetscBool *);
  struct {
    PetscInt    n;              /* number of candidate schemes, including the one currently in use */
    PetscBool   inuse_set;      /* the current scheme has been set */
    const char *name[16];       /* name of the scheme */
    PetscInt    order[16];      /* classical order of each scheme */
    PetscInt    stageorder[16]; /* stage order of each scheme */
    PetscReal   ccfl[16];       /* stability limit relative to explicit Euler */
    PetscReal   cost[16];       /* relative measure of the amount of work required for each scheme */
  } candidates;
  PetscBool   always_accept;
  PetscReal   safety;             /* safety factor relative to target error/stability goal */
  PetscReal   reject_safety;      /* extra safety factor if the last step was rejected */
  PetscReal   clip[2];            /* admissible time step decrease/increase factors */
  PetscReal   dt_min, dt_max;     /* admissible minimum and maximum time step */
  PetscReal   ignore_max;         /* minimum value of the solution to be considered by the adaptor */
  PetscBool   glee_use_local;     /* GLEE adaptor uses global or local error */
  PetscReal   scale_solve_failed; /* scale step by this factor if solver (linear or nonlinear) fails. */
  PetscReal   matchstepfac[2];    /* factors to control the behaviour of matchstep */
  NormType    wnormtype;
  PetscViewer monitor;
  PetscInt    timestepjustdecreased_delay; /* number of timesteps after a decrease in the timestep before the timestep can be increased */
  PetscInt    timestepjustdecreased;
  PetscReal   dt_eval_times_cached; /* time step before hitting a TS evaluation time point */
};

typedef struct _p_DMTS  *DMTS;
typedef struct _DMTSOps *DMTSOps;
struct _DMTSOps {
  TSRHSFunctionFn *rhsfunction;
  TSRHSJacobianFn *rhsjacobian;

  TSIFunctionFn *ifunction;
  PetscErrorCode (*ifunctionview)(void *, PetscViewer);
  PetscErrorCode (*ifunctionload)(void **, PetscViewer);

  TSIJacobianFn *ijacobian;
  PetscErrorCode (*ijacobianview)(void *, PetscViewer);
  PetscErrorCode (*ijacobianload)(void **, PetscViewer);

  TSI2FunctionFn *i2function;
  TSI2JacobianFn *i2jacobian;

  TSTransientVariableFn *transientvar;

  TSSolutionFn *solution;
  TSForcingFn  *forcing;

  PetscErrorCode (*destroy)(DMTS);
  PetscErrorCode (*duplicate)(DMTS, DMTS);
};

/*S
   DMTS - Object held by a `DM` that contains all the callback functions and their contexts needed by a `TS`

   Level: developer

   Notes:
   Users provide callback functions and their contexts to `TS` using, for example, `TSSetIFunction()`. These values are stored
   in a `DMTS` that is contained in the `DM` associated with the `TS`. If no `DM` was provided by
   the user with `TSSetDM()` it is automatically created by `TSGetDM()` with `DMShellCreate()`.

   Users very rarely need to worked directly with the `DMTS` object, rather they work with the `TS` and the `DM` they created

   Multiple `DM` can share a single `DMTS`, often each `DM` is associated with
   a grid refinement level. `DMGetDMTS()` returns the `DMTS` associated with a `DM`. `DMGetDMTSWrite()` returns a unique
   `DMTS` that is only associated with the current `DM`, making a copy of the shared `DMTS` if needed (copy-on-write).

   See `DMKSP` for details on why there is a needed for `DMTS` instead of simply storing the user callbacks directly in the `DM` or the `TS`

   Developer Note:
   The original `dm` inside the `DMTS` is NOT reference counted  (to prevent a reference count loop between a `DM` and a `DMSNES`).
   The `DM` on which this context was first created is cached here to implement one-way
   copy-on-write. When `DMGetDMTSWrite()` sees a request using a different `DM`, it makes a copy of the `DMTS`.

.seealso: [](ch_ts), `TSCreate()`, `DM`, `DMGetDMTSWrite()`, `DMGetDMTS()`, `TSSetIFunction()`, `DMTSSetRHSFunctionContextDestroy()`,
          `DMTSSetRHSJacobian()`, `DMTSGetRHSJacobian()`, `DMTSSetRHSJacobianContextDestroy()`, `DMTSSetIFunction()`, `DMTSGetIFunction()`,
          `DMTSSetIFunctionContextDestroy()`, `DMTSSetIJacobian()`, `DMTSGetIJacobian()`, `DMTSSetIJacobianContextDestroy()`,
          `DMTSSetI2Function()`, `DMTSGetI2Function()`, `DMTSSetI2FunctionContextDestroy()`, `DMTSSetI2Jacobian()`,
          `DMTSGetI2Jacobian()`, `DMTSSetI2JacobianContextDestroy()`, `DMKSP`, `DMSNES`
S*/
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

  /* See the developer note for DMTS above */
  DM originaldm;
};

PETSC_INTERN PetscErrorCode DMTSUnsetRHSFunctionContext_Internal(DM);
PETSC_INTERN PetscErrorCode DMTSUnsetRHSJacobianContext_Internal(DM);
PETSC_INTERN PetscErrorCode DMTSUnsetIFunctionContext_Internal(DM);
PETSC_INTERN PetscErrorCode DMTSUnsetIJacobianContext_Internal(DM);
PETSC_INTERN PetscErrorCode DMTSUnsetI2FunctionContext_Internal(DM);
PETSC_INTERN PetscErrorCode DMTSUnsetI2JacobianContext_Internal(DM);

PETSC_EXTERN PetscErrorCode DMGetDMTS(DM, DMTS *);
PETSC_EXTERN PetscErrorCode DMGetDMTSWrite(DM, DMTS *);
PETSC_EXTERN PetscErrorCode DMCopyDMTS(DM, DM);
PETSC_EXTERN PetscErrorCode DMTSView(DMTS, PetscViewer);
PETSC_EXTERN PetscErrorCode DMTSLoad(DMTS, PetscViewer);
PETSC_EXTERN PetscErrorCode DMTSCopy(DMTS, DMTS);

struct _n_TSEvent {
  PetscReal *fvalue_prev;                                                                   /* value of indicator function at the left end-point of the event interval */
  PetscReal *fvalue;                                                                        /* value of indicator function at the current point */
  PetscReal *fvalue_right;                                                                  /* value of indicator function at the right end-point of the event interval */
  PetscInt  *fsign_prev;                                                                    /* sign of indicator function at the left end-point of the event interval */
  PetscInt  *fsign;                                                                         /* sign of indicator function at the current point */
  PetscInt  *fsign_right;                                                                   /* sign of indicator function at the right end-point of the event interval */
  PetscReal  ptime_prev;                                                                    /* time at the previous point (left end-point of the event interval) */
  PetscReal  ptime_right;                                                                   /* time at the right end-point of the event interval */
  PetscReal  ptime_cache;                                                                   /* the point visited by the TS before the event interval was detected; cached - to reuse if necessary */
  PetscReal  timestep_cache;                                                                /* time step considered by the TS before the event interval was detected; cached - to reuse if necessary */
  PetscInt  *side;                                                                          /* upon bracket subdivision, indicates which sub-bracket is taken further, -1 -> left one, +1 -> right one, +2 -> neither, 0 -> zero-crossing located */
  PetscInt  *side_prev;                                                                     /* counts the repeating previous side's (with values: -n <=> '-1'*n; +n <=> '+1'*n); used in the Anderson-Bjorck iteration */
  PetscReal  timestep_postevent;                                                            /* first time step after the event; can be PETSC_DECIDE */
  PetscReal  timestep_2nd_postevent;                                                        /* second time step after the event; can be PETSC_DECIDE */
  PetscReal  timestep_min;                                                                  /* minimum time step */
  PetscBool *justrefined_AB;                                                                /* this flag shows if the given indicator function i = [0..nevents) participated in Anderson-Bjorck process in the last iteration of TSEventHandler() */
  PetscReal *gamma_AB;                                                                      /* cumulative scaling factor for the Anderson-Bjorck iteration */
  PetscErrorCode (*indicator)(TS, PetscReal, Vec, PetscReal *, void *);                     /* this callback defines the user function(s) whose sign changes indicate events */
  PetscErrorCode (*postevent)(TS, PetscInt, PetscInt[], PetscReal, Vec, PetscBool, void *); /* user post-event callback */
  void       *ctx;                                                                          /* user context for indicator and postevent callbacks */
  PetscInt   *direction;                                                                    /* zero crossing direction to trigger the event: +1 -> going positive, -1 -> going negative, 0 -> any */
  PetscBool  *terminate;                                                                    /* 1 -> terminate time stepping on event location, 0 -> continue */
  PetscInt    nevents;                                                                      /* number of events (indicator functions) to handle on the current MPI process */
  PetscInt    nevents_zero;                                                                 /* number of events triggered */
  PetscInt   *events_zero;                                                                  /* list of the events triggered */
  PetscReal  *vtol;                                                                         /* array of tolerances for the indicator function zero check */
  PetscInt    iterctr;                                                                      /* iteration counter: used both for reporting and as a status indicator */
  PetscBool   processing;                                                                   /* this flag shows if the event-resolving iterations are in progress, or the post-event dt handling is in progress */
  PetscBool   revisit_right;                                                                /* [sync] "revisit the bracket's right end", if true, then fvalue(s) are not calculated, but are taken from fvalue_right(s) */
  PetscViewer monitor;
  /* Struct to record the events */
  struct {
    PetscInt   ctr;      /* Recorder counter */
    PetscReal *time;     /* Event times */
    PetscInt  *stepnum;  /* Step numbers */
    PetscInt  *nevents;  /* Number of events occurring at the event times */
    PetscInt **eventidx; /* Local indices of the events in the event list */
  } recorder;
  PetscInt recsize; /* Size of recorder stack */
  PetscInt refct;   /* Reference count */
};

PETSC_EXTERN PetscErrorCode TSEventInitialize(TSEvent, TS, PetscReal, Vec);
PETSC_EXTERN PetscErrorCode TSEventDestroy(TSEvent *);
PETSC_EXTERN PetscErrorCode TSEventHandler(TS);
PETSC_EXTERN PetscErrorCode TSAdjointEventHandler(TS);

PETSC_EXTERN PetscLogEvent TS_AdjointStep;
PETSC_EXTERN PetscLogEvent TS_Step;
PETSC_EXTERN PetscLogEvent TS_PseudoComputeTimeStep;
PETSC_EXTERN PetscLogEvent TS_FunctionEval;
PETSC_EXTERN PetscLogEvent TS_JacobianEval;
PETSC_EXTERN PetscLogEvent TS_ForwardStep;

typedef enum {
  TS_STEP_INCOMPLETE, /* vec_sol, ptime, etc point to beginning of step */
  TS_STEP_PENDING,    /* vec_sol advanced, but step has not been accepted yet */
  TS_STEP_COMPLETE    /* step accepted and ptime, steps, etc have been advanced */
} TSStepStatus;

struct _n_TSMonitorLGCtx {
  PetscDrawLG lg;
  PetscBool   semilogy;
  PetscInt    howoften; /* when > 0 uses step % howoften, when negative only final solution plotted */
  PetscInt    ksp_its, snes_its;
  char      **names;
  char      **displaynames;
  PetscInt    ndisplayvariables;
  PetscInt   *displayvariables;
  PetscReal  *displayvalues;
  PetscErrorCode (*transform)(void *, Vec, Vec *);
  PetscCtxDestroyFn *transformdestroy;
  void              *transformctx;
};

struct _n_TSMonitorSPCtx {
  PetscDrawSP sp;
  PetscInt    howoften;     /* when > 0 uses step % howoften, when negative only final solution plotted */
  PetscInt    retain;       /* Retain n points plotted to show trajectories, or -1 for all points */
  PetscBool   phase;        /* Plot in phase space rather than coordinate space */
  PetscBool   multispecies; /* Change scatter point color based on species */
  PetscInt    ksp_its, snes_its;
};

struct _n_TSMonitorHGCtx {
  PetscDrawHG *hg;
  PetscInt     howoften; /* when > 0 uses step % howoften, when negative only final solution plotted */
  PetscInt     Ns;       /* The number of species to histogram */
  PetscBool    velocity; /* Plot in velocity space rather than coordinate space */
};

struct _n_TSMonitorEnvelopeCtx {
  Vec max, min;
};

/*
    Checks if the user provide a TSSetIFunction() but an explicit method is called; generate an error in that case
*/
static inline PetscErrorCode TSCheckImplicitTerm(TS ts)
{
  TSIFunctionFn *ifunction;
  DM             dm;

  PetscFunctionBegin;
  PetscCall(TSGetDM(ts, &dm));
  PetscCall(DMTSGetIFunction(dm, &ifunction, NULL));
  PetscCheck(!ifunction, PetscObjectComm((PetscObject)ts), PETSC_ERR_ARG_INCOMP, "You are attempting to use an explicit ODE integrator but provided an implicit function definition with TSSetIFunction()");
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode TSGetRHSMats_Private(TS, Mat *, Mat *);
/* this is declared here as TSHistory is not public */
PETSC_EXTERN PetscErrorCode TSAdaptHistorySetTSHistory(TSAdapt, TSHistory, PetscBool);

PETSC_INTERN PetscErrorCode TSTrajectoryReconstruct_Private(TSTrajectory, TS, PetscReal, Vec, Vec);
PETSC_INTERN PetscErrorCode TSTrajectorySetUp_Basic(TSTrajectory, TS);

PETSC_EXTERN PetscLogEvent TSTrajectory_Set;
PETSC_EXTERN PetscLogEvent TSTrajectory_Get;
PETSC_EXTERN PetscLogEvent TSTrajectory_GetVecs;
PETSC_EXTERN PetscLogEvent TSTrajectory_SetUp;
PETSC_EXTERN PetscLogEvent TSTrajectory_DiskWrite;
PETSC_EXTERN PetscLogEvent TSTrajectory_DiskRead;

struct _n_TSMonitorDrawCtx {
  PetscViewer viewer;
  Vec         initialsolution;
  PetscBool   showinitial;
  PetscInt    howoften; /* when > 0 uses step % howoften, when negative only final solution plotted */
  PetscBool   showtimestepandtime;
};

struct _n_TSMonitorVTKCtx {
  char    *filenametemplate;
  PetscInt interval; /* when > 0 uses step % interval, when negative only final solution plotted */
};

struct _n_TSMonitorSolutionCtx {
  PetscBool skip_initial; // Skip the viewer the first time TSMonitorSolution is run (within a single call to `TSSolve()`)
};

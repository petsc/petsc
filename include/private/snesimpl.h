
#ifndef __SNESIMPL_H
#define __SNESIMPL_H

#include <petscsnes.h>

typedef struct _SNESOps *SNESOps;

struct _SNESOps {
  PetscErrorCode (*computegs)(SNES,Vec,Vec,void*);
  PetscErrorCode (*computefunction)(SNES,Vec,Vec,void*);
  PetscErrorCode (*computejacobian)(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
  PetscErrorCode (*computeinitialguess)(SNES,Vec,void*);
  PetscErrorCode (*computescaling)(Vec,Vec,void*);
  PetscErrorCode (*update)(SNES, PetscInt);                     /* General purpose function for update */
  PetscErrorCode (*converged)(SNES,PetscInt,PetscReal,PetscReal,PetscReal,SNESConvergedReason*,void*);
  PetscErrorCode (*convergeddestroy)(void*);
  PetscErrorCode (*setup)(SNES);                                /* routine to set up the nonlinear solver */
  PetscErrorCode (*solve)(SNES);                                /* actual nonlinear solver */
  PetscErrorCode (*view)(SNES,PetscViewer);
  PetscErrorCode (*setfromoptions)(SNES);                       /* sets options from database */
  PetscErrorCode (*destroy)(SNES);
  PetscErrorCode (*reset)(SNES);
  PetscErrorCode (*usercompute)(SNES,void**);
  PetscErrorCode (*userdestroy)(void**);
  /* the line search to use */
  PetscErrorCode (*linesearch)         (SNES,void*,Vec,Vec,Vec,PetscReal,PetscReal,Vec,Vec,PetscReal*,PetscReal*,PetscBool *);
  /* optional functions for pre and postcheck on a linesearch */
  PetscErrorCode (*precheckstep)       (SNES,Vec,Vec,void*,PetscBool *);                  /* step-checking routine */
  PetscErrorCode (*postcheckstep)      (SNES,Vec,Vec,Vec,void*,PetscBool *,PetscBool *);  /* step-checking routine */
  PetscErrorCode (*computevariablebounds)(SNES,Vec,Vec);        /* user provided routine to set box constrained variable bounds */
  PetscErrorCode (*computepfunction)(SNES,Vec,Vec,void*);
  PetscErrorCode (*computepjacobian)(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
};

/*
   Nonlinear solver context
 */
#define MAXSNESMONITORS 5

struct _p_SNES {
  PETSCHEADER(struct _SNESOps);
  DM   dm;
  SNES pc;
  PetscBool usespc;

  /*  ------------------------ User-provided stuff -------------------------------*/
  void  *user;                   /* user-defined context */

  Vec  vec_rhs;                  /* If non-null, solve F(x) = rhs */
  Vec  vec_sol;                  /* pointer to solution */

  Vec  vec_func;                 /* pointer to function */
  void *funP;                    /* user-defined function context */

  Mat  jacobian;                 /* Jacobian matrix */
  Mat  jacobian_pre;             /* preconditioner matrix */
  void *jacP;                    /* user-defined Jacobian context */
  void *initialguessP;           /* user-defined initial guess context */
  void *gsP;                     /* user-defined Gauss-Seidel context */
  PetscBool usegs;               /* use a user-provided Gauss-Seidel routine */
  KSP  ksp;                      /* linear solver context */
  PetscBool usesksp;
  MatStructure matstruct;        /* Used by Picard solver */

  Vec  vec_sol_update;           /* pointer to solution update */

  Vec  scaling;                  /* scaling vector */
  void *scaP;                    /* scaling context */

  void *precheck;                /* user-defined step-checking context (optional) */
  void *postcheck;               /* user-defined step-checking context (optional) */
  void *lsP;                     /* user-defined line-search context (optional)   */

  PetscReal precheck_picard_angle; /* For use with SNESLineSearchPreCheckPicard */

  /* ------------------------Time stepping hooks-----------------------------------*/

  /* ---------------- PETSc-provided (or user-provided) stuff ---------------------*/

  PetscErrorCode      (*monitor[MAXSNESMONITORS])(SNES,PetscInt,PetscReal,void*); /* monitor routine */
  PetscErrorCode      (*monitordestroy[MAXSNESMONITORS])(void**);                 /* monitor context destroy routine */
  void                *monitorcontext[MAXSNESMONITORS];                           /* monitor context */
  PetscInt            numbermonitors;                                             /* number of monitors */
  void                *cnvP;                                                      /* convergence context */
  SNESConvergedReason reason;
  PetscBool           errorifnotconverged;

  /* --- Routines and data that are unique to each particular solver --- */

  PetscBool      setupcalled;                /* true if setup has been called */
  void           *data;                      /* implementation-specific data */

  /* --------------------------  Parameters -------------------------------------- */

  PetscInt    max_its;            /* max number of iterations */
  PetscInt    max_funcs;          /* max number of function evals */
  PetscInt    nfuncs;             /* number of function evaluations */
  PetscInt    iter;               /* global iteration number */
  PetscInt    linear_its;         /* total number of linear solver iterations */
  PetscReal   norm;               /* residual norm of current iterate */
  PetscReal   rtol;               /* relative tolerance */
  PetscReal   abstol;             /* absolute tolerance */
  PetscReal   xtol;               /* relative tolerance in solution */
  PetscReal   deltatol;           /* trust region convergence tolerance */
  PetscBool   printreason;        /* print reason for convergence/divergence after each solve */
  PetscInt    lagpreconditioner;  /* SNESSetLagPreconditioner() */
  PetscInt    lagjacobian;        /* SNESSetLagJacobian() */
  PetscInt    gridsequence;       /* number of grid sequence steps to take; defaults to zero */
  PetscInt    gssweeps;           /* number of GS sweeps */

  /* ------------------------ Line Search Parameters ---------------------- */

  SNESLineSearchType ls_type;     /* the present line search type */
  PetscReal   damping;            /* line search damping */
  PetscReal   maxstep;            /* line search maximum step size */
  PetscReal   steptol;            /* step convergence tolerance */
  PetscReal   ls_alpha;           /* line search sufficient reduction */
  PetscViewer ls_monitor;         /* monitor for the line search */
  PetscInt    ls_its;             /* number of iterates taken for iterative line searches */


  /* ------------------------ Default work-area management ---------------------- */

  PetscInt    nwork;
  Vec         *work;

  /* ------------------------- Miscellaneous Information ------------------------ */

  PetscReal   *conv_hist;         /* If !0, stores function norm (or
                                    gradient norm) at each iteration */
  PetscInt    *conv_hist_its;     /* linear iterations for each Newton step */
  PetscInt    conv_hist_len;      /* size of convergence history array */
  PetscInt    conv_hist_max;      /* actual amount of data in conv_history */
  PetscBool   conv_hist_reset;    /* reset counter for each new SNES solve */
  PetscBool   conv_malloc;

  /* the next two are used for failures in the line search; they should be put into the LS struct */
  PetscInt    numFailures;        /* number of unsuccessful step attempts */
  PetscInt    maxFailures;        /* maximum number of unsuccessful step attempts */

  PetscInt    numLinearSolveFailures;
  PetscInt    maxLinearSolveFailures;

  PetscBool   domainerror;       /* set with SNESSetFunctionDomainError() */

  PetscBool   ksp_ewconv;        /* flag indicating use of Eisenstat-Walker KSP convergence criteria */
  void        *kspconvctx;       /* Eisenstat-Walker KSP convergence context */

  PetscReal   ttol;           /* used by default convergence test routine */

  Vec         *vwork;            /* more work vectors for Jacobian approx */
  PetscInt    nvwork;

  PetscBool   mf_operator;      /* -snes_mf_operator was used on this snes */

  Vec         xl,xu;             /* upper and lower bounds for box constrained VI problems */
  PetscInt    ntruebounds;       /* number of non-infinite bounds set for VI box constraints */
};

/* Context for Eisenstat-Walker convergence criteria for KSP solvers */
typedef struct {
  PetscInt  version;             /* flag indicating version 1 or 2 of test */
  PetscReal rtol_0;              /* initial rtol */
  PetscReal rtol_last;           /* last rtol */
  PetscReal rtol_max;            /* maximum rtol */
  PetscReal gamma;               /* mult. factor for version 2 rtol computation */
  PetscReal alpha;               /* power for version 2 rtol computation */
  PetscReal alpha2;              /* power for safeguard */
  PetscReal threshold;           /* threshold for imposing safeguard */
  PetscReal lresid_last;         /* linear residual from last iteration */
  PetscReal norm_last;           /* function norm from last iteration */
} SNESKSPEW;

#define SNESLogConvHistory(snes,res,its) \
  { if (snes->conv_hist && snes->conv_hist_max > snes->conv_hist_len) \
    { if (snes->conv_hist)     snes->conv_hist[snes->conv_hist_len]     = res; \
      if (snes->conv_hist_its) snes->conv_hist_its[snes->conv_hist_len] = its; \
      snes->conv_hist_len++;\
    }}

extern PetscErrorCode SNESDefaultGetWork(SNES,PetscInt);

extern PetscErrorCode SNESVIProjectOntoBounds(SNES,Vec);
extern PetscErrorCode SNESVICheckLocalMin_Private(SNES,Mat,Vec,Vec,PetscReal,PetscBool*);
extern PetscErrorCode SNESReset_VI(SNES);
extern PetscErrorCode SNESDestroy_VI(SNES);
extern PetscErrorCode SNESView_VI(SNES,PetscViewer);
extern PetscErrorCode SNESSetFromOptions_VI(SNES);
extern PetscErrorCode SNESSetUp_VI(SNES);
typedef PetscErrorCode (*SNESVIComputeVariableBoundsFunction)(SNES,Vec,Vec);
EXTERN_C_BEGIN
extern PetscErrorCode SNESLineSearchSetType_VI(SNES,SNESLineSearchType);
extern PetscErrorCode SNESVISetComputeVariableBounds_VI(SNES,SNESVIComputeVariableBoundsFunction);
extern PetscErrorCode SNESVISetVariableBounds_VI(SNES,Vec,Vec);
EXTERN_C_END
extern PetscErrorCode SNESDefaultConverged_VI(SNES,PetscInt,PetscReal,PetscReal,PetscReal,SNESConvergedReason*,void*);

PetscErrorCode SNES_KSPSolve(SNES,KSP,Vec,Vec);
PetscErrorCode SNESScaleStep_Private(SNES,Vec,PetscReal*,PetscReal*,PetscReal*,PetscReal*);

extern PetscBool  SNESRegisterAllCalled;
extern PetscFList SNESList;

extern PetscLogEvent SNES_Solve, SNES_LineSearch, SNES_FunctionEval, SNES_JacobianEval, SNES_GSEval;

#endif

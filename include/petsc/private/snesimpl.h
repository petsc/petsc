#ifndef __SNESIMPL_H
#define __SNESIMPL_H

#include <petscsnes.h>
#include <petsc/private/petscimpl.h>

PETSC_EXTERN PetscBool      SNESRegisterAllCalled;
PETSC_EXTERN PetscErrorCode SNESRegisterAll(void);

typedef struct _SNESOps *SNESOps;

struct _SNESOps {
  PetscErrorCode (*computeinitialguess)(SNES, Vec, void *);
  PetscErrorCode (*computescaling)(Vec, Vec, void *);
  PetscErrorCode (*update)(SNES, PetscInt); /* General purpose function for update */
  PetscErrorCode (*converged)(SNES, PetscInt, PetscReal, PetscReal, PetscReal, SNESConvergedReason *, void *);
  PetscErrorCode (*convergeddestroy)(void *);
  PetscErrorCode (*setup)(SNES); /* routine to set up the nonlinear solver */
  PetscErrorCode (*solve)(SNES); /* actual nonlinear solver */
  PetscErrorCode (*view)(SNES, PetscViewer);
  PetscErrorCode (*setfromoptions)(SNES, PetscOptionItems *); /* sets options from database */
  PetscErrorCode (*destroy)(SNES);
  PetscErrorCode (*reset)(SNES);
  PetscErrorCode (*usercompute)(SNES, void **);
  PetscErrorCode (*userdestroy)(void **);
  PetscErrorCode (*computevariablebounds)(SNES, Vec, Vec); /* user provided routine to set box constrained variable bounds */
  PetscErrorCode (*computepfunction)(SNES, Vec, Vec, void *);
  PetscErrorCode (*computepjacobian)(SNES, Vec, Mat, Mat, void *);
  PetscErrorCode (*load)(SNES, PetscViewer);
};

/*
   Nonlinear solver context
 */
#define MAXSNESMONITORS    5
#define MAXSNESREASONVIEWS 5

struct _p_SNES {
  PETSCHEADER(struct _SNESOps);
  DM        dm;
  PetscBool dmAuto; /* SNES created currently used DM automatically */
  SNES      npc;
  PCSide    npcside;
  PetscBool usesnpc; /* type can use a nonlinear preconditioner */

  /*  ------------------------ User-provided stuff -------------------------------*/
  void *user; /* user-defined context */

  Vec vec_rhs; /* If non-null, solve F(x) = rhs */
  Vec vec_sol; /* pointer to solution */

  Vec vec_func; /* pointer to function */

  Mat            jacobian;      /* Jacobian matrix */
  Mat            jacobian_pre;  /* preconditioner matrix */
  Mat            picard;        /* copy of preconditioner matrix needed for Picard with -snes_mf_operator */
  void          *initialguessP; /* user-defined initial guess context */
  KSP            ksp;           /* linear solver context */
  SNESLineSearch linesearch;    /* line search context */
  PetscBool      usesksp;
  MatStructure   matstruct; /* Used by Picard solver */

  Vec vec_sol_update; /* pointer to solution update */

  Vec   scaling; /* scaling vector */
  void *scaP;    /* scaling context */

  PetscReal precheck_picard_angle; /* For use with SNESLineSearchPreCheckPicard */

  /* ------------------------Time stepping hooks-----------------------------------*/

  /* ---------------- PETSc-provided (or user-provided) stuff ---------------------*/

  PetscErrorCode (*monitor[MAXSNESMONITORS])(SNES, PetscInt, PetscReal, void *); /* monitor routine */
  PetscErrorCode (*monitordestroy[MAXSNESMONITORS])(void **);                    /* monitor context destroy routine */
  void               *monitorcontext[MAXSNESMONITORS];                           /* monitor context */
  PetscInt            numbermonitors;                                            /* number of monitors */
  PetscBool           pauseFinal;                                                /* pause all drawing monitor at the final iterate */
  void               *cnvP;                                                      /* convergence context */
  SNESConvergedReason reason;                                                    /* converged reason */
  PetscErrorCode (*reasonview[MAXSNESREASONVIEWS])(SNES, void *);                /* snes converged reason view */
  PetscErrorCode (*reasonviewdestroy[MAXSNESREASONVIEWS])(void **);              /* reason view context destroy routine */
  void     *reasonviewcontext[MAXSNESREASONVIEWS];                               /* reason view context */
  PetscInt  numberreasonviews;                                                   /* number of reason views */
  PetscBool errorifnotconverged;

  /* --- Routines and data that are unique to each particular solver --- */

  PetscBool setupcalled; /* true if setup has been called */
  void     *data;        /* implementation-specific data */

  /* --------------------------  Parameters -------------------------------------- */

  PetscInt  max_its;           /* max number of iterations */
  PetscInt  max_funcs;         /* max number of function evals */
  PetscInt  nfuncs;            /* number of function evaluations */
  PetscInt  iter;              /* global iteration number */
  PetscInt  linear_its;        /* total number of linear solver iterations */
  PetscReal norm;              /* residual norm of current iterate */
  PetscReal ynorm;             /* update norm of current iterate */
  PetscReal xnorm;             /* solution norm of current iterate */
  PetscReal rtol;              /* relative tolerance */
  PetscReal divtol;            /* relative divergence tolerance */
  PetscReal abstol;            /* absolute tolerance */
  PetscReal stol;              /* step length tolerance*/
  PetscReal deltatol;          /* trust region convergence tolerance */
  PetscBool forceiteration;    /* Force SNES to take at least one iteration regardless of the initial residual norm */
  PetscInt  lagpreconditioner; /* SNESSetLagPreconditioner() */
  PetscInt  lagjacobian;       /* SNESSetLagJacobian() */
  PetscInt  jac_iter;          /* The present iteration of the Jacobian lagging */
  PetscBool lagjac_persist;    /* The jac_iter persists until reset */
  PetscInt  pre_iter;          /* The present iteration of the Preconditioner lagging */
  PetscBool lagpre_persist;    /* The pre_iter persists until reset */
  PetscInt  gridsequence;      /* number of grid sequence steps to take; defaults to zero */

  PetscBool tolerancesset; /* SNESSetTolerances() called and tolerances should persist through SNESCreate_XXX()*/

  PetscBool vec_func_init_set; /* the initial function has been set */

  SNESNormSchedule normschedule; /* Norm computation type for SNES instance */
  SNESFunctionType functype;     /* Function type for the SNES instance */

  /* ------------------------ Default work-area management ---------------------- */

  PetscInt nwork;
  Vec     *work;

  /* ------------------------- Miscellaneous Information ------------------------ */

  PetscInt   setfromoptionscalled;
  PetscReal *conv_hist;       /* If !0, stores function norm (or
                                    gradient norm) at each iteration */
  PetscInt  *conv_hist_its;   /* linear iterations for each Newton step */
  size_t     conv_hist_len;   /* size of convergence history array */
  size_t     conv_hist_max;   /* actual amount of data in conv_history */
  PetscBool  conv_hist_reset; /* reset counter for each new SNES solve */
  PetscBool  conv_hist_alloc;
  PetscBool  counters_reset; /* reset counter for each new SNES solve */

  /* the next two are used for failures in the line search; they should be put elsewhere */
  PetscInt numFailures; /* number of unsuccessful step attempts */
  PetscInt maxFailures; /* maximum number of unsuccessful step attempts */

  PetscInt numLinearSolveFailures;
  PetscInt maxLinearSolveFailures;

  PetscBool domainerror;         /* set with SNESSetFunctionDomainError() */
  PetscBool jacobiandomainerror; /* set with SNESSetJacobianDomainError() */
  PetscBool checkjacdomainerror; /* if or not check Jacobian domain error after Jacobian evaluations */

  PetscBool ksp_ewconv; /* flag indicating use of Eisenstat-Walker KSP convergence criteria */
  void     *kspconvctx; /* Eisenstat-Walker KSP convergence context */

  /* SNESConvergedDefault context: split it off into a separate var/struct to be passed as context to SNESConvergedDefault? */
  PetscReal ttol;   /* rtol*initial_residual_norm */
  PetscReal rnorm0; /* initial residual norm (used for divergence testing) */

  Vec     *vwork; /* more work vectors for Jacobian approx */
  PetscInt nvwork;

  PetscBool mf;          /* -snes_mf was used on this snes */
  PetscBool mf_operator; /* -snes_mf_operator was used on this snes */
  PetscInt  mf_version;  /* The version of snes_mf used */

  PetscReal vizerotolerance; /* tolerance for considering an x[] value to be on the bound */
  Vec       xl, xu;          /* upper and lower bounds for box constrained VI problems */
  PetscInt  ntruebounds;     /* number of non-infinite bounds set for VI box constraints */
  PetscBool usersetbounds;   /* bounds have been set via SNESVISetVariableBounds(), rather than via computevariablebounds() callback. */

  PetscBool alwayscomputesfinalresidual; /* Does SNESSolve_XXX always compute the value of the residual at the final
                                             * solution and put it in vec_func?  Used inside SNESSolve_FAS to determine
                                             * if the final residual must be computed before restricting or prolonging
                                             * it. */
};

typedef struct _p_DMSNES  *DMSNES;
typedef struct _DMSNESOps *DMSNESOps;
struct _DMSNESOps {
  PetscErrorCode (*computefunction)(SNES, Vec, Vec, void *);
  PetscErrorCode (*computemffunction)(SNES, Vec, Vec, void *);
  PetscErrorCode (*computejacobian)(SNES, Vec, Mat, Mat, void *);

  /* objective */
  PetscErrorCode (*computeobjective)(SNES, Vec, PetscReal *, void *);

  /* Picard iteration functions */
  PetscErrorCode (*computepfunction)(SNES, Vec, Vec, void *);
  PetscErrorCode (*computepjacobian)(SNES, Vec, Mat, Mat, void *);

  /* User-defined smoother */
  PetscErrorCode (*computegs)(SNES, Vec, Vec, void *);

  PetscErrorCode (*destroy)(DMSNES);
  PetscErrorCode (*duplicate)(DMSNES, DMSNES);
};

struct _p_DMSNES {
  PETSCHEADER(struct _DMSNESOps);
  PetscContainer functionctxcontainer;
  PetscContainer jacobianctxcontainer;
  void          *mffunctionctx;
  void          *gsctx;
  void          *pctx;
  void          *objectivectx;

  void *data;

  /* This is NOT reference counted. The DM on which this context was first created is cached here to implement one-way
   * copy-on-write. When DMGetDMSNESWrite() sees a request using a different DM, it makes a copy. Thus, if a user
   * only interacts directly with one level, e.g., using SNESSetFunction(), then SNESSetUp_FAS() is called to build
   * coarse levels, then the user changes the routine with another call to SNESSetFunction(), it automatically
   * propagates to all the levels. If instead, they get out a specific level and set the function on that level,
   * subsequent changes to the original level will no longer propagate to that level.
   */
  DM originaldm;
};
PETSC_EXTERN PetscErrorCode DMGetDMSNES(DM, DMSNES *);
PETSC_EXTERN PetscErrorCode DMSNESView(DMSNES, PetscViewer);
PETSC_EXTERN PetscErrorCode DMSNESLoad(DMSNES, PetscViewer);
PETSC_EXTERN PetscErrorCode DMGetDMSNESWrite(DM, DMSNES *);

/* Context for Eisenstat-Walker convergence criteria for KSP solvers */
typedef struct {
  PetscInt  version;     /* flag indicating version (1,2,3 or 4) */
  PetscReal rtol_0;      /* initial rtol */
  PetscReal rtol_last;   /* last rtol */
  PetscReal rtol_max;    /* maximum rtol */
  PetscReal gamma;       /* mult. factor for version 2 rtol computation */
  PetscReal alpha;       /* power for version 2 rtol computation */
  PetscReal alpha2;      /* power for safeguard */
  PetscReal threshold;   /* threshold for imposing safeguard */
  PetscReal lresid_last; /* linear residual from last iteration */
  PetscReal norm_last;   /* function norm from last iteration */
  PetscReal norm_first;  /* function norm from the beginning of the first iteration. */
  PetscReal rtol_last_2, rk_last, rk_last_2;
  PetscReal v4_p1, v4_p2, v4_p3, v4_m1, v4_m2, v4_m3, v4_m4;
} SNESKSPEW;

static inline PetscErrorCode SNESLogConvergenceHistory(SNES snes, PetscReal res, PetscInt its)
{
  PetscFunctionBegin;
  PetscCall(PetscObjectSAWsTakeAccess((PetscObject)snes));
  if (snes->conv_hist && snes->conv_hist_max > snes->conv_hist_len) {
    if (snes->conv_hist) snes->conv_hist[snes->conv_hist_len] = res;
    if (snes->conv_hist_its) snes->conv_hist_its[snes->conv_hist_len] = its;
    snes->conv_hist_len++;
  }
  PetscCall(PetscObjectSAWsGrantAccess((PetscObject)snes));
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode SNESVIProjectOntoBounds(SNES, Vec);
PETSC_INTERN PetscErrorCode SNESVICheckLocalMin_Private(SNES, Mat, Vec, Vec, PetscReal, PetscBool *);
PETSC_INTERN PetscErrorCode SNESReset_VI(SNES);
PETSC_INTERN PetscErrorCode SNESDestroy_VI(SNES);
PETSC_INTERN PetscErrorCode SNESView_VI(SNES, PetscViewer);
PETSC_INTERN PetscErrorCode SNESSetFromOptions_VI(SNES, PetscOptionItems *);
PETSC_INTERN PetscErrorCode SNESSetUp_VI(SNES);
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode (*SNESVIComputeVariableBoundsFunction)(SNES, Vec, Vec);
PETSC_INTERN PetscErrorCode SNESVISetComputeVariableBounds_VI(SNES, SNESVIComputeVariableBoundsFunction);
PETSC_INTERN PetscErrorCode SNESVISetVariableBounds_VI(SNES, Vec, Vec);
PETSC_INTERN PetscErrorCode SNESConvergedDefault_VI(SNES, PetscInt, PetscReal, PetscReal, PetscReal, SNESConvergedReason *, void *);

PetscErrorCode              SNESScaleStep_Private(SNES, Vec, PetscReal *, PetscReal *, PetscReal *, PetscReal *);
PETSC_EXTERN PetscErrorCode DMSNESUnsetFunctionContext_Internal(DM);
PETSC_EXTERN PetscErrorCode DMSNESUnsetJacobianContext_Internal(DM);
PETSC_EXTERN PetscErrorCode DMSNESCheck_Internal(SNES, DM, Vec);

PETSC_EXTERN PetscLogEvent SNES_Solve;
PETSC_EXTERN PetscLogEvent SNES_Setup;
PETSC_EXTERN PetscLogEvent SNES_LineSearch;
PETSC_EXTERN PetscLogEvent SNES_FunctionEval;
PETSC_EXTERN PetscLogEvent SNES_JacobianEval;
PETSC_EXTERN PetscLogEvent SNES_NGSEval;
PETSC_EXTERN PetscLogEvent SNES_NGSFuncEval;
PETSC_EXTERN PetscLogEvent SNES_NPCSolve;
PETSC_EXTERN PetscLogEvent SNES_ObjectiveEval;

PETSC_INTERN PetscBool  SNEScite;
PETSC_INTERN const char SNESCitation[];

/* Used by TAOBNK solvers */
PETSC_EXTERN PetscErrorCode KSPPostSolve_SNESEW(KSP, Vec, Vec, SNES);
PETSC_EXTERN PetscErrorCode KSPPreSolve_SNESEW(KSP, Vec, Vec, SNES);
PETSC_EXTERN PetscErrorCode SNESEWSetFromOptions_Private(SNESKSPEW *, MPI_Comm, const char *);

/*
    Either generate an error or mark as diverged when a real from a SNES function norm is Nan or Inf.
    domainerror is reset here, once reason is set, to allow subsequent iterations to be feasible (e.g. line search).
*/
#define SNESCheckFunctionNorm(snes, beta) \
  do { \
    if (PetscIsInfOrNanReal(beta)) { \
      PetscCheck(!snes->errorifnotconverged, PetscObjectComm((PetscObject)snes), PETSC_ERR_NOT_CONVERGED, "SNESSolve has not converged due to Nan or Inf norm"); \
      { \
        PetscBool domainerror; \
        PetscCall(MPIU_Allreduce(&snes->domainerror, &domainerror, 1, MPIU_BOOL, MPI_LOR, PetscObjectComm((PetscObject)snes))); \
        if (domainerror) { \
          snes->reason      = SNES_DIVERGED_FUNCTION_DOMAIN; \
          snes->domainerror = PETSC_FALSE; \
        } else snes->reason = SNES_DIVERGED_FNORM_NAN; \
        PetscFunctionReturn(0); \
      } \
    } \
  } while (0)

#define SNESCheckJacobianDomainerror(snes) \
  do { \
    if (snes->checkjacdomainerror) { \
      PetscBool domainerror; \
      PetscCall(MPIU_Allreduce(&snes->jacobiandomainerror, &domainerror, 1, MPIU_BOOL, MPI_LOR, PetscObjectComm((PetscObject)snes))); \
      if (domainerror) { \
        snes->reason = SNES_DIVERGED_JACOBIAN_DOMAIN; \
        PetscCheck(!snes->errorifnotconverged, PetscObjectComm((PetscObject)snes), PETSC_ERR_NOT_CONVERGED, "SNESSolve has not converged due to Jacobian domain error"); \
        PetscFunctionReturn(0); \
      } \
    } \
  } while (0)

#define SNESCheckKSPSolve(snes) \
  do { \
    KSPConvergedReason kspreason; \
    PetscInt           lits; \
    PetscCall(KSPGetIterationNumber(snes->ksp, &lits)); \
    snes->linear_its += lits; \
    PetscCall(KSPGetConvergedReason(snes->ksp, &kspreason)); \
    if (kspreason < 0) { \
      if (kspreason == KSP_DIVERGED_NANORINF) { \
        PetscBool domainerror; \
        PetscCall(MPIU_Allreduce(&snes->domainerror, &domainerror, 1, MPIU_BOOL, MPI_LOR, PetscObjectComm((PetscObject)snes))); \
        if (domainerror) snes->reason = SNES_DIVERGED_FUNCTION_DOMAIN; \
        else snes->reason = SNES_DIVERGED_LINEAR_SOLVE; \
        PetscFunctionReturn(0); \
      } else { \
        if (++snes->numLinearSolveFailures >= snes->maxLinearSolveFailures) { \
          PetscCall(PetscInfo(snes, "iter=%" PetscInt_FMT ", number linear solve failures %" PetscInt_FMT " greater than current SNES allowed %" PetscInt_FMT ", stopping solve\n", snes->iter, snes->numLinearSolveFailures, snes->maxLinearSolveFailures)); \
          snes->reason = SNES_DIVERGED_LINEAR_SOLVE; \
          PetscFunctionReturn(0); \
        } \
      } \
    } \
  } while (0)

#endif

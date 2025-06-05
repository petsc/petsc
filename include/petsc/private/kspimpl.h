#pragma once

#include <petscksp.h>
#include <petscds.h>
#include <petsc/private/petscimpl.h>

/* SUBMANSEC = KSP */

PETSC_EXTERN PetscBool      KSPRegisterAllCalled;
PETSC_EXTERN PetscBool      KSPMonitorRegisterAllCalled;
PETSC_EXTERN PetscErrorCode KSPRegisterAll(void);
PETSC_EXTERN PetscErrorCode KSPMonitorRegisterAll(void);
PETSC_EXTERN PetscErrorCode KSPGuessRegisterAll(void);
PETSC_EXTERN PetscErrorCode KSPMatRegisterAll(void);

typedef struct _KSPOps *KSPOps;

struct _KSPOps {
  PetscErrorCode (*buildsolution)(KSP, Vec, Vec *);      /* Returns a pointer to the solution, or
                                                          calculates the solution in a
                                                          user-provided area. */
  PetscErrorCode (*buildresidual)(KSP, Vec, Vec, Vec *); /* Returns a pointer to the residual, or
                                                          calculates the residual in a
                                                          user-provided area.  */
  PetscErrorCode (*solve)(KSP);                          /* actual solver */
  PetscErrorCode (*matsolve)(KSP, Mat, Mat);             /* multiple dense RHS solver */
  PetscErrorCode (*setup)(KSP);
  PetscErrorCode (*setfromoptions)(KSP, PetscOptionItems);
  PetscErrorCode (*publishoptions)(KSP);
  PetscErrorCode (*computeextremesingularvalues)(KSP, PetscReal *, PetscReal *);
  PetscErrorCode (*computeeigenvalues)(KSP, PetscInt, PetscReal *, PetscReal *, PetscInt *);
  PetscErrorCode (*computeritz)(KSP, PetscBool, PetscBool, PetscInt *, Vec[], PetscReal *, PetscReal *);
  PetscErrorCode (*destroy)(KSP);
  PetscErrorCode (*view)(KSP, PetscViewer);
  PetscErrorCode (*reset)(KSP);
  PetscErrorCode (*load)(KSP, PetscViewer);
};

typedef struct _KSPGuessOps *KSPGuessOps;

struct _KSPGuessOps {
  PetscErrorCode (*formguess)(KSPGuess, Vec, Vec); /* Form initial guess */
  PetscErrorCode (*update)(KSPGuess, Vec, Vec);    /* Update database */
  PetscErrorCode (*setfromoptions)(KSPGuess);
  PetscErrorCode (*settolerance)(KSPGuess, PetscReal);
  PetscErrorCode (*setup)(KSPGuess);
  PetscErrorCode (*destroy)(KSPGuess);
  PetscErrorCode (*view)(KSPGuess, PetscViewer);
  PetscErrorCode (*reset)(KSPGuess);
};

/*
   Defines the KSPGuess data structure.
*/
struct _p_KSPGuess {
  PETSCHEADER(struct _KSPGuessOps);
  KSP              ksp;       /* the parent KSP */
  Mat              A;         /* the current linear operator */
  PetscObjectState omatstate; /* previous linear operator state */
  void            *data;      /* pointer to the specific implementation */
};

PETSC_EXTERN PetscErrorCode KSPGuessCreate_Fischer(KSPGuess);
PETSC_EXTERN PetscErrorCode KSPGuessCreate_POD(KSPGuess);

/*
     Maximum number of monitors you can run with a single KSP
*/
#define MAXKSPMONITORS    5
#define MAXKSPREASONVIEWS 5
typedef enum {
  KSP_SETUP_NEW = 0,
  KSP_SETUP_NEWMATRIX,
  KSP_SETUP_NEWRHS
} KSPSetUpStage;

/*
   Defines the KSP data structure.
*/
struct _p_KSP {
  PETSCHEADER(struct _KSPOps);
  DM        dm;
  PetscBool dmAuto;   /* DM was created automatically by KSP */
  PetscBool dmActive; /* KSP should use DM for computing operators */
  /*------------------------- User parameters--------------------------*/
  PetscObjectParameterDeclare(PetscInt, max_it); /* maximum number of iterations */
  PetscInt  min_it;                              /* minimum number of iterations */
  KSPGuess  guess;
  PetscBool guess_zero,                                 /* flag for whether initial guess is 0 */
    guess_not_read,                                     /* guess is not read, does not need to be zeroed */
    calc_sings,                                         /* calculate extreme Singular Values */
    calc_ritz,                                          /* calculate (harmonic) Ritz pairs */
    guess_knoll;                                        /* use initial guess of PCApply(ksp->B,b */
  PCSide   pc_side;                                     /* flag for left, right, or symmetric preconditioning */
  PetscInt normsupporttable[KSP_NORM_MAX][PC_SIDE_MAX]; /* Table of supported norms and pc_side, see KSPSetSupportedNorm() */
  PetscObjectParameterDeclare(PetscReal, rtol);         /* relative tolerance */
  PetscObjectParameterDeclare(PetscReal, abstol);       /* absolute tolerance */
  PetscObjectParameterDeclare(PetscReal, ttol);         /* (not set by user)  */
  PetscObjectParameterDeclare(PetscReal, divtol);       /* divergence tolerance */
  PetscReal          rnorm0;                            /* initial residual norm (used for divergence testing) */
  PetscReal          rnorm;                             /* current residual norm */
  KSPConvergedReason reason;
  PetscBool          errorifnotconverged; /* create an error if the KSPSolve() does not converge */

  Vec        vec_sol, vec_rhs; /* pointer to where user has stashed
                                      the solution and rhs, these are
                                      never touched by the code, only
                                      passed back to the user */
  PetscReal *res_hist;         /* If !0 stores residual each at iteration */
  PetscReal *res_hist_alloc;   /* If !0 means user did not provide buffer, needs deallocation */
  PetscCount res_hist_len;     /* current entry count of residual history array */
  PetscCount res_hist_max;     /* total entry count of storage in residual history */
  PetscBool  res_hist_reset;   /* reset history to length zero for each new solve */
  PetscReal *err_hist;         /* If !0 stores error at each iteration */
  PetscReal *err_hist_alloc;   /* If !0 means user did not provide buffer, needs deallocation */
  PetscCount err_hist_len;     /* current entry count of error history array */
  PetscCount err_hist_max;     /* total entry count of storage in error history */
  PetscBool  err_hist_reset;   /* reset history to length zero for each new solve */

  PetscInt  chknorm; /* only compute/check norm if iterations is great than this */
  PetscBool lagnorm; /* Lag the residual norm calculation so that it is computed as part of the
                                        MPI_Allreduce() for computing the inner products for the next iteration. */

  PetscInt nmax; /* maximum number of right-hand sides to be handled simultaneously */

  /* --------User (or default) routines (most return -1 on error) --------*/
  KSPMonitorFn      *monitor[MAXKSPMONITORS];
  PetscCtxDestroyFn *monitordestroy[MAXKSPMONITORS];
  void              *monitorcontext[MAXKSPMONITORS]; /* residual calculation, allows user */
  PetscInt           numbermonitors;                 /* to, for instance, print residual norm, etc. */
  PetscBool          pauseFinal;                     /* Pause all drawing monitor at the final iterate */

  PetscViewer               convergedreasonviewer;
  PetscViewerFormat         convergedreasonformat;
  KSPConvergedReasonViewFn *reasonview[MAXKSPREASONVIEWS];        /* KSP converged reason view */
  PetscCtxDestroyFn        *reasonviewdestroy[MAXKSPREASONVIEWS]; /* optional destroy routine */
  void                     *reasonviewcontext[MAXKSPREASONVIEWS]; /* viewer context */
  PetscInt                  numberreasonviews;                    /* current number of reason viewers */

  KSPConvergenceTestFn *converged;
  PetscCtxDestroyFn    *convergeddestroy;
  void                 *cnvP;

  void *ctx; /* optional user-defined context */

  PC pc;

  void *data; /* holder for misc stuff associated with a particular iterative solver */

  PetscBool         view, viewPre, viewRate, viewMat, viewPMat, viewRhs, viewSol, viewMatExp, viewEV, viewSV, viewEVExp, viewFinalRes, viewPOpExp, viewDScale;
  PetscViewer       viewer, viewerPre, viewerRate, viewerMat, viewerPMat, viewerRhs, viewerSol, viewerMatExp, viewerEV, viewerSV, viewerEVExp, viewerFinalRes, viewerPOpExp, viewerDScale;
  PetscViewerFormat format, formatPre, formatRate, formatMat, formatPMat, formatRhs, formatSol, formatMatExp, formatEV, formatSV, formatEVExp, formatFinalRes, formatPOpExp, formatDScale;

  /* ----------------Default work-area management -------------------- */
  PetscInt nwork;
  Vec     *work;

  KSPSetUpStage setupstage;
  PetscBool     setupnewmatrix; /* true if we need to call ksp->ops->setup with KSP_SETUP_NEWMATRIX */

  PetscInt its;      /* number of iterations so far computed in THIS linear solve*/
  PetscInt totalits; /* number of iterations used by this KSP object since it was created */

  PetscBool transpose_solve; /* solve transpose system instead */
  struct {
    Mat       AT, BT;
    PetscBool use_explicittranspose; /* transpose the system explicitly in KSPSolveTranspose */
    PetscBool reuse_transpose;       /* reuse the previous transposed system */
  } transpose;

  KSPNormType normtype; /* type of norm used for convergence tests */

  PCSide      pc_side_set;  /* PC type set explicitly by user */
  KSPNormType normtype_set; /* Norm type set explicitly by user */

  /*   Allow diagonally scaling the matrix before computing the preconditioner or using
       the Krylov method. Note this is NOT just Jacobi preconditioning */

  PetscBool dscale;     /* diagonal scale system; used with KSPSetDiagonalScale() */
  PetscBool dscalefix;  /* unscale system after solve */
  PetscBool dscalefix2; /* system has been unscaled */
  Vec       diagonal;   /* 1/sqrt(diag of matrix) */
  Vec       truediagonal;

  /* Allow declaring convergence when negative curvature is detected */
  PetscBool converged_neg_curve;

  PetscInt  setfromoptionscalled;
  PetscBool skippcsetfromoptions; /* if set then KSPSetFromOptions() does not call PCSetFromOptions() */

  PetscErrorCode (*presolve)(KSP, Vec, Vec, void *);
  PetscErrorCode (*postsolve)(KSP, Vec, Vec, void *);
  void *prectx, *postctx;

  PetscInt nestlevel; /* how many levels of nesting does the KSP have */
};

typedef struct { /* dummy data structure used in KSPMonitorDynamicTolerance() */
  PetscReal coef;
  PetscReal bnrm;
} KSPDynTolCtx;

typedef struct {
  PetscBool initialrtol;    /* default relative residual decrease is computed from initial residual, not rhs */
  PetscBool mininitialrtol; /* default relative residual decrease is computed from min of initial residual and rhs */
  PetscBool convmaxits;     /* if true, the convergence test returns KSP_CONVERGED_ITS if the maximum number of iterations is reached */
  Vec       work;
} KSPConvergedDefaultCtx;

static inline PetscErrorCode KSPLogResidualHistory(KSP ksp, PetscReal norm)
{
  PetscFunctionBegin;
  PetscCall(PetscObjectSAWsTakeAccess((PetscObject)ksp));
  if (ksp->res_hist && ksp->res_hist_max > ksp->res_hist_len) ksp->res_hist[ksp->res_hist_len++] = norm;
  PetscCall(PetscObjectSAWsGrantAccess((PetscObject)ksp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode KSPLogErrorHistory(KSP ksp)
{
  DM dm;

  PetscFunctionBegin;
  PetscCall(PetscObjectSAWsTakeAccess((PetscObject)ksp));
  PetscCall(KSPGetDM(ksp, &dm));
  if (dm && ksp->err_hist && ksp->err_hist_max > ksp->err_hist_len) {
    PetscSimplePointFn *exactSol;
    void               *exactCtx;
    PetscDS             ds;
    Vec                 u;
    PetscReal           error;
    PetscInt            Nf;

    PetscCall(KSPBuildSolution(ksp, NULL, &u));
    /* TODO Was needed to correct for Newton solution, but I just need to set a solution */
    //PetscCall(VecScale(u, -1.0));
    /* TODO Case when I have a solution */
    if (0) {
      PetscCall(DMGetDS(dm, &ds));
      PetscCall(PetscDSGetNumFields(ds, &Nf));
      PetscCheck(Nf <= 1, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Cannot handle number of fields %" PetscInt_FMT " > 1 right now", Nf);
      PetscCall(PetscDSGetExactSolution(ds, 0, &exactSol, &exactCtx));
      PetscCall(DMComputeL2FieldDiff(dm, 0.0, &exactSol, &exactCtx, u, &error));
    } else {
      /* The null solution A 0 = 0 */
      PetscCall(VecNorm(u, NORM_2, &error));
    }
    ksp->err_hist[ksp->err_hist_len++] = error;
  }
  PetscCall(PetscObjectSAWsGrantAccess((PetscObject)ksp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscScalar KSPNoisyHash_Private(PetscInt xx)
{
  unsigned int x = (unsigned int)xx;
  x              = ((x >> 16) ^ x) * 0x45d9f3b;
  x              = ((x >> 16) ^ x) * 0x45d9f3b;
  x              = ((x >> 16) ^ x);
  return (PetscScalar)(((PetscInt64)x - 2147483648) * 5.e-10); /* center around zero, scaled about -1. to 1.*/
}

static inline PetscErrorCode KSPSetNoisy_Private(Mat A, Vec v)
{
  PetscScalar *a;
  PetscInt     n, istart;
  MatNullSpace nullsp = NULL;

  PetscFunctionBegin;
  if (A) PetscCall(MatGetNullSpace(A, &nullsp));
  PetscCall(VecGetOwnershipRange(v, &istart, NULL));
  PetscCall(VecGetLocalSize(v, &n));
  PetscCall(VecGetArrayWrite(v, &a));
  for (PetscInt i = 0; i < n; ++i) a[i] = KSPNoisyHash_Private(i + istart);
  PetscCall(VecRestoreArrayWrite(v, &a));
  if (nullsp) PetscCall(MatNullSpaceRemove(nullsp, v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode KSPSetUpNorms_Private(KSP, PetscBool, KSPNormType *, PCSide *);

PETSC_INTERN PetscErrorCode KSPPlotEigenContours_Private(KSP, PetscInt, const PetscReal *, const PetscReal *);

typedef struct _p_DMKSP  *DMKSP;
typedef struct _DMKSPOps *DMKSPOps;
struct _DMKSPOps {
  KSPComputeOperatorsFn    *computeoperators;
  KSPComputeRHSFn          *computerhs;
  KSPComputeInitialGuessFn *computeinitialguess;
  PetscErrorCode (*destroy)(DMKSP *);
  PetscErrorCode (*duplicate)(DMKSP, DMKSP);
};

/*S
   DMKSP - Object held by a `DM` that contains all the callback functions and their contexts needed by a `KSP`

   Level: developer

   Notes:
   Users provides callback functions and their contexts to `KSP` using, for example, `KSPSetComputeRHS()`. These values are stored
   in a `DMKSP` that is contained in the `DM` associated with the `KSP`. If no `DM` was provided by
   the user with `KSPSetDM()` it is automatically created by `KSPGetDM()` with `DMShellCreate()`.

   Users very rarely need to worked directly with the `DMKSP` object, rather they work with the `KSP` and the `DM` they created

   Multiple `DM` can share a single `DMKSP`, often each `DM` is associated with
   a grid refinement level. `DMGetDMKSP()` returns the `DMKSP` associated with a `DM`. `DMGetDMKSPWrite()` returns a unique
   `DMKSP` that is only associated with the current `DM`, making a copy of the shared `DMKSP` if needed (copy-on-write).

   Developer Notes:
   It is rather subtle why `DMKSP`, `DMSNES`, and `DMTS` are needed instead of simply storing the user callback functions and contexts in `DM` or `KSP`, `SNES`, or `TS`.
   It is to support composable solvers such as geometric multigrid. We want, by default, the same callback functions and contexts for all the levels in the computation,
   but we need to also support different callbacks and contexts on each level. The copy-on-write approach of `DMGetDMKSPWrite()` makes this possible.

   The `originaldm` inside the `DMKSP` is NOT reference counted (to prevent a reference count loop between a `DM` and a `DMKSP`).
   The `DM` on which this context was first created is cached here to implement one-way
   copy-on-write. When `DMGetDMKSPWrite()` sees a request using a different `DM`, it makes a copy of the `TSDM`. Thus, if a user
   only interacts directly with one level, e.g., using `TSSetIFunction()`, then coarse levels of a multilevel item
   integrator are built, then the user changes the routine with another call to `TSSetIFunction()`, it automatically
   propagates to all the levels. If instead, they get out a specific level and set the function on that level,
   subsequent changes to the original level will no longer propagate to that level.

.seealso: [](ch_ts), `KSP`, `KSPCreate()`, `DM`, `DMGetDMKSPWrite()`, `DMGetDMKSP()`,  `DMSNES`, `DMTS`, `DMKSPSetComputeOperators()`, `DMKSPGetComputeOperators()`,
          `DMKSPSetComputeRHS()`, `DMKSPSetComputeInitialGuess()`
S*/
struct _p_DMKSP {
  PETSCHEADER(struct _DMKSPOps);
  void *operatorsctx;
  void *rhsctx;
  void *initialguessctx;
  void *data;

  /* See developer note for `DMKSP` above */
  DM originaldm;

  void (*fortran_func_pointers[3])(void); /* Store our own function pointers so they are associated with the DMKSP instead of the DM */
};
PETSC_EXTERN PetscErrorCode DMGetDMKSP(DM, DMKSP *);
PETSC_EXTERN PetscErrorCode DMGetDMKSPWrite(DM, DMKSP *);
PETSC_EXTERN PetscErrorCode DMCopyDMKSP(DM, DM);

/*
       These allow the various Krylov methods to apply to either the linear system or its transpose.
*/
static inline PetscErrorCode KSP_RemoveNullSpace(KSP ksp, Vec y)
{
  PetscFunctionBegin;
  if (ksp->pc_side == PC_LEFT) {
    Mat          A;
    MatNullSpace nullsp;

    PetscCall(PCGetOperators(ksp->pc, &A, NULL));
    PetscCall(MatGetNullSpace(A, &nullsp));
    if (nullsp) PetscCall(MatNullSpaceRemove(nullsp, y));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode KSP_RemoveNullSpaceTranspose(KSP ksp, Vec y)
{
  PetscFunctionBegin;
  if (ksp->pc_side == PC_LEFT) {
    Mat          A;
    MatNullSpace nullsp;

    PetscCall(PCGetOperators(ksp->pc, &A, NULL));
    PetscCall(MatGetTransposeNullSpace(A, &nullsp));
    if (nullsp) PetscCall(MatNullSpaceRemove(nullsp, y));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode KSP_MatMult(KSP ksp, Mat A, Vec x, Vec y)
{
  PetscFunctionBegin;
  if (ksp->transpose_solve) PetscCall(MatMultTranspose(A, x, y));
  else PetscCall(MatMult(A, x, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode KSP_MatMultTranspose(KSP ksp, Mat A, Vec x, Vec y)
{
  PetscFunctionBegin;
  if (ksp->transpose_solve) PetscCall(MatMult(A, x, y));
  else PetscCall(MatMultTranspose(A, x, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode KSP_MatMultHermitianTranspose(KSP ksp, Mat A, Vec x, Vec y)
{
  PetscFunctionBegin;
  if (!ksp->transpose_solve) PetscCall(MatMultHermitianTranspose(A, x, y));
  else {
    Vec w;

    PetscCall(VecDuplicate(x, &w));
    PetscCall(VecCopy(x, w));
    PetscCall(VecConjugate(w));
    PetscCall(MatMult(A, w, y));
    PetscCall(VecDestroy(&w));
    PetscCall(VecConjugate(y));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode KSP_PCApply(KSP ksp, Vec x, Vec y)
{
  PetscFunctionBegin;
  if (ksp->transpose_solve) {
    PetscCall(PCApplyTranspose(ksp->pc, x, y));
    PetscCall(KSP_RemoveNullSpaceTranspose(ksp, y));
  } else {
    PetscCall(PCApply(ksp->pc, x, y));
    PetscCall(KSP_RemoveNullSpace(ksp, y));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode KSP_PCApplyTranspose(KSP ksp, Vec x, Vec y)
{
  PetscFunctionBegin;
  if (ksp->transpose_solve) {
    PetscCall(PCApply(ksp->pc, x, y));
    PetscCall(KSP_RemoveNullSpace(ksp, y));
  } else {
    PetscCall(PCApplyTranspose(ksp->pc, x, y));
    PetscCall(KSP_RemoveNullSpaceTranspose(ksp, y));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode KSP_PCApplyHermitianTranspose(KSP ksp, Vec x, Vec y)
{
  PetscFunctionBegin;
  PetscCall(VecConjugate(x));
  PetscCall(KSP_PCApplyTranspose(ksp, x, y));
  PetscCall(VecConjugate(x));
  PetscCall(VecConjugate(y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode KSP_PCMatApply(KSP ksp, Mat X, Mat Y)
{
  PetscFunctionBegin;
  if (ksp->transpose_solve) PetscCall(PCMatApplyTranspose(ksp->pc, X, Y));
  else PetscCall(PCMatApply(ksp->pc, X, Y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode KSP_PCMatApplyTranspose(KSP ksp, Mat X, Mat Y)
{
  PetscFunctionBegin;
  if (!ksp->transpose_solve) PetscCall(PCMatApplyTranspose(ksp->pc, X, Y));
  else PetscCall(PCMatApply(ksp->pc, X, Y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode KSP_PCApplyBAorAB(KSP ksp, Vec x, Vec y, Vec w)
{
  PetscFunctionBegin;
  if (ksp->transpose_solve) {
    PetscCall(PCApplyBAorABTranspose(ksp->pc, ksp->pc_side, x, y, w));
    PetscCall(KSP_RemoveNullSpaceTranspose(ksp, y));
  } else {
    PetscCall(PCApplyBAorAB(ksp->pc, ksp->pc_side, x, y, w));
    PetscCall(KSP_RemoveNullSpace(ksp, y));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode KSP_PCApplyBAorABTranspose(KSP ksp, Vec x, Vec y, Vec w)
{
  PetscFunctionBegin;
  if (ksp->transpose_solve) PetscCall(PCApplyBAorAB(ksp->pc, ksp->pc_side, x, y, w));
  else PetscCall(PCApplyBAorABTranspose(ksp->pc, ksp->pc_side, x, y, w));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscLogEvent KSP_GMRESOrthogonalization;
PETSC_EXTERN PetscLogEvent KSP_SetUp;
PETSC_EXTERN PetscLogEvent KSP_Solve;
PETSC_EXTERN PetscLogEvent KSP_Solve_FS_0;
PETSC_EXTERN PetscLogEvent KSP_Solve_FS_1;
PETSC_EXTERN PetscLogEvent KSP_Solve_FS_2;
PETSC_EXTERN PetscLogEvent KSP_Solve_FS_3;
PETSC_EXTERN PetscLogEvent KSP_Solve_FS_4;
PETSC_EXTERN PetscLogEvent KSP_Solve_FS_S;
PETSC_EXTERN PetscLogEvent KSP_Solve_FS_L;
PETSC_EXTERN PetscLogEvent KSP_Solve_FS_U;
PETSC_EXTERN PetscLogEvent KSP_SolveTranspose;
PETSC_EXTERN PetscLogEvent KSP_MatSolve;
PETSC_EXTERN PetscLogEvent KSP_MatSolveTranspose;

PETSC_INTERN PetscErrorCode MatGetSchurComplement_Basic(Mat, IS, IS, IS, IS, MatReuse, Mat *, MatSchurComplementAinvType, MatReuse, Mat *);
PETSC_INTERN PetscErrorCode PCPreSolveChangeRHS(PC, PetscBool *);

/*MC
   KSPCheckDot - Checks if the result of a dot product used by the corresponding `KSP` contains Inf or NaN. These indicate that the previous
      application of the preconditioner generated an error. Sets a `KSPConvergedReason` and returns if the `PC` set a `PCFailedReason`.

   Collective

   Input Parameter:
.  ksp - the linear solver `KSP` context.

   Output Parameter:
.  beta - the result of the inner product

   Level: developer

   Developer Notes:
   Used to manage returning from `KSP` solvers collectively whose preconditioners have failed, possibly only a subset of MPI processes, in some way

   It uses the fact that `KSP` piggy-backs the collectivity of certain error conditions on the results of norms and inner products.

.seealso: `PCFailedReason`, `KSPConvergedReason`, `KSP`, `KSPCreate()`, `KSPSetType()`, `KSP`, `KSPCheckNorm()`, `KSPCheckSolve()`,
          `KSPSetErrorIfNotConverged()`
M*/
#define KSPCheckDot(ksp, beta) \
  do { \
    if (PetscIsInfOrNanScalar(beta)) { \
      PetscCheck(!ksp->errorifnotconverged, PetscObjectComm((PetscObject)ksp), PETSC_ERR_NOT_CONVERGED, "KSPSolve has not converged due to Nan or Inf inner product"); \
      { \
        PCFailedReason pcreason; \
        PetscCall(PCReduceFailedReason(ksp->pc)); \
        PetscCall(PCGetFailedReason(ksp->pc, &pcreason)); \
        PetscCall(VecFlag(ksp->vec_sol, pcreason)); \
        if (pcreason) { \
          ksp->reason = KSP_DIVERGED_PC_FAILED; \
        } else { \
          ksp->reason = KSP_DIVERGED_NANORINF; \
        } \
        PetscFunctionReturn(PETSC_SUCCESS); \
      } \
    } \
  } while (0)

/*MC
   KSPCheckNorm - Checks if the result of a norm used by the corresponding `KSP` contains `inf` or `NaN`. These indicate that the previous
      application of the preconditioner generated an error. Sets a `KSPConvergedReason` and returns if the `PC` set a `PCFailedReason`.

   Collective

   Input Parameter:
.  ksp - the linear solver `KSP` context.

   Output Parameter:
.  beta - the result of the norm

   Level: developer

   Developer Notes:
   Used to manage returning from `KSP` solvers collectively whose preconditioners have failed, possibly only a subset of MPI processes, in some way.

   It uses the fact that `KSP` piggy-backs the collectivity of certain error conditions on the results of norms and inner products.

.seealso: `PCFailedReason`, `KSPConvergedReason`, `KSP`, `KSPCreate()`, `KSPSetType()`, `KSP`, `KSPCheckDot()`, `KSPCheckSolve()`,
          `KSPSetErrorIfNotConverged()`
M*/
#define KSPCheckNorm(ksp, beta) \
  do { \
    if (PetscIsInfOrNanReal(beta)) { \
      PetscCheck(!ksp->errorifnotconverged, PetscObjectComm((PetscObject)ksp), PETSC_ERR_NOT_CONVERGED, "KSPSolve has not converged due to Nan or Inf norm"); \
      { \
        PCFailedReason pcreason; \
        PetscCall(PCReduceFailedReason(ksp->pc)); \
        PetscCall(PCGetFailedReason(ksp->pc, &pcreason)); \
        PetscCall(VecFlag(ksp->vec_sol, pcreason)); \
        if (pcreason) { \
          ksp->reason = KSP_DIVERGED_PC_FAILED; \
        } else { \
          ksp->reason = KSP_DIVERGED_NANORINF; \
        } \
        ksp->rnorm = beta; \
        PetscFunctionReturn(PETSC_SUCCESS); \
      } \
    } \
  } while (0)

PETSC_INTERN PetscErrorCode KSPMonitorMakeKey_Internal(const char[], PetscViewerType, PetscViewerFormat, char[]);
PETSC_INTERN PetscErrorCode KSPMonitorRange_Private(KSP, PetscInt, PetscReal *);

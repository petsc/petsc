
#ifndef _KSPIMPL_H
#define _KSPIMPL_H

#include <petscksp.h>
#include <petscds.h>
#include <petsc/private/petscimpl.h>

PETSC_EXTERN PetscBool KSPRegisterAllCalled;
PETSC_EXTERN PetscBool KSPMonitorRegisterAllCalled;
PETSC_EXTERN PetscErrorCode KSPRegisterAll(void);
PETSC_EXTERN PetscErrorCode KSPMonitorRegisterAll(void);
PETSC_EXTERN PetscErrorCode KSPGuessRegisterAll(void);
PETSC_EXTERN PetscErrorCode KSPMatRegisterAll(void);

typedef struct _KSPOps *KSPOps;

struct _KSPOps {
  PetscErrorCode (*buildsolution)(KSP,Vec,Vec*);       /* Returns a pointer to the solution, or
                                                          calculates the solution in a
                                                          user-provided area. */
  PetscErrorCode (*buildresidual)(KSP,Vec,Vec,Vec*);   /* Returns a pointer to the residual, or
                                                          calculates the residual in a
                                                          user-provided area.  */
  PetscErrorCode (*solve)(KSP);                        /* actual solver */
  PetscErrorCode (*matsolve)(KSP,Mat,Mat);             /* multiple dense RHS solver */
  PetscErrorCode (*setup)(KSP);
  PetscErrorCode (*setfromoptions)(PetscOptionItems*,KSP);
  PetscErrorCode (*publishoptions)(KSP);
  PetscErrorCode (*computeextremesingularvalues)(KSP,PetscReal*,PetscReal*);
  PetscErrorCode (*computeeigenvalues)(KSP,PetscInt,PetscReal*,PetscReal*,PetscInt *);
  PetscErrorCode (*computeritz)(KSP,PetscBool,PetscBool,PetscInt*,Vec[],PetscReal*,PetscReal*);
  PetscErrorCode (*destroy)(KSP);
  PetscErrorCode (*view)(KSP,PetscViewer);
  PetscErrorCode (*reset)(KSP);
  PetscErrorCode (*load)(KSP,PetscViewer);
};

typedef struct _KSPGuessOps *KSPGuessOps;

struct _KSPGuessOps {
  PetscErrorCode (*formguess)(KSPGuess,Vec,Vec); /* Form initial guess */
  PetscErrorCode (*update)(KSPGuess,Vec,Vec);    /* Update database */
  PetscErrorCode (*setfromoptions)(KSPGuess);
  PetscErrorCode (*settolerance)(KSPGuess,PetscReal);
  PetscErrorCode (*setup)(KSPGuess);
  PetscErrorCode (*destroy)(KSPGuess);
  PetscErrorCode (*view)(KSPGuess,PetscViewer);
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
  void             *data;     /* pointer to the specific implementation */
};

PETSC_EXTERN PetscErrorCode KSPGuessCreate_Fischer(KSPGuess);
PETSC_EXTERN PetscErrorCode KSPGuessCreate_POD(KSPGuess);

/*
     Maximum number of monitors you can run with a single KSP
*/
#define MAXKSPMONITORS 5
#define MAXKSPREASONVIEWS 5
typedef enum {KSP_SETUP_NEW, KSP_SETUP_NEWMATRIX, KSP_SETUP_NEWRHS} KSPSetUpStage;

/*
   Defines the KSP data structure.
*/
struct _p_KSP {
  PETSCHEADER(struct _KSPOps);
  DM              dm;
  PetscBool       dmAuto;       /* DM was created automatically by KSP */
  PetscBool       dmActive;     /* KSP should use DM for computing operators */
  /*------------------------- User parameters--------------------------*/
  PetscInt        max_it;                     /* maximum number of iterations */
  KSPGuess        guess;
  PetscBool       guess_zero,                  /* flag for whether initial guess is 0 */
                  calc_sings,                  /* calculate extreme Singular Values */
                  calc_ritz,                   /* calculate (harmonic) Ritz pairs */
                  guess_knoll;                /* use initial guess of PCApply(ksp->B,b */
  PCSide          pc_side;                  /* flag for left, right, or symmetric preconditioning */
  PetscInt        normsupporttable[KSP_NORM_MAX][PC_SIDE_MAX]; /* Table of supported norms and pc_side, see KSPSetSupportedNorm() */
  PetscReal       rtol,                     /* relative tolerance */
                  abstol,                     /* absolute tolerance */
                  ttol,                     /* (not set by user)  */
                  divtol;                   /* divergence tolerance */
  PetscReal       rnorm0;                   /* initial residual norm (used for divergence testing) */
  PetscReal       rnorm;                    /* current residual norm */
  KSPConvergedReason    reason;
  PetscBool             errorifnotconverged; /* create an error if the KSPSolve() does not converge */

  Vec vec_sol,vec_rhs;            /* pointer to where user has stashed
                                      the solution and rhs, these are
                                      never touched by the code, only
                                      passed back to the user */
  PetscReal     *res_hist;            /* If !0 stores residual each at iteration */
  PetscReal     *res_hist_alloc;      /* If !0 means user did not provide buffer, needs deallocation */
  size_t        res_hist_len;         /* current size of residual history array */
  size_t        res_hist_max;         /* actual amount of storage in residual history */
  PetscBool     res_hist_reset;       /* reset history to length zero for each new solve */
  PetscReal     *err_hist;            /* If !0 stores error at each iteration */
  PetscReal     *err_hist_alloc;      /* If !0 means user did not provide buffer, needs deallocation */
  size_t        err_hist_len;         /* current size of error history array */
  size_t        err_hist_max;         /* actual amount of storage in error history */
  PetscBool     err_hist_reset;       /* reset history to length zero for each new solve */

  PetscInt      chknorm;             /* only compute/check norm if iterations is great than this */
  PetscBool     lagnorm;             /* Lag the residual norm calculation so that it is computed as part of the
                                        MPI_Allreduce() for computing the inner products for the next iteration. */

  PetscInt   nmax;                   /* maximum number of right-hand sides to be handled simultaneously */

  /* --------User (or default) routines (most return -1 on error) --------*/
  PetscErrorCode (*monitor[MAXKSPMONITORS])(KSP,PetscInt,PetscReal,void*); /* returns control to user after */
  PetscErrorCode (*monitordestroy[MAXKSPMONITORS])(void**);         /* */
  void *monitorcontext[MAXKSPMONITORS];                  /* residual calculation, allows user */
  PetscInt  numbermonitors;                                   /* to, for instance, print residual norm, etc. */
  PetscBool        pauseFinal;        /* Pause all drawing monitor at the final iterate */

  PetscErrorCode (*reasonview[MAXKSPREASONVIEWS])(KSP,void*);       /* KSP converged reason view */
  PetscErrorCode (*reasonviewdestroy[MAXKSPREASONVIEWS])(void**);   /* Optional destroy routine */
  void *reasonviewcontext[MAXKSPREASONVIEWS];                       /* User context */
  PetscInt  numberreasonviews;                                      /* Number if reason viewers */

  PetscErrorCode (*converged)(KSP,PetscInt,PetscReal,KSPConvergedReason*,void*);
  PetscErrorCode (*convergeddestroy)(void*);
  void       *cnvP;

  void       *user;             /* optional user-defined context */

  PC         pc;

  void       *data;                      /* holder for misc stuff associated
                                   with a particular iterative solver */

  PetscBool         view,   viewPre,   viewRate,   viewMat,   viewPMat,   viewRhs,   viewSol,   viewMatExp,   viewEV,   viewSV,   viewEVExp,   viewFinalRes,   viewPOpExp,   viewDScale;
  PetscViewer       viewer, viewerPre, viewerRate, viewerMat, viewerPMat, viewerRhs, viewerSol, viewerMatExp, viewerEV, viewerSV, viewerEVExp, viewerFinalRes, viewerPOpExp, viewerDScale;
  PetscViewerFormat format, formatPre, formatRate, formatMat, formatPMat, formatRhs, formatSol, formatMatExp, formatEV, formatSV, formatEVExp, formatFinalRes, formatPOpExp, formatDScale;

  /* ----------------Default work-area management -------------------- */
  PetscInt       nwork;
  Vec            *work;

  KSPSetUpStage  setupstage;
  PetscBool      setupnewmatrix; /* true if we need to call ksp->ops->setup with KSP_SETUP_NEWMATRIX */

  PetscInt       its;       /* number of iterations so far computed in THIS linear solve*/
  PetscInt       totalits;   /* number of iterations used by this KSP object since it was created */

  PetscBool      transpose_solve;    /* solve transpose system instead */
  struct {
    Mat       AT,BT;
    PetscBool use_explicittranspose; /* transpose the system explicitly in KSPSolveTranspose */
    PetscBool reuse_transpose;       /* reuse the previous transposed system */
  } transpose;

  KSPNormType    normtype;          /* type of norm used for convergence tests */

  PCSide         pc_side_set;   /* PC type set explicitly by user */
  KSPNormType    normtype_set;  /* Norm type set explicitly by user */

  /*   Allow diagonally scaling the matrix before computing the preconditioner or using
       the Krylov method. Note this is NOT just Jacobi preconditioning */

  PetscBool    dscale;       /* diagonal scale system; used with KSPSetDiagonalScale() */
  PetscBool    dscalefix;    /* unscale system after solve */
  PetscBool    dscalefix2;   /* system has been unscaled */
  Vec          diagonal;     /* 1/sqrt(diag of matrix) */
  Vec          truediagonal;

  PetscInt     setfromoptionscalled;
  PetscBool    skippcsetfromoptions; /* if set then KSPSetFromOptions() does not call PCSetFromOptions() */

  PetscViewer  eigviewer;   /* Viewer where computed eigenvalues are displayed */

  PetscErrorCode (*presolve)(KSP,Vec,Vec,void*);
  PetscErrorCode (*postsolve)(KSP,Vec,Vec,void*);
  void           *prectx,*postctx;
};

typedef struct { /* dummy data structure used in KSPMonitorDynamicTolerance() */
  PetscReal coef;
  PetscReal bnrm;
} KSPDynTolCtx;

typedef struct {
  PetscBool  initialrtol;    /* default relative residual decrease is computed from initial residual, not rhs */
  PetscBool  mininitialrtol; /* default relative residual decrease is computed from min of initial residual and rhs */
  PetscBool  convmaxits;     /* if true, the convergence test returns KSP_CONVERGED_ITS if the maximum number of iterations is reached */
  Vec        work;
} KSPConvergedDefaultCtx;

static inline PetscErrorCode KSPLogResidualHistory(KSP ksp,PetscReal norm)
{
  PetscFunctionBegin;
  CHKERRQ(PetscObjectSAWsTakeAccess((PetscObject)ksp));
  if (ksp->res_hist && ksp->res_hist_max > ksp->res_hist_len) {
    ksp->res_hist[ksp->res_hist_len++] = norm;
  }
  CHKERRQ(PetscObjectSAWsGrantAccess((PetscObject)ksp));
  PetscFunctionReturn(0);
}

static inline PetscErrorCode KSPLogErrorHistory(KSP ksp)
{
  DM dm;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectSAWsTakeAccess((PetscObject) ksp));
  CHKERRQ(KSPGetDM(ksp, &dm));
  if (dm && ksp->err_hist && ksp->err_hist_max > ksp->err_hist_len) {
    PetscSimplePointFunc exactSol;
    void                *exactCtx;
    PetscDS              ds;
    Vec                  u;
    PetscReal            error;
    PetscInt             Nf;

    CHKERRQ(KSPBuildSolution(ksp, NULL, &u));
    /* TODO Was needed to correct for Newton solution, but I just need to set a solution */
    //CHKERRQ(VecScale(u, -1.0));
    /* TODO Case when I have a solution */
    if (0) {
      CHKERRQ(DMGetDS(dm, &ds));
      CHKERRQ(PetscDSGetNumFields(ds, &Nf));
      PetscCheck(Nf <= 1,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Cannot handle number of fields %D > 1 right now", Nf);
      CHKERRQ(PetscDSGetExactSolution(ds, 0, &exactSol, &exactCtx));
      CHKERRQ(DMComputeL2FieldDiff(dm, 0.0, &exactSol, &exactCtx, u, &error));
    } else {
      /* The null solution A 0 = 0 */
      CHKERRQ(VecNorm(u, NORM_2, &error));
    }
    ksp->err_hist[ksp->err_hist_len++] = error;
  }
  CHKERRQ(PetscObjectSAWsGrantAccess((PetscObject) ksp));
  PetscFunctionReturn(0);
}

static inline PetscScalar KSPNoisyHash_Private(PetscInt xx)
{
  unsigned int x = (unsigned int) xx;
  x = ((x >> 16) ^ x) * 0x45d9f3b;
  x = ((x >> 16) ^ x) * 0x45d9f3b;
  x = ((x >> 16) ^ x);
  return (PetscScalar)((PetscInt64)x-2147483648)*5.e-10; /* center around zero, scaled about -1. to 1.*/
}

static inline PetscErrorCode KSPSetNoisy_Private(Vec v)
{
  PetscScalar *a;
  PetscInt     n, istart;

  PetscFunctionBegin;
  CHKERRQ(VecGetOwnershipRange(v, &istart, NULL));
  CHKERRQ(VecGetLocalSize(v, &n));
  CHKERRQ(VecGetArrayWrite(v, &a));
  for (PetscInt i = 0; i < n; ++i) a[i] = KSPNoisyHash_Private(i+istart);
  CHKERRQ(VecRestoreArrayWrite(v, &a));
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode KSPSetUpNorms_Private(KSP,PetscBool,KSPNormType*,PCSide*);

PETSC_INTERN PetscErrorCode KSPPlotEigenContours_Private(KSP,PetscInt,const PetscReal*,const PetscReal*);

typedef struct _p_DMKSP *DMKSP;
typedef struct _DMKSPOps *DMKSPOps;
struct _DMKSPOps {
  PetscErrorCode (*computeoperators)(KSP,Mat,Mat,void*);
  PetscErrorCode (*computerhs)(KSP,Vec,void*);
  PetscErrorCode (*computeinitialguess)(KSP,Vec,void*);
  PetscErrorCode (*destroy)(DMKSP*);
  PetscErrorCode (*duplicate)(DMKSP,DMKSP);
};

struct _p_DMKSP {
  PETSCHEADER(struct _DMKSPOps);
  void *operatorsctx;
  void *rhsctx;
  void *initialguessctx;
  void *data;

  /* This is NOT reference counted. The DM on which this context was first created is cached here to implement one-way
   * copy-on-write. When DMGetDMKSPWrite() sees a request using a different DM, it makes a copy. Thus, if a user
   * only interacts directly with one level, e.g., using KSPSetComputeOperators(), then coarse levels are constructed by
   * PCMG, then the user changes the routine with another call to KSPSetComputeOperators(), it automatically propagates
   * to all the levels. If instead, they get out a specific level and set the routines on that level, subsequent changes
   * to the original level will no longer propagate to that level.
   */
  DM originaldm;

  void (*fortran_func_pointers[3])(void); /* Store our own function pointers so they are associated with the DMKSP instead of the DM */
};
PETSC_EXTERN PetscErrorCode DMGetDMKSP(DM,DMKSP*);
PETSC_EXTERN PetscErrorCode DMGetDMKSPWrite(DM,DMKSP*);
PETSC_EXTERN PetscErrorCode DMCopyDMKSP(DM,DM);

/*
       These allow the various Krylov methods to apply to either the linear system or its transpose.
*/
static inline PetscErrorCode KSP_RemoveNullSpace(KSP ksp,Vec y)
{
  PetscFunctionBegin;
  if (ksp->pc_side == PC_LEFT) {
    Mat          A;
    MatNullSpace nullsp;

    CHKERRQ(PCGetOperators(ksp->pc,&A,NULL));
    CHKERRQ(MatGetNullSpace(A,&nullsp));
    if (nullsp) CHKERRQ(MatNullSpaceRemove(nullsp,y));
  }
  PetscFunctionReturn(0);
}

static inline PetscErrorCode KSP_RemoveNullSpaceTranspose(KSP ksp,Vec y)
{
  PetscFunctionBegin;
  if (ksp->pc_side == PC_LEFT) {
    Mat          A;
    MatNullSpace nullsp;

    CHKERRQ(PCGetOperators(ksp->pc,&A,NULL));
    CHKERRQ(MatGetTransposeNullSpace(A,&nullsp));
    if (nullsp) CHKERRQ(MatNullSpaceRemove(nullsp,y));
  }
  PetscFunctionReturn(0);
}

static inline PetscErrorCode KSP_MatMult(KSP ksp,Mat A,Vec x,Vec y)
{
  PetscFunctionBegin;
  if (ksp->transpose_solve) CHKERRQ(MatMultTranspose(A,x,y));
  else                      CHKERRQ(MatMult(A,x,y));
  PetscFunctionReturn(0);
}

static inline PetscErrorCode KSP_MatMultTranspose(KSP ksp,Mat A,Vec x,Vec y)
{
  PetscFunctionBegin;
  if (ksp->transpose_solve) CHKERRQ(MatMult(A,x,y));
  else                      CHKERRQ(MatMultTranspose(A,x,y));
  PetscFunctionReturn(0);
}

static inline PetscErrorCode KSP_MatMultHermitianTranspose(KSP ksp,Mat A,Vec x,Vec y)
{
  PetscFunctionBegin;
  if (!ksp->transpose_solve) CHKERRQ(MatMultHermitianTranspose(A,x,y));
  else {
    Vec w;

    CHKERRQ(VecDuplicate(x,&w));
    CHKERRQ(VecCopy(x,w));
    CHKERRQ(VecConjugate(w));
    CHKERRQ(MatMult(A,w,y));
    CHKERRQ(VecDestroy(&w));
    CHKERRQ(VecConjugate(y));
  }
  PetscFunctionReturn(0);
}

static inline PetscErrorCode KSP_PCApply(KSP ksp,Vec x,Vec y)
{
  PetscFunctionBegin;
  if (ksp->transpose_solve) {
    CHKERRQ(PCApplyTranspose(ksp->pc,x,y));
    CHKERRQ(KSP_RemoveNullSpaceTranspose(ksp,y));
  } else {
    CHKERRQ(PCApply(ksp->pc,x,y));
    CHKERRQ(KSP_RemoveNullSpace(ksp,y));
  }
  PetscFunctionReturn(0);
}

static inline PetscErrorCode KSP_PCApplyTranspose(KSP ksp,Vec x,Vec y)
{
  PetscFunctionBegin;
  if (ksp->transpose_solve) {
    CHKERRQ(PCApply(ksp->pc,x,y));
    CHKERRQ(KSP_RemoveNullSpace(ksp,y));
  } else {
    CHKERRQ(PCApplyTranspose(ksp->pc,x,y));
    CHKERRQ(KSP_RemoveNullSpaceTranspose(ksp,y));
  }
  PetscFunctionReturn(0);
}

static inline PetscErrorCode KSP_PCApplyHermitianTranspose(KSP ksp,Vec x,Vec y)
{
  PetscFunctionBegin;
  CHKERRQ(VecConjugate(x));
  CHKERRQ(KSP_PCApplyTranspose(ksp,x,y));
  CHKERRQ(VecConjugate(x));
  CHKERRQ(VecConjugate(y));
  PetscFunctionReturn(0);
}

static inline PetscErrorCode KSP_PCApplyBAorAB(KSP ksp,Vec x,Vec y,Vec w)
{
  PetscFunctionBegin;
  if (ksp->transpose_solve) {
    CHKERRQ(PCApplyBAorABTranspose(ksp->pc,ksp->pc_side,x,y,w));
    CHKERRQ(KSP_RemoveNullSpaceTranspose(ksp,y));
  } else {
    CHKERRQ(PCApplyBAorAB(ksp->pc,ksp->pc_side,x,y,w));
    CHKERRQ(KSP_RemoveNullSpace(ksp,y));
  }
  PetscFunctionReturn(0);
}

static inline PetscErrorCode KSP_PCApplyBAorABTranspose(KSP ksp,Vec x,Vec y,Vec w)
{
  PetscFunctionBegin;
  if (ksp->transpose_solve) CHKERRQ(PCApplyBAorAB(ksp->pc,ksp->pc_side,x,y,w));
  else                      CHKERRQ(PCApplyBAorABTranspose(ksp->pc,ksp->pc_side,x,y,w));
  PetscFunctionReturn(0);
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

PETSC_INTERN PetscErrorCode MatGetSchurComplement_Basic(Mat,IS,IS,IS,IS,MatReuse,Mat*,MatSchurComplementAinvType,MatReuse,Mat*);
PETSC_INTERN PetscErrorCode PCPreSolveChangeRHS(PC,PetscBool*);

/*MC
   KSPCheckDot - Checks if the result of a dot product used by the corresponding KSP contains Inf or NaN. These indicate that the previous
      application of the preconditioner generated an error

   Collective on ksp

   Input Parameter:
.  ksp - the linear solver (KSP) context.

   Output Parameter:
.  beta - the result of the inner product

   Level: developer

   Developer Note:
   this is used to manage returning from KSP solvers whose preconditioners have failed in some way

.seealso: KSPCreate(), KSPSetType(), KSP, KSPCheckNorm(), KSPCheckSolve()
M*/
#define KSPCheckDot(ksp,beta) do { \
  if (PetscIsInfOrNanScalar(beta)) { \
    PetscCheck(!ksp->errorifnotconverged,PetscObjectComm((PetscObject)ksp),PETSC_ERR_NOT_CONVERGED,"KSPSolve has not converged due to Nan or Inf inner product");\
    else {\
      PCFailedReason pcreason;\
      PetscInt       sendbuf,recvbuf; \
      CHKERRQ(PCGetFailedReasonRank(ksp->pc,&pcreason));\
      sendbuf = (PetscInt)pcreason; \
      CHKERRMPI(MPI_Allreduce(&sendbuf,&recvbuf,1,MPIU_INT,MPI_MAX,PetscObjectComm((PetscObject)ksp)));\
      if (recvbuf) {                                                           \
        CHKERRQ(PCSetFailedReason(ksp->pc,(PCFailedReason)recvbuf)); \
        ksp->reason = KSP_DIVERGED_PC_FAILED;\
        CHKERRQ(VecSetInf(ksp->vec_sol));\
      } else {\
        ksp->reason = KSP_DIVERGED_NANORINF;\
      }\
      PetscFunctionReturn(0);\
    }\
  } } while (0)

/*MC
   KSPCheckNorm - Checks if the result of a norm used by the corresponding KSP contains Inf or NaN. These indicate that the previous
      application of the preconditioner generated an error

   Collective on ksp

   Input Parameter:
.  ksp - the linear solver (KSP) context.

   Output Parameter:
.  beta - the result of the norm

   Level: developer

   Developer Note:
   this is used to manage returning from KSP solvers whose preconditioners have failed in some way

.seealso: KSPCreate(), KSPSetType(), KSP, KSPCheckDot(), KSPCheckSolve()
M*/
#define KSPCheckNorm(ksp,beta) do { \
  if (PetscIsInfOrNanReal(beta)) { \
    PetscCheck(!ksp->errorifnotconverged,PetscObjectComm((PetscObject)ksp),PETSC_ERR_NOT_CONVERGED,"KSPSolve has not converged due to Nan or Inf norm");\
    else {\
      PCFailedReason pcreason;\
      PetscInt       sendbuf,recvbuf; \
      CHKERRQ(PCGetFailedReasonRank(ksp->pc,&pcreason));\
      sendbuf = (PetscInt)pcreason; \
      CHKERRMPI(MPI_Allreduce(&sendbuf,&recvbuf,1,MPIU_INT,MPI_MAX,PetscObjectComm((PetscObject)ksp)));\
      if (recvbuf) {                                                           \
        CHKERRQ(PCSetFailedReason(ksp->pc,(PCFailedReason)recvbuf)); \
        ksp->reason = KSP_DIVERGED_PC_FAILED;                         \
        CHKERRQ(VecSetInf(ksp->vec_sol));\
        ksp->rnorm  = beta; \
      } else {\
        CHKERRQ(PCSetFailedReason(ksp->pc,PC_NOERROR)); \
        ksp->reason = KSP_DIVERGED_NANORINF;\
        ksp->rnorm  = beta; \
      }                                       \
      PetscFunctionReturn(0);\
    }\
  } } while (0)

#endif

PETSC_INTERN PetscErrorCode KSPMonitorMakeKey_Internal(const char[], PetscViewerType, PetscViewerFormat, char[]);
PETSC_INTERN PetscErrorCode KSPMonitorRange_Private(KSP,PetscInt,PetscReal*);

#include <petsc/private/kspimpl.h> /*I "petscksp.h" I*/
#include <petscblaslapack.h>

typedef struct {
  PetscInt         method; /* 1, 2 or 3 */
  PetscInt         curl;   /* Current number of basis vectors */
  PetscInt         maxl;   /* Maximum number of basis vectors */
  PetscBool        monitor;
  PetscScalar     *alpha;  /* */
  Vec             *xtilde; /* Saved x vectors */
  Vec             *btilde; /* Saved b vectors, methods 1 and 3 */
  Vec              Ax;     /* method 2 */
  Vec              guess;
  PetscScalar     *corr;         /* correlation matrix in column-major format, method 3 */
  PetscReal        tol;          /* tolerance for determining rank, method 3 */
  Vec              last_b;       /* last b provided to FormGuess (not owned by this object), method 3 */
  PetscObjectState last_b_state; /* state of last_b as of the last call to FormGuess, method 3 */
  PetscScalar     *last_b_coefs; /* dot products of last_b and btilde, method 3 */
} KSPGuessFischer;

static PetscErrorCode KSPGuessReset_Fischer(KSPGuess guess)
{
  KSPGuessFischer *itg  = (KSPGuessFischer *)guess->data;
  PetscLayout      Alay = NULL, vlay = NULL;
  PetscBool        cong;

  PetscFunctionBegin;
  itg->curl = 0;
  /* destroy vectors if the size of the linear system has changed */
  if (guess->A) PetscCall(MatGetLayouts(guess->A, &Alay, NULL));
  if (itg->xtilde) PetscCall(VecGetLayout(itg->xtilde[0], &vlay));
  cong = PETSC_FALSE;
  if (vlay && Alay) PetscCall(PetscLayoutCompare(Alay, vlay, &cong));
  if (!cong) {
    PetscCall(VecDestroyVecs(itg->maxl, &itg->btilde));
    PetscCall(VecDestroyVecs(itg->maxl, &itg->xtilde));
    PetscCall(VecDestroy(&itg->guess));
    PetscCall(VecDestroy(&itg->Ax));
  }
  if (itg->corr) PetscCall(PetscMemzero(itg->corr, sizeof(*itg->corr) * itg->maxl * itg->maxl));
  itg->last_b       = NULL;
  itg->last_b_state = 0;
  if (itg->last_b_coefs) PetscCall(PetscMemzero(itg->last_b_coefs, sizeof(*itg->last_b_coefs) * itg->maxl));
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPGuessSetUp_Fischer(KSPGuess guess)
{
  KSPGuessFischer *itg = (KSPGuessFischer *)guess->data;

  PetscFunctionBegin;
  if (!itg->alpha) PetscCall(PetscMalloc1(itg->maxl, &itg->alpha));
  if (!itg->xtilde) PetscCall(KSPCreateVecs(guess->ksp, itg->maxl, &itg->xtilde, 0, NULL));
  if (!itg->btilde && (itg->method == 1 || itg->method == 3)) PetscCall(KSPCreateVecs(guess->ksp, itg->maxl, &itg->btilde, 0, NULL));
  if (!itg->Ax && itg->method == 2) PetscCall(VecDuplicate(itg->xtilde[0], &itg->Ax));
  if (!itg->guess && (itg->method == 1 || itg->method == 2)) PetscCall(VecDuplicate(itg->xtilde[0], &itg->guess));
  if (!itg->corr && itg->method == 3) PetscCall(PetscCalloc1(itg->maxl * itg->maxl, &itg->corr));
  if (!itg->last_b_coefs && itg->method == 3) PetscCall(PetscCalloc1(itg->maxl, &itg->last_b_coefs));
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPGuessDestroy_Fischer(KSPGuess guess)
{
  KSPGuessFischer *itg = (KSPGuessFischer *)guess->data;

  PetscFunctionBegin;
  PetscCall(PetscFree(itg->alpha));
  PetscCall(VecDestroyVecs(itg->maxl, &itg->btilde));
  PetscCall(VecDestroyVecs(itg->maxl, &itg->xtilde));
  PetscCall(VecDestroy(&itg->guess));
  PetscCall(VecDestroy(&itg->Ax));
  PetscCall(PetscFree(itg->corr));
  PetscCall(PetscFree(itg->last_b_coefs));
  PetscCall(PetscFree(itg));
  PetscCall(PetscObjectComposeFunction((PetscObject)guess, "KSPGuessFischerSetModel_C", NULL));
  PetscFunctionReturn(0);
}

/* Note: do not change the b right hand side as is done in the publication */
static PetscErrorCode KSPGuessFormGuess_Fischer_1(KSPGuess guess, Vec b, Vec x)
{
  KSPGuessFischer *itg = (KSPGuessFischer *)guess->data;
  PetscInt         i;

  PetscFunctionBegin;
  PetscCall(VecSet(x, 0.0));
  PetscCall(VecMDot(b, itg->curl, itg->btilde, itg->alpha));
  if (itg->monitor) {
    PetscCall(PetscPrintf(((PetscObject)guess)->comm, "KSPFischerGuess alphas ="));
    for (i = 0; i < itg->curl; i++) PetscCall(PetscPrintf(((PetscObject)guess)->comm, " %g", (double)PetscAbsScalar(itg->alpha[i])));
    PetscCall(PetscPrintf(((PetscObject)guess)->comm, "\n"));
  }
  PetscCall(VecMAXPY(x, itg->curl, itg->alpha, itg->xtilde));
  PetscCall(VecCopy(x, itg->guess));
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPGuessUpdate_Fischer_1(KSPGuess guess, Vec b, Vec x)
{
  KSPGuessFischer *itg = (KSPGuessFischer *)guess->data;
  PetscReal        norm;
  int              curl = itg->curl, i;

  PetscFunctionBegin;
  if (curl == itg->maxl) {
    PetscCall(KSP_MatMult(guess->ksp, guess->A, x, itg->btilde[0]));
    /* PetscCall(VecCopy(b,itg->btilde[0])); */
    PetscCall(VecNormalize(itg->btilde[0], &norm));
    PetscCall(VecCopy(x, itg->xtilde[0]));
    PetscCall(VecScale(itg->xtilde[0], 1.0 / norm));
    itg->curl = 1;
  } else {
    if (!curl) {
      PetscCall(VecCopy(x, itg->xtilde[curl]));
    } else {
      PetscCall(VecWAXPY(itg->xtilde[curl], -1.0, itg->guess, x));
    }
    PetscCall(KSP_MatMult(guess->ksp, guess->A, itg->xtilde[curl], itg->btilde[curl]));
    PetscCall(VecMDot(itg->btilde[curl], curl, itg->btilde, itg->alpha));
    for (i = 0; i < curl; i++) itg->alpha[i] = -itg->alpha[i];
    PetscCall(VecMAXPY(itg->btilde[curl], curl, itg->alpha, itg->btilde));
    PetscCall(VecMAXPY(itg->xtilde[curl], curl, itg->alpha, itg->xtilde));
    PetscCall(VecNormalize(itg->btilde[curl], &norm));
    if (norm) {
      PetscCall(VecScale(itg->xtilde[curl], 1.0 / norm));
      itg->curl++;
    } else {
      PetscCall(PetscInfo(guess->ksp, "Not increasing dimension of Fischer space because new direction is identical to previous\n"));
    }
  }
  PetscFunctionReturn(0);
}

/*
  Given a basis generated already this computes a new guess x from the new right hand side b
  Figures out the components of b in each btilde direction and adds them to x
  Note: do not change the b right hand side as is done in the publication
*/
static PetscErrorCode KSPGuessFormGuess_Fischer_2(KSPGuess guess, Vec b, Vec x)
{
  KSPGuessFischer *itg = (KSPGuessFischer *)guess->data;
  PetscInt         i;

  PetscFunctionBegin;
  PetscCall(VecSet(x, 0.0));
  PetscCall(VecMDot(b, itg->curl, itg->xtilde, itg->alpha));
  if (itg->monitor) {
    PetscCall(PetscPrintf(((PetscObject)guess)->comm, "KSPFischerGuess alphas ="));
    for (i = 0; i < itg->curl; i++) PetscCall(PetscPrintf(((PetscObject)guess)->comm, " %g", (double)PetscAbsScalar(itg->alpha[i])));
    PetscCall(PetscPrintf(((PetscObject)guess)->comm, "\n"));
  }
  PetscCall(VecMAXPY(x, itg->curl, itg->alpha, itg->xtilde));
  PetscCall(VecCopy(x, itg->guess));
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPGuessUpdate_Fischer_2(KSPGuess guess, Vec b, Vec x)
{
  KSPGuessFischer *itg = (KSPGuessFischer *)guess->data;
  PetscScalar      norm;
  int              curl = itg->curl, i;

  PetscFunctionBegin;
  if (curl == itg->maxl) {
    PetscCall(KSP_MatMult(guess->ksp, guess->A, x, itg->Ax)); /* norm = sqrt(x'Ax) */
    PetscCall(VecDot(x, itg->Ax, &norm));
    PetscCall(VecCopy(x, itg->xtilde[0]));
    PetscCall(VecScale(itg->xtilde[0], 1.0 / PetscSqrtScalar(norm)));
    itg->curl = 1;
  } else {
    if (!curl) {
      PetscCall(VecCopy(x, itg->xtilde[curl]));
    } else {
      PetscCall(VecWAXPY(itg->xtilde[curl], -1.0, itg->guess, x));
    }
    PetscCall(KSP_MatMult(guess->ksp, guess->A, itg->xtilde[curl], itg->Ax));
    PetscCall(VecMDot(itg->Ax, curl, itg->xtilde, itg->alpha));
    for (i = 0; i < curl; i++) itg->alpha[i] = -itg->alpha[i];
    PetscCall(VecMAXPY(itg->xtilde[curl], curl, itg->alpha, itg->xtilde));

    PetscCall(KSP_MatMult(guess->ksp, guess->A, itg->xtilde[curl], itg->Ax)); /* norm = sqrt(xtilde[curl]'Axtilde[curl]) */
    PetscCall(VecDot(itg->xtilde[curl], itg->Ax, &norm));
    if (PetscAbsScalar(norm) != 0.0) {
      PetscCall(VecScale(itg->xtilde[curl], 1.0 / PetscSqrtScalar(norm)));
      itg->curl++;
    } else {
      PetscCall(PetscInfo(guess->ksp, "Not increasing dimension of Fischer space because new direction is identical to previous\n"));
    }
  }
  PetscFunctionReturn(0);
}

/*
  Rather than the standard algorithm implemented in 2, we treat the provided x and b vectors to be spanning sets (not necessarily linearly independent) and use them to compute a windowed correlation matrix. Since the correlation matrix may be singular we solve it with the pseudoinverse, provided by SYEV/HEEV.
*/
static PetscErrorCode KSPGuessFormGuess_Fischer_3(KSPGuess guess, Vec b, Vec x)
{
  KSPGuessFischer *itg = (KSPGuessFischer *)guess->data;
  PetscInt         i, j, m;
  PetscReal       *s_values;
  PetscScalar     *corr, *work, *scratch_vec, zero = 0.0, one = 1.0;
  PetscBLASInt     blas_m, blas_info, blas_rank = 0, blas_lwork, blas_one = 1;
#if defined(PETSC_USE_COMPLEX)
  PetscReal *rwork;
#endif

  /* project provided b onto space of stored btildes */
  PetscFunctionBegin;
  PetscCall(VecSet(x, 0.0));
  m           = itg->curl;
  itg->last_b = b;
  PetscCall(PetscObjectStateGet((PetscObject)b, &itg->last_b_state));
  if (m > 0) {
    PetscCall(PetscBLASIntCast(m, &blas_m));
    blas_lwork = (/* assume a block size of m */ blas_m + 2) * blas_m;
#if defined(PETSC_USE_COMPLEX)
    PetscCall(PetscCalloc5(m * m, &corr, m, &s_values, blas_lwork, &work, 3 * m - 2, &rwork, m, &scratch_vec));
#else
    PetscCall(PetscCalloc4(m * m, &corr, m, &s_values, blas_lwork, &work, m, &scratch_vec));
#endif
    PetscCall(VecMDot(b, itg->curl, itg->btilde, itg->last_b_coefs));
    for (j = 0; j < m; ++j) {
      for (i = 0; i < m; ++i) corr[m * j + i] = itg->corr[(itg->maxl) * j + i];
    }
    PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
    PetscReal max_s_value = 0.0;
#if defined(PETSC_USE_COMPLEX)
    PetscCallBLAS("LAPACKheev", LAPACKheev_("V", "L", &blas_m, corr, &blas_m, s_values, work, &blas_lwork, rwork, &blas_info));
#else
    PetscCallBLAS("LAPACKsyev", LAPACKsyev_("V", "L", &blas_m, corr, &blas_m, s_values, work, &blas_lwork, &blas_info));
#endif

    if (blas_info == 0) {
      /* make corr store singular vectors and s_values store singular values */
      for (j = 0; j < m; ++j) {
        if (s_values[j] < 0.0) {
          s_values[j] = PetscAbsReal(s_values[j]);
          for (i = 0; i < m; ++i) corr[m * j + i] *= -1.0;
        }
        max_s_value = PetscMax(max_s_value, s_values[j]);
      }

      /* manually apply the action of the pseudoinverse */
      PetscCallBLAS("BLASgemv", BLASgemv_("T", &blas_m, &blas_m, &one, corr, &blas_m, itg->last_b_coefs, &blas_one, &zero, scratch_vec, &blas_one));
      for (j = 0; j < m; ++j) {
        if (s_values[j] > itg->tol * max_s_value) {
          scratch_vec[j] /= s_values[j];
          blas_rank += 1;
        } else {
          scratch_vec[j] = 0.0;
        }
      }
      PetscCallBLAS("BLASgemv", BLASgemv_("N", &blas_m, &blas_m, &one, corr, &blas_m, scratch_vec, &blas_one, &zero, itg->alpha, &blas_one));

    } else {
      PetscCall(PetscInfo(guess, "Warning eigenvalue solver failed with error code %d - setting initial guess to zero\n", (int)blas_info));
      PetscCall(PetscMemzero(itg->alpha, sizeof(*itg->alpha) * itg->maxl));
    }
    PetscCall(PetscFPTrapPop());

    if (itg->monitor && blas_info == 0) {
      PetscCall(PetscPrintf(((PetscObject)guess)->comm, "KSPFischerGuess correlation rank = %d\n", (int)blas_rank));
      PetscCall(PetscPrintf(((PetscObject)guess)->comm, "KSPFischerGuess singular values = "));
      for (i = 0; i < itg->curl; i++) PetscCall(PetscPrintf(((PetscObject)guess)->comm, " %g", (double)s_values[i]));
      PetscCall(PetscPrintf(((PetscObject)guess)->comm, "\n"));

      PetscCall(PetscPrintf(((PetscObject)guess)->comm, "KSPFischerGuess alphas ="));
      for (i = 0; i < itg->curl; i++) PetscCall(PetscPrintf(((PetscObject)guess)->comm, " %g", (double)PetscAbsScalar(itg->alpha[i])));
      PetscCall(PetscPrintf(((PetscObject)guess)->comm, "\n"));
    }
    /* Form the initial guess by using b's projection coefficients with the xs */
    PetscCall(VecMAXPY(x, itg->curl, itg->alpha, itg->xtilde));
#if defined(PETSC_USE_COMPLEX)
    PetscCall(PetscFree5(corr, s_values, work, rwork, scratch_vec));
#else
    PetscCall(PetscFree4(corr, s_values, work, scratch_vec));
#endif
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPGuessUpdate_Fischer_3(KSPGuess guess, Vec b, Vec x)
{
  KSPGuessFischer *itg    = (KSPGuessFischer *)guess->data;
  PetscBool        rotate = itg->curl == itg->maxl ? PETSC_TRUE : PETSC_FALSE;
  PetscInt         i, j;
  PetscObjectState b_state;
  PetscScalar     *last_column;
  Vec              oldest;

  PetscFunctionBegin;
  if (rotate) {
    /* we have the maximum number of vectors so rotate: oldest vector is at index 0 */
    oldest = itg->xtilde[0];
    for (i = 1; i < itg->curl; ++i) itg->xtilde[i - 1] = itg->xtilde[i];
    itg->xtilde[itg->curl - 1] = oldest;
    PetscCall(VecCopy(x, itg->xtilde[itg->curl - 1]));

    oldest = itg->btilde[0];
    for (i = 1; i < itg->curl; ++i) itg->btilde[i - 1] = itg->btilde[i];
    itg->btilde[itg->curl - 1] = oldest;
    PetscCall(VecCopy(b, itg->btilde[itg->curl - 1]));
    /* shift correlation matrix up and left */
    for (j = 1; j < itg->maxl; ++j) {
      for (i = 1; i < itg->maxl; ++i) itg->corr[(j - 1) * itg->maxl + i - 1] = itg->corr[j * itg->maxl + i];
    }
  } else {
    /* append new vectors */
    PetscCall(VecCopy(x, itg->xtilde[itg->curl]));
    PetscCall(VecCopy(b, itg->btilde[itg->curl]));
    itg->curl++;
  }

  /*
      Populate new column of the correlation matrix and then copy it into the
      row. itg->maxl is the allocated length per column: itg->curl is the actual
      column length.
      If possible reuse the dot products from FormGuess
  */
  last_column = itg->corr + (itg->curl - 1) * itg->maxl;
  PetscCall(PetscObjectStateGet((PetscObject)b, &b_state));
  if (b_state == itg->last_b_state && b == itg->last_b) {
    if (rotate) {
      for (i = 1; i < itg->maxl; ++i) itg->last_b_coefs[i - 1] = itg->last_b_coefs[i];
    }
    PetscCall(VecDot(b, b, &itg->last_b_coefs[itg->curl - 1]));
    PetscCall(PetscArraycpy(last_column, itg->last_b_coefs, itg->curl));
  } else {
    PetscCall(VecMDot(b, itg->curl, itg->btilde, last_column));
  }
  for (i = 0; i < itg->curl; ++i) itg->corr[i * itg->maxl + itg->curl - 1] = last_column[i];
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPGuessSetFromOptions_Fischer(KSPGuess guess)
{
  KSPGuessFischer *ITG  = (KSPGuessFischer *)guess->data;
  PetscInt         nmax = 2, model[2];
  PetscBool        flg;

  PetscFunctionBegin;
  model[0] = ITG->method;
  model[1] = ITG->maxl;
  PetscOptionsBegin(PetscObjectComm((PetscObject)guess), ((PetscObject)guess)->prefix, "Fischer guess options", "KSPGuess");
  PetscCall(PetscOptionsIntArray("-ksp_guess_fischer_model", "Model type and dimension of basis", "KSPGuessFischerSetModel", model, &nmax, &flg));
  if (flg) PetscCall(KSPGuessFischerSetModel(guess, model[0], model[1]));
  PetscCall(PetscOptionsReal("-ksp_guess_fischer_tol", "Tolerance to determine rank via ratio of singular values", "KSPGuessSetTolerance", ITG->tol, &ITG->tol, NULL));
  PetscCall(PetscOptionsBool("-ksp_guess_fischer_monitor", "Monitor the guess", NULL, ITG->monitor, &ITG->monitor, NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPGuessSetTolerance_Fischer(KSPGuess guess, PetscReal tol)
{
  KSPGuessFischer *itg = (KSPGuessFischer *)guess->data;

  PetscFunctionBegin;
  itg->tol = tol;
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPGuessView_Fischer(KSPGuess guess, PetscViewer viewer)
{
  KSPGuessFischer *itg = (KSPGuessFischer *)guess->data;
  PetscBool        isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) PetscCall(PetscViewerASCIIPrintf(viewer, "Model %" PetscInt_FMT ", size %" PetscInt_FMT "\n", itg->method, itg->maxl));
  PetscFunctionReturn(0);
}

/*@
   KSPGuessFischerSetModel - Use the Paul Fischer algorithm or its variants to compute the initial guess

   Logically Collective on guess

   Input Parameters:
+  guess - the initial guess context
.  model - use model 1, model 2, model 3, or any other number to turn it off
-  size  - size of subspace used to generate initial guess

    Options Database Key:
.   -ksp_guess_fischer_model <model,size> - uses the Fischer initial guess generator for repeated linear solves

   Level: advanced

.seealso: [](chapter_ksp), `KSPGuess`, `KSPGuessCreate()`, `KSPSetUseFischerGuess()`, `KSPSetGuess()`, `KSPGetGuess()`, `KSP`
@*/
PetscErrorCode KSPGuessFischerSetModel(KSPGuess guess, PetscInt model, PetscInt size)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(guess, KSPGUESS_CLASSID, 1);
  PetscValidLogicalCollectiveInt(guess, model, 2);
  PetscTryMethod(guess, "KSPGuessFischerSetModel_C", (KSPGuess, PetscInt, PetscInt), (guess, model, size));
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPGuessFischerSetModel_Fischer(KSPGuess guess, PetscInt model, PetscInt size)
{
  KSPGuessFischer *itg = (KSPGuessFischer *)guess->data;

  PetscFunctionBegin;
  if (model == 1) {
    guess->ops->update    = KSPGuessUpdate_Fischer_1;
    guess->ops->formguess = KSPGuessFormGuess_Fischer_1;
  } else if (model == 2) {
    guess->ops->update    = KSPGuessUpdate_Fischer_2;
    guess->ops->formguess = KSPGuessFormGuess_Fischer_2;
  } else if (model == 3) {
    guess->ops->update    = KSPGuessUpdate_Fischer_3;
    guess->ops->formguess = KSPGuessFormGuess_Fischer_3;
  } else {
    guess->ops->update    = NULL;
    guess->ops->formguess = NULL;
    itg->method           = 0;
    PetscFunctionReturn(0);
  }
  if (size != itg->maxl) {
    PetscCall(PetscFree(itg->alpha));
    PetscCall(VecDestroyVecs(itg->maxl, &itg->btilde));
    PetscCall(VecDestroyVecs(itg->maxl, &itg->xtilde));
    PetscCall(VecDestroy(&itg->guess));
    PetscCall(VecDestroy(&itg->Ax));
  }
  itg->method = model;
  itg->maxl   = size;
  PetscFunctionReturn(0);
}

/*
    KSPGUESSFISCHER - Implements Paul Fischer's two initial guess algorithms and a nonorthogonalizing variant for situations where
    a linear system is solved repeatedly

  References:
. * - https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19940020363_1994020363.pdf

   Notes:
    the algorithm is different from Fischer's paper because we do not CHANGE the right hand side of the new
    problem and solve the problem with an initial guess of zero, rather we solve the original problem
    with a nonzero initial guess (this is done so that the linear solver convergence tests are based on
    the original RHS). We use the xtilde = x - xguess as the new direction so that it is not
    mostly orthogonal to the previous solutions.

    These are not intended to be used directly, they are called by KSP automatically with the command line options -ksp_guess_type fischer -ksp_guess_fischer_model <int,int> or programmatically as
.vb
    KSPGetGuess(ksp,&guess);
    KSPGuessSetType(guess,KSPGUESSFISCHER);
    KSPGuessFischerSetModel(guess,model,basis);
    KSPGuessSetTolerance(guess,PETSC_MACHINE_EPSILON);

    The default tolerance (which is only used in Method 3) is 32*PETSC_MACHINE_EPSILON. This value was chosen
    empirically by trying a range of tolerances and picking the one that lowered the solver iteration count the most
    with five vectors.

    Method 2 is only for positive definite matrices, since it uses the A norm.

    Method 3 is not in the original paper. It is the same as the first two methods except that it
    does not orthogonalize the input vectors or use A at all. This choice is faster but provides a
    less effective initial guess for large (about 10) numbers of stored vectors.

    Developer note:
      The option -ksp_fischer_guess <int,int> is still available for backward compatibility

    Level: intermediate

@*/
PetscErrorCode KSPGuessCreate_Fischer(KSPGuess guess)
{
  KSPGuessFischer *fischer;

  PetscFunctionBegin;
  PetscCall(PetscNew(&fischer));
  fischer->method = 1; /* defaults to method 1 */
  fischer->maxl   = 10;
  fischer->tol    = 32.0 * PETSC_MACHINE_EPSILON;
  guess->data     = fischer;

  guess->ops->setfromoptions = KSPGuessSetFromOptions_Fischer;
  guess->ops->destroy        = KSPGuessDestroy_Fischer;
  guess->ops->settolerance   = KSPGuessSetTolerance_Fischer;
  guess->ops->setup          = KSPGuessSetUp_Fischer;
  guess->ops->view           = KSPGuessView_Fischer;
  guess->ops->reset          = KSPGuessReset_Fischer;
  guess->ops->update         = KSPGuessUpdate_Fischer_1;
  guess->ops->formguess      = KSPGuessFormGuess_Fischer_1;

  PetscCall(PetscObjectComposeFunction((PetscObject)guess, "KSPGuessFischerSetModel_C", KSPGuessFischerSetModel_Fischer));
  PetscFunctionReturn(0);
}

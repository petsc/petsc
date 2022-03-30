#define HPDDM_MIXED_PRECISION 1
#include <petsc/private/petschpddm.h> /*I "petscksp.h" I*/

const char *const KSPHPDDMTypes[]          = { KSPGMRES, "bgmres", KSPCG, "bcg", "gcrodr", "bgcrodr", "bfbcg", KSPPREONLY };
const char *const KSPHPDDMPrecisionTypes[] = { "HALF", "SINGLE", "DOUBLE", "QUADRUPLE", "KSPHPDDMPrecisionType", "KSP_HPDDM_PRECISION_", NULL };
const char *const HPDDMOrthogonalization[] = { "cgs", "mgs" };
const char *const HPDDMQR[]                = { "cholqr", "cgs", "mgs" };
const char *const HPDDMVariant[]           = { "left", "right", "flexible" };
const char *const HPDDMRecycleTarget[]     = { "SM", "LM", "SR", "LR", "SI", "LI" };
const char *const HPDDMRecycleStrategy[]   = { "A", "B" };

PetscBool HPDDMCite = PETSC_FALSE;
const char HPDDMCitation[] = "@article{jolivet2020petsc,\n"
"  Author = {Jolivet, Pierre and Roman, Jose E. and Zampini, Stefano},\n"
"  Title = {{KSPHPDDM} and {PCHPDDM}: Extending {PETSc} with Robust Overlapping {Schwarz} Preconditioners and Advanced {Krylov} Methods},\n"
"  Year = {2021},\n"
"  Publisher = {Elsevier},\n"
"  Journal = {Computer \\& Mathematics with Applications},\n"
"  Volume = {84},\n"
"  Pages = {277--295},\n"
"  Url = {https://github.com/prj-/jolivet2020petsc}\n"
"}\n";

#if defined(PETSC_HAVE_SLEPC) && defined(PETSC_USE_SHARED_LIBRARIES)
static PetscBool loadedDL = PETSC_FALSE;
#endif

static PetscErrorCode KSPSetFromOptions_HPDDM(PetscOptionItems *PetscOptionsObject, KSP ksp)
{
  KSP_HPDDM   *data = (KSP_HPDDM*)ksp->data;
  PetscInt    i, j;
  PetscMPIInt size;

  PetscFunctionBegin;
  PetscCall(PetscOptionsHead(PetscOptionsObject, "KSPHPDDM options, cf. https://github.com/hpddm/hpddm"));
  i = (data->cntl[0] == static_cast<char>(PETSC_DECIDE) ? HPDDM_KRYLOV_METHOD_GMRES : data->cntl[0]);
  PetscCall(PetscOptionsEList("-ksp_hpddm_type", "Type of Krylov method", "KSPHPDDMGetType", KSPHPDDMTypes, PETSC_STATIC_ARRAY_LENGTH(KSPHPDDMTypes), KSPHPDDMTypes[HPDDM_KRYLOV_METHOD_GMRES], &i, NULL));
  if (i == PETSC_STATIC_ARRAY_LENGTH(KSPHPDDMTypes) - 1)
    i = HPDDM_KRYLOV_METHOD_NONE; /* need to shift the value since HPDDM_KRYLOV_METHOD_RICHARDSON is not registered in PETSc */
  data->cntl[0] = i;
  PetscCall(PetscOptionsEnum("-ksp_hpddm_precision", "Precision in which Krylov bases are stored", "KSPHPDDM", KSPHPDDMPrecisionTypes, (PetscEnum)data->precision, (PetscEnum*)&data->precision, NULL));
  PetscCheck(data->precision == KSP_HPDDM_PRECISION_SINGLE || data->precision == KSP_HPDDM_PRECISION_DOUBLE, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unhandled %s precision", KSPHPDDMPrecisionTypes[data->precision]);
  if (data->cntl[0] != HPDDM_KRYLOV_METHOD_NONE) {
    if (data->cntl[0] != HPDDM_KRYLOV_METHOD_BCG && data->cntl[0] != HPDDM_KRYLOV_METHOD_BFBCG) {
      i = (data->cntl[1] == static_cast<char>(PETSC_DECIDE) ? HPDDM_VARIANT_LEFT : data->cntl[1]);
      if (ksp->pc_side_set == PC_SIDE_DEFAULT) PetscCall(PetscOptionsEList("-ksp_hpddm_variant", "Left, right, or variable preconditioning", "KSPHPDDM", HPDDMVariant, PETSC_STATIC_ARRAY_LENGTH(HPDDMVariant), HPDDMVariant[HPDDM_VARIANT_LEFT], &i, NULL));
      else if (ksp->pc_side_set == PC_RIGHT) i = HPDDM_VARIANT_RIGHT;
      else PetscCheck(ksp->pc_side_set != PC_SYMMETRIC, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Symmetric preconditioning not implemented");
      data->cntl[1] = i;
      if (i > 0) PetscCall(KSPSetPCSide(ksp, PC_RIGHT));
    }
    if (data->cntl[0] == HPDDM_KRYLOV_METHOD_BGMRES || data->cntl[0] == HPDDM_KRYLOV_METHOD_BGCRODR || data->cntl[0] == HPDDM_KRYLOV_METHOD_BFBCG) {
      data->rcntl[0] = (std::abs(data->rcntl[0] - static_cast<PetscReal>(PETSC_DECIDE)) < PETSC_SMALL ? -1.0 : data->rcntl[0]);
      PetscCall(PetscOptionsReal("-ksp_hpddm_deflation_tol", "Tolerance when deflating right-hand sides inside block methods", "KSPHPDDM", data->rcntl[0], data->rcntl, NULL));
      i = (data->scntl[data->cntl[0] != HPDDM_KRYLOV_METHOD_BFBCG] == static_cast<unsigned short>(PETSC_DECIDE) ? 1 : PetscMax(1, data->scntl[data->cntl[0] != HPDDM_KRYLOV_METHOD_BFBCG]));
      PetscCall(PetscOptionsRangeInt("-ksp_hpddm_enlarge_krylov_subspace", "Split the initial right-hand side into multiple vectors", "KSPHPDDM", i, &i, NULL, 1, std::numeric_limits<unsigned short>::max() - 1));
      data->scntl[data->cntl[0] != HPDDM_KRYLOV_METHOD_BFBCG] = i;
    } else data->scntl[data->cntl[0] != HPDDM_KRYLOV_METHOD_BCG] = 0;
    if (data->cntl[0] == HPDDM_KRYLOV_METHOD_GMRES || data->cntl[0] == HPDDM_KRYLOV_METHOD_BGMRES || data->cntl[0] == HPDDM_KRYLOV_METHOD_GCRODR || data->cntl[0] == HPDDM_KRYLOV_METHOD_BGCRODR) {
      i = (data->cntl[2] == static_cast<char>(PETSC_DECIDE) ? HPDDM_ORTHOGONALIZATION_CGS : data->cntl[2] & 3);
      PetscCall(PetscOptionsEList("-ksp_hpddm_orthogonalization", "Classical (faster) or Modified (more robust) Gram--Schmidt process", "KSPHPDDM", HPDDMOrthogonalization, PETSC_STATIC_ARRAY_LENGTH(HPDDMOrthogonalization), HPDDMOrthogonalization[HPDDM_ORTHOGONALIZATION_CGS], &i, NULL));
      j = (data->cntl[2] == static_cast<char>(PETSC_DECIDE) ? HPDDM_QR_CHOLQR : ((data->cntl[2] >> 2) & 7));
      PetscCall(PetscOptionsEList("-ksp_hpddm_qr", "Distributed QR factorizations computed with Cholesky QR, Classical or Modified Gram--Schmidt process", "KSPHPDDM", HPDDMQR, PETSC_STATIC_ARRAY_LENGTH(HPDDMQR), HPDDMQR[HPDDM_QR_CHOLQR], &j, NULL));
      data->cntl[2] = static_cast<char>(i) + (static_cast<char>(j) << 2);
      i = (data->scntl[0] == static_cast<unsigned short>(PETSC_DECIDE) ? PetscMin(30, ksp->max_it) : data->scntl[0]);
      PetscCall(PetscOptionsRangeInt("-ksp_gmres_restart", "Maximum number of Arnoldi vectors generated per cycle", "KSPHPDDM", i, &i, NULL, PetscMin(1, ksp->max_it), PetscMin(ksp->max_it, std::numeric_limits<unsigned short>::max() - 1)));
      data->scntl[0] = i;
    }
    if (data->cntl[0] == HPDDM_KRYLOV_METHOD_BCG || data->cntl[0] == HPDDM_KRYLOV_METHOD_BFBCG) {
      j = (data->cntl[1] == static_cast<char>(PETSC_DECIDE) ? HPDDM_QR_CHOLQR : data->cntl[1]);
      PetscCall(PetscOptionsEList("-ksp_hpddm_qr", "Distributed QR factorizations computed with Cholesky QR, Classical or Modified Gram--Schmidt process", "KSPHPDDM", HPDDMQR, PETSC_STATIC_ARRAY_LENGTH(HPDDMQR), HPDDMQR[HPDDM_QR_CHOLQR], &j, NULL));
      data->cntl[1] = j;
    }
    if (data->cntl[0] == HPDDM_KRYLOV_METHOD_GCRODR || data->cntl[0] == HPDDM_KRYLOV_METHOD_BGCRODR) {
      i = (data->icntl[0] == static_cast<int>(PETSC_DECIDE) ? PetscMin(20, data->scntl[0] - 1) : data->icntl[0]);
      PetscCall(PetscOptionsRangeInt("-ksp_hpddm_recycle", "Number of harmonic Ritz vectors to compute", "KSPHPDDM", i, &i, NULL, 1, data->scntl[0] - 1));
      data->icntl[0] = i;
      if (!PetscDefined(HAVE_SLEPC) || !PetscDefined(USE_SHARED_LIBRARIES) || data->cntl[0] == HPDDM_KRYLOV_METHOD_GCRODR) {
        i = (data->cntl[3] == static_cast<char>(PETSC_DECIDE) ? HPDDM_RECYCLE_TARGET_SM : data->cntl[3]);
        PetscCall(PetscOptionsEList("-ksp_hpddm_recycle_target", "Criterion to select harmonic Ritz vectors", "KSPHPDDM", HPDDMRecycleTarget, PETSC_STATIC_ARRAY_LENGTH(HPDDMRecycleTarget), HPDDMRecycleTarget[HPDDM_RECYCLE_TARGET_SM], &i, NULL));
        data->cntl[3] = i;
      } else {
        PetscCheck(data->precision == PETSC_KSPHPDDM_DEFAULT_PRECISION, PetscObjectComm((PetscObject)ksp), PETSC_ERR_ARG_INCOMP, "Cannot use SLEPc with a different precision than PETSc for harmonic Ritz eigensolves");
        PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)ksp), &size));
        i = (data->cntl[3] == static_cast<char>(PETSC_DECIDE) ? 1 : data->cntl[3]);
        PetscCall(PetscOptionsRangeInt("-ksp_hpddm_recycle_redistribute", "Number of processes used to solve eigenvalue problems when recycling in BGCRODR", "KSPHPDDM", i, &i, NULL, 1, PetscMin(size, 192)));
        data->cntl[3] = i;
      }
      i = (data->cntl[4] == static_cast<char>(PETSC_DECIDE) ? HPDDM_RECYCLE_STRATEGY_A : data->cntl[4]);
      PetscCall(PetscOptionsEList("-ksp_hpddm_recycle_strategy", "Generalized eigenvalue problem to solve for recycling", "KSPHPDDM", HPDDMRecycleStrategy, PETSC_STATIC_ARRAY_LENGTH(HPDDMRecycleStrategy), HPDDMRecycleStrategy[HPDDM_RECYCLE_STRATEGY_A], &i, NULL));
      data->cntl[4] = i;
    }
  } else {
    data->cntl[0] = HPDDM_KRYLOV_METHOD_NONE;
    data->scntl[1] = 1;
  }
  PetscCheck(ksp->nmax >= std::numeric_limits<int>::min() && ksp->nmax <= std::numeric_limits<int>::max(), PetscObjectComm((PetscObject)ksp), PETSC_ERR_ARG_OUTOFRANGE, "KSPMatSolve() block size %" PetscInt_FMT " not representable by an integer, which is not handled by KSPHPDDM", ksp->nmax);
  else data->icntl[1] = static_cast<int>(ksp->nmax);
  PetscCall(PetscOptionsTail());
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPView_HPDDM(KSP ksp, PetscViewer viewer)
{
  KSP_HPDDM            *data = (KSP_HPDDM*)ksp->data;
  HPDDM::PETScOperator *op = data->op;
  const PetscScalar    *array = op ? op->storage() : NULL;
  PetscBool            ascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &ascii));
  if (op && ascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "HPDDM type: %s\n", KSPHPDDMTypes[std::min(static_cast<PetscInt>(data->cntl[0]), static_cast<PetscInt>(PETSC_STATIC_ARRAY_LENGTH(KSPHPDDMTypes) - 1))]));
    PetscCall(PetscViewerASCIIPrintf(viewer, "precision: %s\n", KSPHPDDMPrecisionTypes[data->precision]));
    if (data->cntl[0] == HPDDM_KRYLOV_METHOD_BGMRES || data->cntl[0] == HPDDM_KRYLOV_METHOD_BGCRODR || data->cntl[0] == HPDDM_KRYLOV_METHOD_BFBCG) {
      if (std::abs(data->rcntl[0] - static_cast<PetscReal>(PETSC_DECIDE)) < PETSC_SMALL) PetscCall(PetscViewerASCIIPrintf(viewer, "no deflation at restarts\n", PetscBools[array ? PETSC_TRUE : PETSC_FALSE]));
      else PetscCall(PetscViewerASCIIPrintf(viewer, "deflation tolerance: %g\n", data->rcntl[0]));
    }
    if (data->cntl[0] == HPDDM_KRYLOV_METHOD_GCRODR || data->cntl[0] == HPDDM_KRYLOV_METHOD_BGCRODR) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "deflation subspace attached? %s\n", PetscBools[array ? PETSC_TRUE : PETSC_FALSE]));
      if (!PetscDefined(HAVE_SLEPC) || !PetscDefined(USE_SHARED_LIBRARIES) || data->cntl[0] == HPDDM_KRYLOV_METHOD_GCRODR) PetscCall(PetscViewerASCIIPrintf(viewer, "deflation target: %s\n", HPDDMRecycleTarget[static_cast<PetscInt>(data->cntl[3])]));
      else PetscCall(PetscViewerASCIIPrintf(viewer, "redistribution size: %d\n", static_cast<PetscMPIInt>(data->cntl[3])));
    }
    if (data->icntl[1] != static_cast<int>(PETSC_DECIDE)) PetscCall(PetscViewerASCIIPrintf(viewer, "  block size is %d\n", data->icntl[1]));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPSetUp_HPDDM(KSP ksp)
{
  KSP_HPDDM *data = (KSP_HPDDM*)ksp->data;
  Mat       A;
  PetscInt  n, bs;
  PetscBool match;

  PetscFunctionBegin;
  PetscCall(KSPGetOperators(ksp, &A, NULL));
  PetscCall(MatGetLocalSize(A, &n, NULL));
  PetscCall(MatGetBlockSize(A, &bs));
  PetscCall(PetscObjectTypeCompareAny((PetscObject)A, &match, MATSEQKAIJ, MATMPIKAIJ, ""));
  if (match) n /= bs;
  data->op = new HPDDM::PETScOperator(ksp, n);
  if (PetscUnlikely(!ksp->setfromoptionscalled || data->cntl[0] == static_cast<char>(PETSC_DECIDE))) { /* what follows is basically a copy/paste of KSPSetFromOptions_HPDDM, with no call to PetscOptions() */
    PetscCall(PetscInfo(ksp, "KSPSetFromOptions() not called or uninitialized internal structure, hardwiring default KSPHPDDM options\n"));
    if (data->cntl[0] == static_cast<char>(PETSC_DECIDE))
      data->cntl[0] = 0; /* GMRES by default */
    if (data->cntl[0] != HPDDM_KRYLOV_METHOD_NONE) { /* following options do not matter with PREONLY */
      if (data->cntl[0] != HPDDM_KRYLOV_METHOD_BCG && data->cntl[0] != HPDDM_KRYLOV_METHOD_BFBCG) {
        data->cntl[1] = HPDDM_VARIANT_LEFT; /* left preconditioning by default */
        if (ksp->pc_side_set == PC_RIGHT) data->cntl[1] = HPDDM_VARIANT_RIGHT;
        else PetscCheck(ksp->pc_side_set != PC_SYMMETRIC, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Symmetric preconditioning not implemented");
        if (data->cntl[1] > 0) PetscCall(KSPSetPCSide(ksp, PC_RIGHT));
      }
      if (data->cntl[0] == HPDDM_KRYLOV_METHOD_BGMRES || data->cntl[0] == HPDDM_KRYLOV_METHOD_BGCRODR || data->cntl[0] == HPDDM_KRYLOV_METHOD_BFBCG) {
        data->rcntl[0] = -1.0; /* no deflation by default */
        data->scntl[data->cntl[0] != HPDDM_KRYLOV_METHOD_BFBCG] = 1; /* Krylov subspace not enlarged by default */
      } else data->scntl[data->cntl[0] != HPDDM_KRYLOV_METHOD_BCG] = 0;
      if (data->cntl[0] == HPDDM_KRYLOV_METHOD_GMRES || data->cntl[0] == HPDDM_KRYLOV_METHOD_BGMRES || data->cntl[0] == HPDDM_KRYLOV_METHOD_GCRODR || data->cntl[0] == HPDDM_KRYLOV_METHOD_BGCRODR) {
        data->cntl[2] = static_cast<char>(HPDDM_ORTHOGONALIZATION_CGS) + (static_cast<char>(HPDDM_QR_CHOLQR) << 2); /* CGS and CholQR by default */
        data->scntl[0] = PetscMin(30, ksp->max_it); /* restart parameter of 30 by default */
      }
      if (data->cntl[0] == HPDDM_KRYLOV_METHOD_BCG || data->cntl[0] == HPDDM_KRYLOV_METHOD_BFBCG) {
        data->cntl[1] = HPDDM_QR_CHOLQR; /* CholQR by default */
      }
      if (data->cntl[0] == HPDDM_KRYLOV_METHOD_GCRODR || data->cntl[0] == HPDDM_KRYLOV_METHOD_BGCRODR) {
        data->icntl[0] = PetscMin(20, data->scntl[0] - 1); /* recycled subspace of size 20 by default */
        if (!PetscDefined(HAVE_SLEPC) || !PetscDefined(USE_SHARED_LIBRARIES) || data->cntl[0] == HPDDM_KRYLOV_METHOD_GCRODR) {
          data->cntl[3] = HPDDM_RECYCLE_TARGET_SM; /* default recycling target */
        } else {
          data->cntl[3] = 1; /* redistribution parameter of 1 by default */
        }
        data->cntl[4] = HPDDM_RECYCLE_STRATEGY_A; /* default recycling strategy */
      }
    } else data->scntl[1] = 1;
  }
  PetscCheck(ksp->nmax >= std::numeric_limits<int>::min() && ksp->nmax <= std::numeric_limits<int>::max(), PetscObjectComm((PetscObject)ksp), PETSC_ERR_ARG_OUTOFRANGE, "KSPMatSolve() block size %" PetscInt_FMT " not representable by an integer, which is not handled by KSPHPDDM", ksp->nmax);
  else data->icntl[1] = static_cast<int>(ksp->nmax);
  PetscFunctionReturn(0);
}

static inline PetscErrorCode KSPHPDDMReset_Private(KSP ksp)
{
  KSP_HPDDM *data = (KSP_HPDDM*)ksp->data;

  PetscFunctionBegin;
  /* cast PETSC_DECIDE into the appropriate types to avoid compiler warnings */
  std::fill_n(data->rcntl, PETSC_STATIC_ARRAY_LENGTH(data->rcntl), static_cast<PetscReal>(PETSC_DECIDE));
  std::fill_n(data->icntl, PETSC_STATIC_ARRAY_LENGTH(data->icntl), static_cast<int>(PETSC_DECIDE));
  std::fill_n(data->scntl, PETSC_STATIC_ARRAY_LENGTH(data->scntl), static_cast<unsigned short>(PETSC_DECIDE));
  std::fill_n(data->cntl , PETSC_STATIC_ARRAY_LENGTH(data->cntl) , static_cast<char>(PETSC_DECIDE));
  data->precision = PETSC_KSPHPDDM_DEFAULT_PRECISION;
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPReset_HPDDM(KSP ksp)
{
  KSP_HPDDM *data = (KSP_HPDDM*)ksp->data;

  PetscFunctionBegin;
  if (data->op) {
    delete data->op;
    data->op = NULL;
  }
  PetscCall(KSPHPDDMReset_Private(ksp));
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPDestroy_HPDDM(KSP ksp)
{
  PetscFunctionBegin;
  PetscCall(KSPReset_HPDDM(ksp));
  PetscCall(KSPDestroyDefault(ksp));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPHPDDMSetDeflationSpace_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPHPDDMGetDeflationSpace_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPHPDDMSetType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPHPDDMGetType_C", NULL));
  PetscFunctionReturn(0);
}

static inline PetscErrorCode KSPSolve_HPDDM_Private(KSP ksp, const PetscScalar *b, PetscScalar *x, PetscInt n)
{
  KSP_HPDDM                           *data = (KSP_HPDDM*)ksp->data;
  KSPConvergedDefaultCtx              *ctx = (KSPConvergedDefaultCtx*)ksp->cnvP;
  HPDDM::upscaled_type<PetscScalar>   *dbl[2];
  HPDDM::downscaled_type<PetscScalar> *sgl[2];
  const PetscInt                      N = data->op->getDof() * n;
  PetscBool                           scale;

  PetscFunctionBegin;
  PetscCall(PCGetDiagonalScale(ksp->pc, &scale));
  PetscCheck(!scale, PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP, "Krylov method %s does not support diagonal scaling", ((PetscObject)ksp)->type_name);
  if (n > 1) {
    if (ksp->converged == KSPConvergedDefault) {
      PetscCheck(!ctx->mininitialrtol, PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP, "Krylov method %s does not support KSPConvergedDefaultSetUMIRNorm()", ((PetscObject)ksp)->type_name);
      if (!ctx->initialrtol) {
        PetscCall(PetscInfo(ksp, "Forcing KSPConvergedDefaultSetUIRNorm() since KSPConvergedDefault() cannot handle multiple norms\n"));
        ctx->initialrtol = PETSC_TRUE;
      }
    } else PetscCall(PetscInfo(ksp, "Using a special \"converged\" callback, be careful, it is used in KSPHPDDM to track blocks of residuals\n"));
  }
  /* initial guess is always nonzero with recycling methods if there is a deflation subspace available */
  if ((data->cntl[0] == HPDDM_KRYLOV_METHOD_GCRODR || data->cntl[0] == HPDDM_KRYLOV_METHOD_BGCRODR) && data->op->storage()) ksp->guess_zero = PETSC_FALSE;
  ksp->its = 0;
  ksp->reason = KSP_CONVERGED_ITERATING;
  if (data->precision == KSP_HPDDM_PRECISION_DOUBLE && PetscDefined(USE_REAL_SINGLE)) {
    PetscCall(PetscMalloc2(N, dbl, N, dbl + 1));
    std::copy_n(b, N, dbl[0]);
    std::copy_n(x, N, dbl[1]);
    PetscCall(static_cast<PetscErrorCode>(HPDDM::IterativeMethod::solve(*data->op, dbl[0], dbl[1], n, PetscObjectComm((PetscObject)ksp))));
    std::copy_n(dbl[1], N, x);
    PetscCall(PetscFree2(dbl[0], dbl[1]));
  } else if (data->precision == KSP_HPDDM_PRECISION_SINGLE && PetscDefined(USE_REAL_DOUBLE)) {
    PetscCall(PetscMalloc1(N, sgl));
    sgl[1] = reinterpret_cast<HPDDM::downscaled_type<PetscScalar>*>(x);
    std::copy_n(b, N, sgl[0]);
    for (PetscInt i = 0; i < N; ++i) sgl[1][i] = x[i];
    PetscCall(static_cast<PetscErrorCode>(HPDDM::IterativeMethod::solve(*data->op, sgl[0], sgl[1], n, PetscObjectComm((PetscObject)ksp))));
    if (N) {
      sgl[0][0] = sgl[1][0];
      std::copy_backward(sgl[1] + 1, sgl[1] + N, x + N);
      x[0] = sgl[0][0];
    }
    PetscCall(PetscFree(sgl[0]));
  } else PetscCall(static_cast<PetscErrorCode>(HPDDM::IterativeMethod::solve(*data->op, b, x, n, PetscObjectComm((PetscObject)ksp))));
  if (!ksp->reason) { /* KSPConvergedDefault() is still returning 0 (= KSP_CONVERGED_ITERATING) */
    if (ksp->its >= ksp->max_it) ksp->reason = KSP_DIVERGED_ITS;
    else ksp->reason = KSP_CONVERGED_RTOL; /* early exit by HPDDM, which only happens on breakdowns or convergence */
  }
  ksp->its = PetscMin(ksp->its, ksp->max_it);
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPSolve_HPDDM(KSP ksp)
{
  KSP_HPDDM         *data = (KSP_HPDDM*)ksp->data;
  Mat               A, B;
  PetscScalar       *x, *bt = NULL, **ptr;
  const PetscScalar *b;
  PetscInt          i, j, n;
  PetscBool         flg;

  PetscFunctionBegin;
  PetscCall(PetscCitationsRegister(HPDDMCitation, &HPDDMCite));
  PetscCall(KSPGetOperators(ksp, &A, NULL));
  PetscCall(PetscObjectTypeCompareAny((PetscObject)A, &flg, MATSEQKAIJ, MATMPIKAIJ, ""));
  PetscCall(VecGetArrayWrite(ksp->vec_sol, &x));
  PetscCall(VecGetArrayRead(ksp->vec_rhs, &b));
  if (!flg) PetscCall(KSPSolve_HPDDM_Private(ksp, b, x, 1));
  else {
    PetscCall(MatKAIJGetScaledIdentity(A, &flg));
    PetscCall(MatKAIJGetAIJ(A, &B));
    PetscCall(MatGetBlockSize(A, &n));
    PetscCall(MatGetLocalSize(B, &i, NULL));
    j = data->op->getDof();
    if (!flg) i *= n; /* S and T are not scaled identities, cannot use block methods */
    if (i != j) { /* switching between block and standard methods */
      delete data->op;
      data->op = new HPDDM::PETScOperator(ksp, i);
    }
    if (flg && n > 1) {
      PetscCall(PetscMalloc1(i * n, &bt));
      /* from row- to column-major to be consistent with HPDDM */
      HPDDM::Wrapper<PetscScalar>::omatcopy<'T'>(i, n, b, n, bt, i);
      ptr = const_cast<PetscScalar**>(&b);
      std::swap(*ptr, bt);
      HPDDM::Wrapper<PetscScalar>::imatcopy<'T'>(i, n, x, n, i);
    }
    PetscCall(KSPSolve_HPDDM_Private(ksp, b, x, flg ? n : 1));
    if (flg && n > 1) {
      std::swap(*ptr, bt);
      PetscCall(PetscFree(bt));
      /* from column- to row-major to be consistent with MatKAIJ format */
      HPDDM::Wrapper<PetscScalar>::imatcopy<'T'>(n, i, x, i, n);
    }
  }
  PetscCall(VecRestoreArrayRead(ksp->vec_rhs, &b));
  PetscCall(VecRestoreArrayWrite(ksp->vec_sol, &x));
  PetscFunctionReturn(0);
}

/*@
     KSPHPDDMSetDeflationSpace - Sets the deflation space used by Krylov methods with recycling. This space is viewed as a set of vectors stored in a MATDENSE (column major).

   Input Parameters:
+     ksp - iterative context
-     U - deflation space to be used during KSPSolve()

   Level: intermediate

.seealso:  KSPCreate(), KSPType (for list of available types), KSPHPDDMGetDeflationSpace()
@*/
PetscErrorCode KSPHPDDMSetDeflationSpace(KSP ksp, Mat U)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscValidHeaderSpecific(U, MAT_CLASSID, 2);
  PetscCheckSameComm(ksp, 1, U, 2);
  PetscUseMethod(ksp, "KSPHPDDMSetDeflationSpace_C", (KSP, Mat), (ksp, U));
  PetscFunctionReturn(0);
}

/*@
     KSPHPDDMGetDeflationSpace - Gets the deflation space computed by Krylov methods with recycling or NULL if KSPSolve() has not been called yet. This space is viewed as a set of vectors stored in a MATDENSE (column major). It is the responsibility of the user to free the returned Mat.

   Input Parameter:
.     ksp - iterative context

   Output Parameter:
.     U - deflation space generated during KSPSolve()

   Level: intermediate

.seealso:  KSPCreate(), KSPType (for list of available types), KSPHPDDMSetDeflationSpace()
@*/
PetscErrorCode KSPHPDDMGetDeflationSpace(KSP ksp, Mat *U)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  if (U) {
    PetscValidPointer(U, 2);
    PetscUseMethod(ksp, "KSPHPDDMGetDeflationSpace_C", (KSP, Mat*), (ksp, U));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPHPDDMSetDeflationSpace_HPDDM(KSP ksp, Mat U)
{
  KSP_HPDDM            *data = (KSP_HPDDM*)ksp->data;
  HPDDM::PETScOperator *op = data->op;
  Mat                  A;
  const PetscScalar    *array;
  PetscScalar          *copy;
  PetscInt             m1, M1, m2, M2, n2, N2, ldu;
  PetscBool            match;

  PetscFunctionBegin;
  if (!op) {
    PetscCall(KSPSetUp(ksp));
    op = data->op;
  }
  PetscCheck(data->precision == PETSC_KSPHPDDM_DEFAULT_PRECISION, PETSC_COMM_SELF, PETSC_ERR_SUP, "%s != %s", KSPHPDDMPrecisionTypes[data->precision], KSPHPDDMPrecisionTypes[PETSC_KSPHPDDM_DEFAULT_PRECISION]);
  PetscCall(KSPGetOperators(ksp, &A, NULL));
  PetscCall(MatGetLocalSize(A, &m1, NULL));
  PetscCall(MatGetLocalSize(U, &m2, &n2));
  PetscCall(MatGetSize(A, &M1, NULL));
  PetscCall(MatGetSize(U, &M2, &N2));
  PetscCheck(m1 == m2 && M1 == M2, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Cannot use a deflation space with (m2,M2) = (%" PetscInt_FMT ",%" PetscInt_FMT ") for a linear system with (m1,M1) = (%" PetscInt_FMT ",%" PetscInt_FMT ")", m2, M2, m1, M1);
  PetscCall(PetscObjectTypeCompareAny((PetscObject)U, &match, MATSEQDENSE, MATMPIDENSE, ""));
  PetscCheck(match, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Provided deflation space not stored in a dense Mat");
  PetscCall(MatDenseGetArrayRead(U, &array));
  copy = op->allocate(m2, 1, N2);
  PetscCheck(copy, PETSC_COMM_SELF, PETSC_ERR_POINTER, "Memory allocation error");
  PetscCall(MatDenseGetLDA(U, &ldu));
  HPDDM::Wrapper<PetscScalar>::omatcopy<'N'>(N2, m2, array, ldu, copy, m2);
  PetscCall(MatDenseRestoreArrayRead(U, &array));
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPHPDDMGetDeflationSpace_HPDDM(KSP ksp, Mat *U)
{
  KSP_HPDDM            *data = (KSP_HPDDM*)ksp->data;
  HPDDM::PETScOperator *op = data->op;
  Mat                  A;
  const PetscScalar    *array;
  PetscScalar          *copy;
  PetscInt             m1, M1, N2;

  PetscFunctionBegin;
  if (!op) {
    PetscCall(KSPSetUp(ksp));
    op = data->op;
  }
  PetscCheck(data->precision == PETSC_KSPHPDDM_DEFAULT_PRECISION, PETSC_COMM_SELF, PETSC_ERR_SUP, "%s != %s", KSPHPDDMPrecisionTypes[data->precision], KSPHPDDMPrecisionTypes[PETSC_KSPHPDDM_DEFAULT_PRECISION]);
  array = op->storage();
  N2 = op->k().first * op->k().second;
  if (!array) *U = NULL;
  else {
    PetscCall(KSPGetOperators(ksp, &A, NULL));
    PetscCall(MatGetLocalSize(A, &m1, NULL));
    PetscCall(MatGetSize(A, &M1, NULL));
    PetscCall(MatCreateDense(PetscObjectComm((PetscObject)ksp), m1, PETSC_DECIDE, M1, N2, NULL, U));
    PetscCall(MatDenseGetArrayWrite(*U, &copy));
    PetscCall(PetscArraycpy(copy, array, m1 * N2));
    PetscCall(MatDenseRestoreArrayWrite(*U, &copy));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPMatSolve_HPDDM(KSP ksp, Mat B, Mat X)
{
  KSP_HPDDM            *data = (KSP_HPDDM*)ksp->data;
  HPDDM::PETScOperator *op = data->op;
  Mat                  A;
  const PetscScalar    *b;
  PetscScalar          *x;
  PetscInt             n, lda;

  PetscFunctionBegin;
  PetscCall(PetscCitationsRegister(HPDDMCitation, &HPDDMCite));
  if (!op) {
    PetscCall(KSPSetUp(ksp));
    op = data->op;
  }
  PetscCall(KSPGetOperators(ksp, &A, NULL));
  PetscCall(MatGetLocalSize(B, &n, NULL));
  PetscCall(MatDenseGetLDA(B, &lda));
  PetscCheck(n == lda, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Unhandled leading dimension lda = %" PetscInt_FMT " with n = %" PetscInt_FMT, lda, n);
  PetscCall(MatGetLocalSize(A, &n, NULL));
  PetscCall(MatDenseGetLDA(X, &lda));
  PetscCheck(n == lda, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Unhandled leading dimension lda = %" PetscInt_FMT " with n = %" PetscInt_FMT, lda, n);
  PetscCall(MatDenseGetArrayRead(B, &b));
  PetscCall(MatDenseGetArrayWrite(X, &x));
  PetscCall(MatGetSize(X, NULL, &n));
  PetscCall(KSPSolve_HPDDM_Private(ksp, b, x, n));
  PetscCall(MatDenseRestoreArrayWrite(X, &x));
  PetscCall(MatDenseRestoreArrayRead(B, &b));
  PetscFunctionReturn(0);
}

/*@
     KSPHPDDMSetType - Sets the type of Krylov method used in KSPHPDDM.

   Input Parameters:
+     ksp - iterative context
-     type - any of gmres, bgmres, cg, bcg, gcrodr, bgcrodr, bfbcg, or preonly

   Level: intermediate

   Notes:
     Unlike KSPReset(), this function does not destroy any deflation space attached to the KSP.
     As an example, in the following sequence: KSPHPDDMSetType(ksp, KSPGCRODR); KSPSolve(ksp, b, x); KSPHPDDMSetType(ksp, KSPGMRES); KSPHPDDMSetType(ksp, KSPGCRODR); KSPSolve(ksp, b, x); the recycled space is reused in the second KSPSolve().

.seealso:  KSPCreate(), KSPType (for list of available types), KSPHPDDMType, KSPHPDDMGetType()
@*/
PetscErrorCode KSPHPDDMSetType(KSP ksp, KSPHPDDMType type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscUseMethod(ksp, "KSPHPDDMSetType_C", (KSP, KSPHPDDMType), (ksp, type));
  PetscFunctionReturn(0);
}

/*@
     KSPHPDDMGetType - Gets the type of Krylov method used in KSPHPDDM.

   Input Parameter:
.     ksp - iterative context

   Output Parameter:
.     type - any of gmres, bgmres, cg, bcg, gcrodr, bgcrodr, bfbcg, or preonly

   Level: intermediate

.seealso:  KSPCreate(), KSPType (for list of available types), KSPHPDDMType, KSPHPDDMSetType()
@*/
PetscErrorCode KSPHPDDMGetType(KSP ksp, KSPHPDDMType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  if (type) {
    PetscValidPointer(type, 2);
    PetscUseMethod(ksp, "KSPHPDDMGetType_C", (KSP, KSPHPDDMType*), (ksp, type));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPHPDDMSetType_HPDDM(KSP ksp, KSPHPDDMType type)
{
  KSP_HPDDM *data = (KSP_HPDDM*)ksp->data;
  PetscInt  i;
  PetscBool flg = PETSC_FALSE;

  PetscFunctionBegin;
  for (i = 0; i < static_cast<PetscInt>(PETSC_STATIC_ARRAY_LENGTH(KSPHPDDMTypes)); ++i) {
    PetscCall(PetscStrcmp(KSPHPDDMTypes[type], KSPHPDDMTypes[i], &flg));
    if (flg) break;
  }
  PetscCheck(i != PETSC_STATIC_ARRAY_LENGTH(KSPHPDDMTypes), PetscObjectComm((PetscObject)ksp), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown KSPHPDDMType %s", type);
  if (data->cntl[0] != static_cast<char>(PETSC_DECIDE) && data->cntl[0] != i) PetscCall(KSPHPDDMReset_Private(ksp));
  data->cntl[0] = i;
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPHPDDMGetType_HPDDM(KSP ksp, KSPHPDDMType *type)
{
  KSP_HPDDM *data = (KSP_HPDDM*)ksp->data;

  PetscFunctionBegin;
  PetscCheck(data->cntl[0] != static_cast<char>(PETSC_DECIDE), PETSC_COMM_SELF, PETSC_ERR_ORDER, "KSPHPDDMType not set yet");
  /* need to shift by -1 for HPDDM_KRYLOV_METHOD_NONE */
  *type = static_cast<KSPHPDDMType>(PetscMin(data->cntl[0], static_cast<char>(PETSC_STATIC_ARRAY_LENGTH(KSPHPDDMTypes) - 1)));
  PetscFunctionReturn(0);
}

/*MC
     KSPHPDDM - Interface with the HPDDM library.

   This KSP may be used to further select methods that are currently not implemented natively in PETSc, e.g., GCRODR [2006], a recycled Krylov method which is similar to KSPLGMRES, see [2016] for a comparison. ex75.c shows how to reproduce the results from the aforementioned paper [2006]. A chronological bibliography of relevant publications linked with KSP available in HPDDM through KSPHPDDM, and not available directly in PETSc, may be found below. The interface is explained in details in [2021].

   Options Database Keys:
+   -ksp_gmres_restart <restart, default=30> - see KSPGMRES
.   -ksp_hpddm_type <type, default=gmres> - any of gmres, bgmres, cg, bcg, gcrodr, bgcrodr, bfbcg, or preonly, see KSPHPDDMType
.   -ksp_hpddm_precision <value, default=same as PetscScalar> - any of single or double, see KSPHPDDMPrecision
.   -ksp_hpddm_deflation_tol <eps, default=\-1.0> - tolerance when deflating right-hand sides inside block methods (no deflation by default, only relevant with block methods)
.   -ksp_hpddm_enlarge_krylov_subspace <p, default=1> - split the initial right-hand side into multiple vectors (only relevant with nonblock methods)
.   -ksp_hpddm_orthogonalization <type, default=cgs> - any of cgs or mgs, see KSPGMRES
.   -ksp_hpddm_qr <type, default=cholqr> - distributed QR factorizations with any of cholqr, cgs, or mgs (only relevant with block methods)
.   -ksp_hpddm_variant <type, default=left> - any of left, right, or flexible (this option is superseded by KSPSetPCSide())
.   -ksp_hpddm_recycle <n, default=0> - number of harmonic Ritz vectors to compute (only relevant with GCRODR or BGCRODR)
.   -ksp_hpddm_recycle_target <type, default=SM> - criterion to select harmonic Ritz vectors using either SM, LM, SR, LR, SI, or LI (only relevant with GCRODR or BGCRODR). For BGCRODR, if PETSc is compiled with SLEPc, this option is not relevant, since SLEPc is used instead. Options are set with the prefix -ksp_hpddm_recycle_eps_
.   -ksp_hpddm_recycle_strategy <type, default=A> - generalized eigenvalue problem A or B to solve for recycling (only relevant with flexible GCRODR or BGCRODR)
-   -ksp_hpddm_recycle_symmetric <true, default=false> - symmetric generalized eigenproblems in BGCRODR, useful to switch to distributed solvers like EPSELEMENTAL or EPSSCALAPACK (only relevant when PETSc is compiled with SLEPc)

   References:
+   1980 - The block conjugate gradient algorithm and related methods. O'Leary. Linear Algebra and its Applications.
.   2006 - Recycling Krylov subspaces for sequences of linear systems. Parks, de Sturler, Mackey, Johnson, and Maiti. SIAM Journal on Scientific Computing
.   2013 - A modified block flexible GMRES method with deflation at each iteration for the solution of non-Hermitian linear systems with multiple right-hand sides. Calandra, Gratton, Lago, Vasseur, and Carvalho. SIAM Journal on Scientific Computing.
.   2016 - Block iterative methods and recycling for improved scalability of linear solvers. Jolivet and Tournier. SC16.
.   2017 - A breakdown-free block conjugate gradient method. Ji and Li. BIT Numerical Mathematics.
-   2021 - KSPHPDDM and PCHPDDM: extending PETSc with advanced Krylov methods and robust multilevel overlapping Schwarz preconditioners. Jolivet, Roman, and Zampini. Computer & Mathematics with Applications.

   Level: intermediate

.seealso:  KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP, KSPGMRES, KSPCG, KSPLGMRES, KSPDGMRES
M*/
PETSC_EXTERN PetscErrorCode KSPCreate_HPDDM(KSP ksp)
{
  KSP_HPDDM  *data;
  PetscInt   i;
  const char *common[] = { KSPGMRES, KSPCG, KSPPREONLY };
  PetscBool  flg = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(PetscNewLog(ksp, &data));
  ksp->data = (void*)data;
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_PRECONDITIONED, PC_LEFT, 2));
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_UNPRECONDITIONED, PC_RIGHT, 1));
  ksp->ops->solve          = KSPSolve_HPDDM;
  ksp->ops->matsolve       = KSPMatSolve_HPDDM;
  ksp->ops->setup          = KSPSetUp_HPDDM;
  ksp->ops->setfromoptions = KSPSetFromOptions_HPDDM;
  ksp->ops->destroy        = KSPDestroy_HPDDM;
  ksp->ops->view           = KSPView_HPDDM;
  ksp->ops->reset          = KSPReset_HPDDM;
  PetscCall(KSPHPDDMReset_Private(ksp));
  for (i = 0; i < static_cast<PetscInt>(PETSC_STATIC_ARRAY_LENGTH(common)); ++i) {
    PetscCall(PetscStrcmp(((PetscObject)ksp)->type_name, common[i], &flg));
    if (flg) break;
  }
  if (!i) data->cntl[0] = HPDDM_KRYLOV_METHOD_GMRES;
  else if (i == 1) data->cntl[0] = HPDDM_KRYLOV_METHOD_CG;
  else if (i == 2) data->cntl[0] = HPDDM_KRYLOV_METHOD_NONE;
  if (data->cntl[0] != static_cast<char>(PETSC_DECIDE)) PetscCall(PetscInfo(ksp, "Using the previously set KSPType %s\n", common[i]));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPHPDDMSetDeflationSpace_C", KSPHPDDMSetDeflationSpace_HPDDM));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPHPDDMGetDeflationSpace_C", KSPHPDDMGetDeflationSpace_HPDDM));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPHPDDMSetType_C", KSPHPDDMSetType_HPDDM));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPHPDDMGetType_C", KSPHPDDMGetType_HPDDM));
#if defined(PETSC_HAVE_SLEPC) && defined(PETSC_USE_SHARED_LIBRARIES)
  if (!loadedDL) PetscCall(HPDDMLoadDL_Private(&loadedDL));
#endif
  data->precision = PETSC_KSPHPDDM_DEFAULT_PRECISION;
  PetscFunctionReturn(0);
}

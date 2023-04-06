#define HPDDM_MIXED_PRECISION 1
#include <petsc/private/petschpddm.h> /*I "petscksp.h" I*/

const char *const KSPHPDDMTypes[]          = {KSPGMRES, "bgmres", KSPCG, "bcg", "gcrodr", "bgcrodr", "bfbcg", KSPPREONLY};
const char *const KSPHPDDMPrecisionTypes[] = {"HALF", "SINGLE", "DOUBLE", "QUADRUPLE", "KSPHPDDMPrecisionType", "KSP_HPDDM_PRECISION_", nullptr};
const char *const HPDDMOrthogonalization[] = {"cgs", "mgs"};
const char *const HPDDMQR[]                = {"cholqr", "cgs", "mgs"};
const char *const HPDDMVariant[]           = {"left", "right", "flexible"};
const char *const HPDDMRecycleTarget[]     = {"SM", "LM", "SR", "LR", "SI", "LI"};
const char *const HPDDMRecycleStrategy[]   = {"A", "B"};

PetscBool  HPDDMCite       = PETSC_FALSE;
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

#if PetscDefined(HAVE_SLEPC) && PetscDefined(HAVE_DYNAMIC_LIBRARIES) && PetscDefined(USE_SHARED_LIBRARIES)
static PetscBool loadedDL = PETSC_FALSE;
#endif

static PetscErrorCode KSPSetFromOptions_HPDDM(KSP ksp, PetscOptionItems *PetscOptionsObject)
{
  KSP_HPDDM  *data = (KSP_HPDDM *)ksp->data;
  PetscInt    i, j;
  PetscMPIInt size;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "KSPHPDDM options, cf. https://github.com/hpddm/hpddm");
  i = (data->cntl[0] == static_cast<char>(PETSC_DECIDE) ? HPDDM_KRYLOV_METHOD_GMRES : data->cntl[0]);
  PetscCall(PetscOptionsEList("-ksp_hpddm_type", "Type of Krylov method", "KSPHPDDMGetType", KSPHPDDMTypes, PETSC_STATIC_ARRAY_LENGTH(KSPHPDDMTypes), KSPHPDDMTypes[HPDDM_KRYLOV_METHOD_GMRES], &i, nullptr));
  if (i == PETSC_STATIC_ARRAY_LENGTH(KSPHPDDMTypes) - 1) i = HPDDM_KRYLOV_METHOD_NONE; /* need to shift the value since HPDDM_KRYLOV_METHOD_RICHARDSON is not registered in PETSc */
  data->cntl[0] = i;
  PetscCall(PetscOptionsEnum("-ksp_hpddm_precision", "Precision in which Krylov bases are stored", "KSPHPDDM", KSPHPDDMPrecisionTypes, (PetscEnum)data->precision, (PetscEnum *)&data->precision, nullptr));
  PetscCheck(data->precision != KSP_HPDDM_PRECISION_QUADRUPLE || PetscDefined(HAVE_REAL___FLOAT128), PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP_SYS, "Unsupported %s precision", KSPHPDDMPrecisionTypes[data->precision]);
  PetscCheck(std::abs(data->precision - PETSC_KSPHPDDM_DEFAULT_PRECISION) <= 1, PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP, "Unhandled mixed %s and %s precisions", KSPHPDDMPrecisionTypes[data->precision], KSPHPDDMPrecisionTypes[PETSC_KSPHPDDM_DEFAULT_PRECISION]);
  if (data->cntl[0] != HPDDM_KRYLOV_METHOD_NONE) {
    if (data->cntl[0] != HPDDM_KRYLOV_METHOD_BCG && data->cntl[0] != HPDDM_KRYLOV_METHOD_BFBCG) {
      i = (data->cntl[1] == static_cast<char>(PETSC_DECIDE) ? HPDDM_VARIANT_LEFT : data->cntl[1]);
      if (ksp->pc_side_set == PC_SIDE_DEFAULT)
        PetscCall(PetscOptionsEList("-ksp_hpddm_variant", "Left, right, or variable preconditioning", "KSPHPDDM", HPDDMVariant, PETSC_STATIC_ARRAY_LENGTH(HPDDMVariant), HPDDMVariant[HPDDM_VARIANT_LEFT], &i, nullptr));
      else if (ksp->pc_side_set == PC_RIGHT) i = HPDDM_VARIANT_RIGHT;
      data->cntl[1] = i;
      if (i > 0) PetscCall(KSPSetPCSide(ksp, PC_RIGHT));
    }
    if (data->cntl[0] == HPDDM_KRYLOV_METHOD_BGMRES || data->cntl[0] == HPDDM_KRYLOV_METHOD_BGCRODR || data->cntl[0] == HPDDM_KRYLOV_METHOD_BFBCG) {
      data->rcntl[0] = (PetscAbsReal(data->rcntl[0] - static_cast<PetscReal>(PETSC_DECIDE)) < PETSC_SMALL ? -1.0 : data->rcntl[0]);
      PetscCall(PetscOptionsReal("-ksp_hpddm_deflation_tol", "Tolerance when deflating right-hand sides inside block methods", "KSPHPDDM", data->rcntl[0], data->rcntl, nullptr));
      i = (data->scntl[data->cntl[0] != HPDDM_KRYLOV_METHOD_BFBCG] == static_cast<unsigned short>(PETSC_DECIDE) ? 1 : PetscMax(1, data->scntl[data->cntl[0] != HPDDM_KRYLOV_METHOD_BFBCG]));
      PetscCall(PetscOptionsRangeInt("-ksp_hpddm_enlarge_krylov_subspace", "Split the initial right-hand side into multiple vectors", "KSPHPDDM", i, &i, nullptr, 1, std::numeric_limits<unsigned short>::max() - 1));
      data->scntl[data->cntl[0] != HPDDM_KRYLOV_METHOD_BFBCG] = i;
    } else data->scntl[data->cntl[0] != HPDDM_KRYLOV_METHOD_BCG] = 0;
    if (data->cntl[0] == HPDDM_KRYLOV_METHOD_GMRES || data->cntl[0] == HPDDM_KRYLOV_METHOD_BGMRES || data->cntl[0] == HPDDM_KRYLOV_METHOD_GCRODR || data->cntl[0] == HPDDM_KRYLOV_METHOD_BGCRODR) {
      i = (data->cntl[2] == static_cast<char>(PETSC_DECIDE) ? HPDDM_ORTHOGONALIZATION_CGS : data->cntl[2] & 3);
      PetscCall(PetscOptionsEList("-ksp_hpddm_orthogonalization", "Classical (faster) or Modified (more robust) Gram--Schmidt process", "KSPHPDDM", HPDDMOrthogonalization, PETSC_STATIC_ARRAY_LENGTH(HPDDMOrthogonalization), HPDDMOrthogonalization[HPDDM_ORTHOGONALIZATION_CGS], &i, nullptr));
      j = (data->cntl[2] == static_cast<char>(PETSC_DECIDE) ? HPDDM_QR_CHOLQR : ((data->cntl[2] >> 2) & 7));
      PetscCall(PetscOptionsEList("-ksp_hpddm_qr", "Distributed QR factorizations computed with Cholesky QR, Classical or Modified Gram--Schmidt process", "KSPHPDDM", HPDDMQR, PETSC_STATIC_ARRAY_LENGTH(HPDDMQR), HPDDMQR[HPDDM_QR_CHOLQR], &j, nullptr));
      data->cntl[2] = static_cast<char>(i) + (static_cast<char>(j) << 2);
      i             = (data->scntl[0] == static_cast<unsigned short>(PETSC_DECIDE) ? PetscMin(30, ksp->max_it) : data->scntl[0]);
      PetscCall(PetscOptionsRangeInt("-ksp_gmres_restart", "Maximum number of Arnoldi vectors generated per cycle", "KSPHPDDM", i, &i, nullptr, PetscMin(1, ksp->max_it), PetscMin(ksp->max_it, std::numeric_limits<unsigned short>::max() - 1)));
      data->scntl[0] = i;
    }
    if (data->cntl[0] == HPDDM_KRYLOV_METHOD_BCG || data->cntl[0] == HPDDM_KRYLOV_METHOD_BFBCG) {
      j = (data->cntl[1] == static_cast<char>(PETSC_DECIDE) ? HPDDM_QR_CHOLQR : data->cntl[1]);
      PetscCall(PetscOptionsEList("-ksp_hpddm_qr", "Distributed QR factorizations computed with Cholesky QR, Classical or Modified Gram--Schmidt process", "KSPHPDDM", HPDDMQR, PETSC_STATIC_ARRAY_LENGTH(HPDDMQR), HPDDMQR[HPDDM_QR_CHOLQR], &j, nullptr));
      data->cntl[1] = j;
    }
    if (data->cntl[0] == HPDDM_KRYLOV_METHOD_GCRODR || data->cntl[0] == HPDDM_KRYLOV_METHOD_BGCRODR) {
      i = (data->icntl[0] == static_cast<int>(PETSC_DECIDE) ? PetscMin(20, data->scntl[0] - 1) : data->icntl[0]);
      PetscCall(PetscOptionsRangeInt("-ksp_hpddm_recycle", "Number of harmonic Ritz vectors to compute", "KSPHPDDM", i, &i, nullptr, 1, data->scntl[0] - 1));
      data->icntl[0] = i;
      if (!PetscDefined(HAVE_SLEPC) || !PetscDefined(USE_SHARED_LIBRARIES) || data->cntl[0] == HPDDM_KRYLOV_METHOD_GCRODR) {
        i = (data->cntl[3] == static_cast<char>(PETSC_DECIDE) ? HPDDM_RECYCLE_TARGET_SM : data->cntl[3]);
        PetscCall(PetscOptionsEList("-ksp_hpddm_recycle_target", "Criterion to select harmonic Ritz vectors", "KSPHPDDM", HPDDMRecycleTarget, PETSC_STATIC_ARRAY_LENGTH(HPDDMRecycleTarget), HPDDMRecycleTarget[HPDDM_RECYCLE_TARGET_SM], &i, nullptr));
        data->cntl[3] = i;
      } else {
        PetscCheck(data->precision == PETSC_KSPHPDDM_DEFAULT_PRECISION, PetscObjectComm((PetscObject)ksp), PETSC_ERR_ARG_INCOMP, "Cannot use SLEPc with a different precision than PETSc for harmonic Ritz eigensolves");
        PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)ksp), &size));
        i = (data->cntl[3] == static_cast<char>(PETSC_DECIDE) ? 1 : data->cntl[3]);
        PetscCall(PetscOptionsRangeInt("-ksp_hpddm_recycle_redistribute", "Number of processes used to solve eigenvalue problems when recycling in BGCRODR", "KSPHPDDM", i, &i, nullptr, 1, PetscMin(size, 192)));
        data->cntl[3] = i;
      }
      i = (data->cntl[4] == static_cast<char>(PETSC_DECIDE) ? HPDDM_RECYCLE_STRATEGY_A : data->cntl[4]);
      PetscCall(PetscOptionsEList("-ksp_hpddm_recycle_strategy", "Generalized eigenvalue problem to solve for recycling", "KSPHPDDM", HPDDMRecycleStrategy, PETSC_STATIC_ARRAY_LENGTH(HPDDMRecycleStrategy), HPDDMRecycleStrategy[HPDDM_RECYCLE_STRATEGY_A], &i, nullptr));
      data->cntl[4] = i;
    }
  } else {
    data->cntl[0]  = HPDDM_KRYLOV_METHOD_NONE;
    data->scntl[1] = 1;
  }
  PetscCheck(ksp->nmax >= std::numeric_limits<int>::min() && ksp->nmax <= std::numeric_limits<int>::max(), PetscObjectComm((PetscObject)ksp), PETSC_ERR_ARG_OUTOFRANGE, "KSPMatSolve() block size %" PetscInt_FMT " not representable by an integer, which is not handled by KSPHPDDM",
             ksp->nmax);
  data->icntl[1] = static_cast<int>(ksp->nmax);
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPView_HPDDM(KSP ksp, PetscViewer viewer)
{
  KSP_HPDDM            *data  = (KSP_HPDDM *)ksp->data;
  HPDDM::PETScOperator *op    = data->op;
  const PetscScalar    *array = op ? op->storage() : nullptr;
  PetscBool             ascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &ascii));
  if (op && ascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "HPDDM type: %s\n", KSPHPDDMTypes[std::min(static_cast<PetscInt>(data->cntl[0]), static_cast<PetscInt>(PETSC_STATIC_ARRAY_LENGTH(KSPHPDDMTypes) - 1))]));
    PetscCall(PetscViewerASCIIPrintf(viewer, "precision: %s\n", KSPHPDDMPrecisionTypes[data->precision]));
    if (data->cntl[0] == HPDDM_KRYLOV_METHOD_BGMRES || data->cntl[0] == HPDDM_KRYLOV_METHOD_BGCRODR || data->cntl[0] == HPDDM_KRYLOV_METHOD_BFBCG) {
      if (PetscAbsReal(data->rcntl[0] - static_cast<PetscReal>(PETSC_DECIDE)) < PETSC_SMALL) PetscCall(PetscViewerASCIIPrintf(viewer, "no deflation at restarts\n"));
      else PetscCall(PetscViewerASCIIPrintf(viewer, "deflation tolerance: %g\n", static_cast<double>(data->rcntl[0])));
    }
    if (data->cntl[0] == HPDDM_KRYLOV_METHOD_GCRODR || data->cntl[0] == HPDDM_KRYLOV_METHOD_BGCRODR) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "deflation subspace attached? %s\n", PetscBools[array ? PETSC_TRUE : PETSC_FALSE]));
      if (!PetscDefined(HAVE_SLEPC) || !PetscDefined(USE_SHARED_LIBRARIES) || data->cntl[0] == HPDDM_KRYLOV_METHOD_GCRODR) PetscCall(PetscViewerASCIIPrintf(viewer, "deflation target: %s\n", HPDDMRecycleTarget[static_cast<PetscInt>(data->cntl[3])]));
      else PetscCall(PetscViewerASCIIPrintf(viewer, "redistribution size: %d\n", static_cast<PetscMPIInt>(data->cntl[3])));
    }
    if (data->icntl[1] != static_cast<int>(PETSC_DECIDE)) PetscCall(PetscViewerASCIIPrintf(viewer, "  block size is %d\n", data->icntl[1]));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPSetUp_HPDDM(KSP ksp)
{
  KSP_HPDDM *data = (KSP_HPDDM *)ksp->data;
  Mat        A;
  PetscInt   n, bs;
  PetscBool  match;

  PetscFunctionBegin;
  PetscCall(KSPGetOperators(ksp, &A, nullptr));
  PetscCall(MatGetLocalSize(A, &n, nullptr));
  PetscCall(MatGetBlockSize(A, &bs));
  PetscCall(PetscObjectTypeCompareAny((PetscObject)A, &match, MATSEQKAIJ, MATMPIKAIJ, ""));
  if (match) n /= bs;
  data->op = new HPDDM::PETScOperator(ksp, n);
  if (PetscUnlikely(!ksp->setfromoptionscalled || data->cntl[0] == static_cast<char>(PETSC_DECIDE))) { /* what follows is basically a copy/paste of KSPSetFromOptions_HPDDM, with no call to PetscOptions() */
    PetscCall(PetscInfo(ksp, "KSPSetFromOptions() not called or uninitialized internal structure, hardwiring default KSPHPDDM options\n"));
    if (data->cntl[0] == static_cast<char>(PETSC_DECIDE)) data->cntl[0] = 0; /* GMRES by default */
    if (data->cntl[0] != HPDDM_KRYLOV_METHOD_NONE) {                         /* following options do not matter with PREONLY */
      if (data->cntl[0] != HPDDM_KRYLOV_METHOD_BCG && data->cntl[0] != HPDDM_KRYLOV_METHOD_BFBCG) {
        data->cntl[1] = HPDDM_VARIANT_LEFT; /* left preconditioning by default */
        if (ksp->pc_side_set == PC_RIGHT) data->cntl[1] = HPDDM_VARIANT_RIGHT;
        if (data->cntl[1] > 0) PetscCall(KSPSetPCSide(ksp, PC_RIGHT));
      }
      if (data->cntl[0] == HPDDM_KRYLOV_METHOD_BGMRES || data->cntl[0] == HPDDM_KRYLOV_METHOD_BGCRODR || data->cntl[0] == HPDDM_KRYLOV_METHOD_BFBCG) {
        data->rcntl[0]                                          = -1.0; /* no deflation by default */
        data->scntl[data->cntl[0] != HPDDM_KRYLOV_METHOD_BFBCG] = 1;    /* Krylov subspace not enlarged by default */
      } else data->scntl[data->cntl[0] != HPDDM_KRYLOV_METHOD_BCG] = 0;
      if (data->cntl[0] == HPDDM_KRYLOV_METHOD_GMRES || data->cntl[0] == HPDDM_KRYLOV_METHOD_BGMRES || data->cntl[0] == HPDDM_KRYLOV_METHOD_GCRODR || data->cntl[0] == HPDDM_KRYLOV_METHOD_BGCRODR) {
        data->cntl[2]  = static_cast<char>(HPDDM_ORTHOGONALIZATION_CGS) + (static_cast<char>(HPDDM_QR_CHOLQR) << 2); /* CGS and CholQR by default */
        data->scntl[0] = PetscMin(30, ksp->max_it);                                                                  /* restart parameter of 30 by default */
      }
      if (data->cntl[0] == HPDDM_KRYLOV_METHOD_BCG || data->cntl[0] == HPDDM_KRYLOV_METHOD_BFBCG) { data->cntl[1] = HPDDM_QR_CHOLQR; /* CholQR by default */ }
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
  PetscCheck(ksp->nmax >= std::numeric_limits<int>::min() && ksp->nmax <= std::numeric_limits<int>::max(), PetscObjectComm((PetscObject)ksp), PETSC_ERR_ARG_OUTOFRANGE, "KSPMatSolve() block size %" PetscInt_FMT " not representable by an integer, which is not handled by KSPHPDDM",
             ksp->nmax);
  data->icntl[1] = static_cast<int>(ksp->nmax);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode KSPReset_HPDDM_Private(KSP ksp)
{
  KSP_HPDDM *data = (KSP_HPDDM *)ksp->data;

  PetscFunctionBegin;
  /* cast PETSC_DECIDE into the appropriate types to avoid compiler warnings */
  std::fill_n(data->rcntl, PETSC_STATIC_ARRAY_LENGTH(data->rcntl), static_cast<PetscReal>(PETSC_DECIDE));
  std::fill_n(data->icntl, PETSC_STATIC_ARRAY_LENGTH(data->icntl), static_cast<int>(PETSC_DECIDE));
  std::fill_n(data->scntl, PETSC_STATIC_ARRAY_LENGTH(data->scntl), static_cast<unsigned short>(PETSC_DECIDE));
  std::fill_n(data->cntl, PETSC_STATIC_ARRAY_LENGTH(data->cntl), static_cast<char>(PETSC_DECIDE));
  data->precision = PETSC_KSPHPDDM_DEFAULT_PRECISION;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPReset_HPDDM(KSP ksp)
{
  KSP_HPDDM *data = (KSP_HPDDM *)ksp->data;

  PetscFunctionBegin;
  delete data->op;
  data->op = nullptr;
  PetscCall(KSPReset_HPDDM_Private(ksp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPDestroy_HPDDM(KSP ksp)
{
  PetscFunctionBegin;
  PetscCall(KSPReset_HPDDM(ksp));
  PetscCall(KSPDestroyDefault(ksp));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPHPDDMSetDeflationMat_C", nullptr));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPHPDDMGetDeflationMat_C", nullptr));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPHPDDMSetType_C", nullptr));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPHPDDMGetType_C", nullptr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <PetscMemType type = PETSC_MEMTYPE_HOST>
static inline PetscErrorCode KSPSolve_HPDDM_Private(KSP ksp, const PetscScalar *b, PetscScalar *x, PetscInt n)
{
  KSP_HPDDM              *data = (KSP_HPDDM *)ksp->data;
  KSPConvergedDefaultCtx *ctx  = (KSPConvergedDefaultCtx *)ksp->cnvP;
  const PetscInt          N    = data->op->getDof() * n;
  PetscBool               flg;
#if !PetscDefined(USE_REAL_DOUBLE) || PetscDefined(HAVE_F2CBLASLAPACK___FLOAT128_BINDINGS)
  HPDDM::upscaled_type<PetscScalar> *high[2];
#endif
#if !PetscDefined(USE_REAL_SINGLE) || PetscDefined(HAVE_F2CBLASLAPACK___FP16_BINDINGS)
  typedef HPDDM::downscaled_type<PetscReal> PetscDownscaledReal PETSC_ATTRIBUTE_MAY_ALIAS;
  #if !PetscDefined(USE_COMPLEX)
  PetscDownscaledReal *low[2];
  #else
  typedef PetscReal PetscAliasedReal   PETSC_ATTRIBUTE_MAY_ALIAS;
  HPDDM::downscaled_type<PetscScalar> *low[2];
  PetscAliasedReal                    *x_r;
  PetscDownscaledReal                 *low_r;
  #endif
#endif
#if PetscDefined(HAVE_CUDA)
  Mat     A;
  VecType vtype;
#endif

  PetscFunctionBegin;
#if PetscDefined(HAVE_CUDA)
  PetscCall(KSPGetOperators(ksp, &A, nullptr));
  PetscCall(MatGetVecType(A, &vtype));
  std::initializer_list<std::string>                 list = {VECCUDA, VECSEQCUDA, VECMPICUDA};
  std::initializer_list<std::string>::const_iterator it   = std::find(list.begin(), list.end(), std::string(vtype));
  PetscCheck(type != PETSC_MEMTYPE_HOST || it == list.end(), PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP, "MatGetVecType() must return a Vec with the same PetscMemType as the right-hand side and solution, PetscMemType(%s) != %s", vtype, PetscMemTypeToString(type));
#endif
  PetscCall(PCGetDiagonalScale(ksp->pc, &flg));
  PetscCheck(!flg, PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP, "Krylov method %s does not support diagonal scaling", ((PetscObject)ksp)->type_name);
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
  ksp->its    = 0;
  ksp->reason = KSP_CONVERGED_ITERATING;
  if (data->precision > PETSC_KSPHPDDM_DEFAULT_PRECISION) { /* Krylov basis stored in higher precision than PetscScalar */
#if !PetscDefined(USE_REAL_DOUBLE) || PetscDefined(HAVE_F2CBLASLAPACK___FLOAT128_BINDINGS)
    if (type == PETSC_MEMTYPE_HOST) {
      PetscCall(PetscMalloc2(N, high, N, high + 1));
      HPDDM::copy_n(b, N, high[0]);
      HPDDM::copy_n(x, N, high[1]);
      PetscCall(HPDDM::IterativeMethod::solve(*data->op, high[0], high[1], n, PetscObjectComm((PetscObject)ksp)));
      HPDDM::copy_n(high[1], N, x);
      PetscCall(PetscFree2(high[0], high[1]));
    } else {
      PetscCheck(PetscDefined(HAVE_CUDA) && PetscDefined(USE_REAL_SINGLE), PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP, "CUDA in PETSc has no support for precisions other than single or double");
  #if PetscDefined(HAVE_CUDA)
    #if PetscDefined(HAVE_HPDDM)
      PetscCall(KSPSolve_HPDDM_CUDA_Private(data, b, x, n, PetscObjectComm((PetscObject)ksp)));
    #else
      SETERRQ(PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP, "No CUDA support with --download-hpddm from SLEPc");
    #endif
  #endif
    }
#else
    PetscCheck(data->precision != KSP_HPDDM_PRECISION_QUADRUPLE, PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP, "Reconfigure with --download-f2cblaslapack --with-f2cblaslapack-float128-bindings");
#endif
  } else if (data->precision < PETSC_KSPHPDDM_DEFAULT_PRECISION) { /* Krylov basis stored in lower precision than PetscScalar */
#if !PetscDefined(USE_REAL_SINGLE) || PetscDefined(HAVE_F2CBLASLAPACK___FP16_BINDINGS)
    if (type == PETSC_MEMTYPE_HOST) {
      PetscCall(PetscMalloc1(N, low));
  #if !PetscDefined(USE_COMPLEX)
      low[1] = reinterpret_cast<PetscDownscaledReal *>(x);
  #else
      low[1] = reinterpret_cast<HPDDM::downscaled_type<PetscScalar> *>(x);
  #endif
      std::copy_n(b, N, low[0]);
      for (PetscInt i = 0; i < N; ++i) low[1][i] = x[i];
      PetscCall(HPDDM::IterativeMethod::solve(*data->op, low[0], low[1], n, PetscObjectComm((PetscObject)ksp)));
  #if !PetscDefined(USE_COMPLEX)
      for (PetscInt i = N; i-- > 0;) x[i] = low[1][i];
  #else
      x_r = reinterpret_cast<PetscAliasedReal *>(x), low_r = reinterpret_cast<PetscDownscaledReal *>(x_r);
      for (PetscInt i = 2 * N; i-- > 0;) x_r[i] = low_r[i];
  #endif
      PetscCall(PetscFree(low[0]));
    } else {
      PetscCheck(PetscDefined(HAVE_CUDA) && PetscDefined(USE_REAL_DOUBLE), PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP, "CUDA in PETSc has no support for precisions other than single or double");
  #if PetscDefined(HAVE_CUDA)
    #if PetscDefined(HAVE_HPDDM)
      PetscCall(KSPSolve_HPDDM_CUDA_Private(data, b, x, n, PetscObjectComm((PetscObject)ksp)));
    #else
      SETERRQ(PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP, "No CUDA support with --download-hpddm from SLEPc");
    #endif
  #endif
    }
#else
    PetscCheck(data->precision != KSP_HPDDM_PRECISION_HALF, PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP, "Reconfigure with --download-f2cblaslapack --with-f2cblaslapack-fp16-bindings");
#endif
  } else { /* Krylov basis stored in the same precision as PetscScalar */
    if (type == PETSC_MEMTYPE_HOST) PetscCall(HPDDM::IterativeMethod::solve(*data->op, b, x, n, PetscObjectComm((PetscObject)ksp)));
    else {
      PetscCheck(PetscDefined(USE_REAL_SINGLE) || PetscDefined(USE_REAL_DOUBLE), PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP, "CUDA in PETSc has no support for precisions other than single or double");
#if PetscDefined(HAVE_CUDA)
  #if PetscDefined(HAVE_HPDDM)
      PetscCall(KSPSolve_HPDDM_CUDA_Private(data, b, x, n, PetscObjectComm((PetscObject)ksp)));
  #else
      SETERRQ(PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP, "No CUDA support with --download-hpddm from SLEPc");
  #endif
#endif
    }
  }
  if (!ksp->reason) { /* KSPConvergedDefault() is still returning 0 (= KSP_CONVERGED_ITERATING) */
    if (ksp->its >= ksp->max_it) ksp->reason = KSP_DIVERGED_ITS;
    else ksp->reason = KSP_CONVERGED_RTOL; /* early exit by HPDDM, which only happens on breakdowns or convergence */
  }
  ksp->its = PetscMin(ksp->its, ksp->max_it);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPSolve_HPDDM(KSP ksp)
{
  KSP_HPDDM         *data = (KSP_HPDDM *)ksp->data;
  Mat                A, B;
  PetscScalar       *x, *bt = nullptr, **ptr;
  const PetscScalar *b;
  PetscInt           i, j, n;
  PetscBool          flg;
  PetscMemType       type[2];

  PetscFunctionBegin;
  PetscCall(PetscCitationsRegister(HPDDMCitation, &HPDDMCite));
  PetscCall(KSPGetOperators(ksp, &A, nullptr));
  PetscCall(PetscObjectTypeCompareAny((PetscObject)A, &flg, MATSEQKAIJ, MATMPIKAIJ, ""));
  PetscCall(VecGetArrayWriteAndMemType(ksp->vec_sol, &x, type));
  PetscCall(VecGetArrayReadAndMemType(ksp->vec_rhs, &b, type + 1));
  PetscCheck(type[0] == type[1], PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_INCOMP, "Right-hand side and solution vectors must have the same PetscMemType, %s != %s", PetscMemTypeToString(type[0]), PetscMemTypeToString(type[1]));
  if (!flg) {
    if (PetscMemTypeCUDA(type[0])) PetscCall(KSPSolve_HPDDM_Private<PETSC_MEMTYPE_CUDA>(ksp, b, x, 1));
    else {
      PetscCheck(PetscMemTypeHost(type[0]), PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "PetscMemType (%s) is neither PETSC_MEMTYPE_HOST nor PETSC_MEMTYPE_CUDA", PetscMemTypeToString(type[0]));
      PetscCall(KSPSolve_HPDDM_Private(ksp, b, x, 1));
    }
  } else {
    PetscCheck(PetscMemTypeHost(type[0]), PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "PetscMemType (%s) is not PETSC_MEMTYPE_HOST", PetscMemTypeToString(type[0]));
    PetscCall(MatKAIJGetScaledIdentity(A, &flg));
    PetscCall(MatKAIJGetAIJ(A, &B));
    PetscCall(MatGetBlockSize(A, &n));
    PetscCall(MatGetLocalSize(B, &i, nullptr));
    j = data->op->getDof();
    if (!flg) i *= n; /* S and T are not scaled identities, cannot use block methods */
    if (i != j) {     /* switching between block and standard methods */
      delete data->op;
      data->op = new HPDDM::PETScOperator(ksp, i);
    }
    if (flg && n > 1) {
      PetscCall(PetscMalloc1(i * n, &bt));
      /* from row- to column-major to be consistent with HPDDM */
      HPDDM::Wrapper<PetscScalar>::omatcopy<'T'>(i, n, b, n, bt, i);
      ptr = const_cast<PetscScalar **>(&b);
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
  PetscCall(VecRestoreArrayReadAndMemType(ksp->vec_rhs, &b));
  PetscCall(VecRestoreArrayWriteAndMemType(ksp->vec_sol, &x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
     KSPHPDDMSetDeflationMat - Sets the deflation space used by Krylov methods in `KSPHPDDM` with recycling. This space is viewed as a set of vectors stored in
     a `MATDENSE` (column major).

   Input Parameters:
+     ksp - iterative context
-     U - deflation space to be used during KSPSolve()

   Level: intermediate

.seealso: [](chapter_ksp), `KSPHPDDM`, `KSPCreate()`, `KSPType`, `KSPHPDDMGetDeflationMat()`
@*/
PetscErrorCode KSPHPDDMSetDeflationMat(KSP ksp, Mat U)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscValidHeaderSpecific(U, MAT_CLASSID, 2);
  PetscCheckSameComm(ksp, 1, U, 2);
  PetscUseMethod(ksp, "KSPHPDDMSetDeflationMat_C", (KSP, Mat), (ksp, U));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
     KSPHPDDMGetDeflationMat - Gets the deflation space computed by Krylov methods in `KSPHPDDM`  with recycling or NULL if `KSPSolve()` has not been called yet.
     This space is viewed as a set of vectors stored in a `MATDENSE` (column major). It is the responsibility of the user to free the returned `Mat`.

   Input Parameter:
.     ksp - iterative context

   Output Parameter:
.     U - deflation space generated during `KSPSolve()`

   Level: intermediate

.seealso: [](chapter_ksp), `KSPHPDDM`, `KSPCreate()`, `KSPType`, `KSPHPDDMSetDeflationMat()`
@*/
PetscErrorCode KSPHPDDMGetDeflationMat(KSP ksp, Mat *U)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  if (U) {
    PetscValidPointer(U, 2);
    PetscUseMethod(ksp, "KSPHPDDMGetDeflationMat_C", (KSP, Mat *), (ksp, U));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPHPDDMSetDeflationMat_HPDDM(KSP ksp, Mat U)
{
  KSP_HPDDM            *data = (KSP_HPDDM *)ksp->data;
  HPDDM::PETScOperator *op   = data->op;
  Mat                   A;
  const PetscScalar    *array;
  PetscScalar          *copy;
  PetscInt              m1, M1, m2, M2, n2, N2, ldu;
  PetscBool             match;

  PetscFunctionBegin;
  if (!op) {
    PetscCall(KSPSetUp(ksp));
    op = data->op;
  }
  PetscCheck(data->precision == PETSC_KSPHPDDM_DEFAULT_PRECISION, PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP, "%s != %s", KSPHPDDMPrecisionTypes[data->precision], KSPHPDDMPrecisionTypes[PETSC_KSPHPDDM_DEFAULT_PRECISION]);
  PetscCall(KSPGetOperators(ksp, &A, nullptr));
  PetscCall(MatGetLocalSize(A, &m1, nullptr));
  PetscCall(MatGetLocalSize(U, &m2, &n2));
  PetscCall(MatGetSize(A, &M1, nullptr));
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
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPHPDDMGetDeflationMat_HPDDM(KSP ksp, Mat *U)
{
  KSP_HPDDM            *data = (KSP_HPDDM *)ksp->data;
  HPDDM::PETScOperator *op   = data->op;
  Mat                   A;
  const PetscScalar    *array;
  PetscScalar          *copy;
  PetscInt              m1, M1, N2;

  PetscFunctionBegin;
  if (!op) {
    PetscCall(KSPSetUp(ksp));
    op = data->op;
  }
  PetscCheck(data->precision == PETSC_KSPHPDDM_DEFAULT_PRECISION, PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP, "%s != %s", KSPHPDDMPrecisionTypes[data->precision], KSPHPDDMPrecisionTypes[PETSC_KSPHPDDM_DEFAULT_PRECISION]);
  array = op->storage();
  N2    = op->k().first * op->k().second;
  if (!array) *U = nullptr;
  else {
    PetscCall(KSPGetOperators(ksp, &A, nullptr));
    PetscCall(MatGetLocalSize(A, &m1, nullptr));
    PetscCall(MatGetSize(A, &M1, nullptr));
    PetscCall(MatCreateDense(PetscObjectComm((PetscObject)ksp), m1, PETSC_DECIDE, M1, N2, nullptr, U));
    PetscCall(MatDenseGetArrayWrite(*U, &copy));
    PetscCall(PetscArraycpy(copy, array, m1 * N2));
    PetscCall(MatDenseRestoreArrayWrite(*U, &copy));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPMatSolve_HPDDM(KSP ksp, Mat B, Mat X)
{
  KSP_HPDDM         *data = (KSP_HPDDM *)ksp->data;
  Mat                A;
  const PetscScalar *b;
  PetscScalar       *x;
  PetscInt           n, lda;
  PetscMemType       type[2];

  PetscFunctionBegin;
  PetscCall(PetscCitationsRegister(HPDDMCitation, &HPDDMCite));
  if (!data->op) PetscCall(KSPSetUp(ksp));
  PetscCall(KSPGetOperators(ksp, &A, nullptr));
  PetscCall(MatGetLocalSize(B, &n, nullptr));
  PetscCall(MatDenseGetLDA(B, &lda));
  PetscCheck(n == lda, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Unhandled leading dimension lda = %" PetscInt_FMT " with n = %" PetscInt_FMT, lda, n);
  PetscCall(MatGetLocalSize(A, &n, nullptr));
  PetscCall(MatDenseGetLDA(X, &lda));
  PetscCheck(n == lda, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Unhandled leading dimension lda = %" PetscInt_FMT " with n = %" PetscInt_FMT, lda, n);
  PetscCall(MatGetSize(X, nullptr, &n));
  PetscCall(MatDenseGetArrayWriteAndMemType(X, &x, type));
  PetscCall(MatDenseGetArrayReadAndMemType(B, &b, type + 1));
  PetscCheck(type[0] == type[1], PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_INCOMP, "Right-hand side and solution matrices must have the same PetscMemType, %s != %s", PetscMemTypeToString(type[0]), PetscMemTypeToString(type[1]));
  if (PetscMemTypeCUDA(type[0])) PetscCall(KSPSolve_HPDDM_Private<PETSC_MEMTYPE_CUDA>(ksp, b, x, n));
  else {
    PetscCheck(PetscMemTypeHost(type[0]), PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "PetscMemType (%s) is neither PETSC_MEMTYPE_HOST nor PETSC_MEMTYPE_CUDA", PetscMemTypeToString(type[0]));
    PetscCall(KSPSolve_HPDDM_Private(ksp, b, x, n));
  }
  PetscCall(MatDenseRestoreArrayReadAndMemType(B, &b));
  PetscCall(MatDenseRestoreArrayWriteAndMemType(X, &x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
     KSPHPDDMSetType - Sets the type of Krylov method used in `KSPHPDDM`.

   Collective

   Input Parameters:
+     ksp - iterative context
-     type - any of gmres, bgmres, cg, bcg, gcrodr, bgcrodr, bfbcg, or preonly

   Level: intermediate

   Notes:
     Unlike `KSPReset()`, this function does not destroy any deflation space attached to the `KSP`.

     As an example, in the following sequence:
.vb
     KSPHPDDMSetType(ksp, KSPGCRODR);
     KSPSolve(ksp, b, x);
     KSPHPDDMSetType(ksp, KSPGMRES);
     KSPHPDDMSetType(ksp, KSPGCRODR);
     KSPSolve(ksp, b, x);
.ve
    the recycled space is reused in the second `KSPSolve()`.

.seealso: [](chapter_ksp), `KSPCreate()`, `KSPType`, `KSPHPDDMType`, `KSPHPDDMGetType()`
@*/
PetscErrorCode KSPHPDDMSetType(KSP ksp, KSPHPDDMType type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscValidLogicalCollectiveEnum(ksp, type, 2);
  PetscUseMethod(ksp, "KSPHPDDMSetType_C", (KSP, KSPHPDDMType), (ksp, type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
     KSPHPDDMGetType - Gets the type of Krylov method used in `KSPHPDDM`.

   Input Parameter:
.     ksp - iterative context

   Output Parameter:
.     type - any of gmres, bgmres, cg, bcg, gcrodr, bgcrodr, bfbcg, or preonly

   Level: intermediate

.seealso: [](chapter_ksp), `KSPCreate()`, `KSPType`, `KSPHPDDMType`, `KSPHPDDMSetType()`
@*/
PetscErrorCode KSPHPDDMGetType(KSP ksp, KSPHPDDMType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  if (type) {
    PetscValidPointer(type, 2);
    PetscUseMethod(ksp, "KSPHPDDMGetType_C", (KSP, KSPHPDDMType *), (ksp, type));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPHPDDMSetType_HPDDM(KSP ksp, KSPHPDDMType type)
{
  KSP_HPDDM *data = (KSP_HPDDM *)ksp->data;
  PetscInt   i;
  PetscBool  flg = PETSC_FALSE;

  PetscFunctionBegin;
  for (i = 0; i < static_cast<PetscInt>(PETSC_STATIC_ARRAY_LENGTH(KSPHPDDMTypes)); ++i) {
    PetscCall(PetscStrcmp(KSPHPDDMTypes[type], KSPHPDDMTypes[i], &flg));
    if (flg) break;
  }
  PetscCheck(i != PETSC_STATIC_ARRAY_LENGTH(KSPHPDDMTypes), PetscObjectComm((PetscObject)ksp), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown KSPHPDDMType %d", type);
  if (data->cntl[0] != static_cast<char>(PETSC_DECIDE) && data->cntl[0] != i) PetscCall(KSPReset_HPDDM_Private(ksp));
  data->cntl[0] = i;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPHPDDMGetType_HPDDM(KSP ksp, KSPHPDDMType *type)
{
  KSP_HPDDM *data = (KSP_HPDDM *)ksp->data;

  PetscFunctionBegin;
  PetscCheck(data->cntl[0] != static_cast<char>(PETSC_DECIDE), PETSC_COMM_SELF, PETSC_ERR_ORDER, "KSPHPDDMType not set yet");
  /* need to shift by -1 for HPDDM_KRYLOV_METHOD_NONE */
  *type = static_cast<KSPHPDDMType>(PetscMin(data->cntl[0], static_cast<char>(PETSC_STATIC_ARRAY_LENGTH(KSPHPDDMTypes) - 1)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
     KSPHPDDM - Interface with the HPDDM library. This `KSP` may be used to further select methods that are currently not implemented natively in PETSc, e.g.,
     GCRODR [2006], a recycled Krylov method which is similar to `KSPLGMRES`, see [2016] for a comparison. ex75.c shows how to reproduce the results
     from the aforementioned paper [2006]. A chronological bibliography of relevant publications linked with `KSP` available in HPDDM through `KSPHPDDM`,
     and not available directly in PETSc, may be found below. The interface is explained in details in [2021].

   Options Database Keys:
+   -ksp_gmres_restart <restart, default=30> - see `KSPGMRES`
.   -ksp_hpddm_type <type, default=gmres> - any of gmres, bgmres, cg, bcg, gcrodr, bgcrodr, bfbcg, or preonly, see `KSPHPDDMType`
.   -ksp_hpddm_precision <value, default=same as PetscScalar> - any of half, single, double or quadruple, see `KSPHPDDMPrecision`
.   -ksp_hpddm_deflation_tol <eps, default=\-1.0> - tolerance when deflating right-hand sides inside block methods (no deflation by default, only relevant with block methods)
.   -ksp_hpddm_enlarge_krylov_subspace <p, default=1> - split the initial right-hand side into multiple vectors (only relevant with nonblock methods)
.   -ksp_hpddm_orthogonalization <type, default=cgs> - any of cgs or mgs, see KSPGMRES
.   -ksp_hpddm_qr <type, default=cholqr> - distributed QR factorizations with any of cholqr, cgs, or mgs (only relevant with block methods)
.   -ksp_hpddm_variant <type, default=left> - any of left, right, or flexible (this option is superseded by `KSPSetPCSide()`)
.   -ksp_hpddm_recycle <n, default=0> - number of harmonic Ritz vectors to compute (only relevant with GCRODR or BGCRODR)
.   -ksp_hpddm_recycle_target <type, default=SM> - criterion to select harmonic Ritz vectors using either SM, LM, SR, LR, SI, or LI (only relevant with GCRODR or BGCRODR).
     For BGCRODR, if PETSc is compiled with SLEPc, this option is not relevant, since SLEPc is used instead. Options are set with the prefix -ksp_hpddm_recycle_eps_
.   -ksp_hpddm_recycle_strategy <type, default=A> - generalized eigenvalue problem A or B to solve for recycling (only relevant with flexible GCRODR or BGCRODR)
-   -ksp_hpddm_recycle_symmetric <true, default=false> - symmetric generalized eigenproblems in BGCRODR, useful to switch to distributed solvers like EPSELEMENTAL or EPSSCALAPACK
     (only relevant when PETSc is compiled with SLEPc)

   Level: intermediate

   References:
+   1980 - The block conjugate gradient algorithm and related methods. O'Leary. Linear Algebra and its Applications.
.   2006 - Recycling Krylov subspaces for sequences of linear systems. Parks, de Sturler, Mackey, Johnson, and Maiti. SIAM Journal on Scientific Computing
.   2013 - A modified block flexible GMRES method with deflation at each iteration for the solution of non-Hermitian linear systems with multiple right-hand sides.
           Calandra, Gratton, Lago, Vasseur, and Carvalho. SIAM Journal on Scientific Computing.
.   2016 - Block iterative methods and recycling for improved scalability of linear solvers. Jolivet and Tournier. SC16.
.   2017 - A breakdown-free block conjugate gradient method. Ji and Li. BIT Numerical Mathematics.
-   2021 - KSPHPDDM and PCHPDDM: extending PETSc with advanced Krylov methods and robust multilevel overlapping Schwarz preconditioners. Jolivet, Roman, and Zampini.
           Computer & Mathematics with Applications.

.seealso: [](chapter_ksp), [](sec_flexibleksp), `KSPCreate()`, `KSPSetType()`, `KSPType`, `KSP`, `KSPGMRES`, `KSPCG`, `KSPLGMRES`, `KSPDGMRES`
M*/

PETSC_EXTERN PetscErrorCode KSPCreate_HPDDM(KSP ksp)
{
  KSP_HPDDM  *data;
  PetscInt    i;
  const char *common[] = {KSPGMRES, KSPCG, KSPPREONLY};
  PetscBool   flg      = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(PetscNew(&data));
  ksp->data = (void *)data;
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_PRECONDITIONED, PC_LEFT, 2));
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_UNPRECONDITIONED, PC_RIGHT, 1));
  ksp->ops->solve          = KSPSolve_HPDDM;
  ksp->ops->matsolve       = KSPMatSolve_HPDDM;
  ksp->ops->setup          = KSPSetUp_HPDDM;
  ksp->ops->setfromoptions = KSPSetFromOptions_HPDDM;
  ksp->ops->destroy        = KSPDestroy_HPDDM;
  ksp->ops->view           = KSPView_HPDDM;
  ksp->ops->reset          = KSPReset_HPDDM;
  PetscCall(KSPReset_HPDDM_Private(ksp));
  for (i = 0; i < static_cast<PetscInt>(PETSC_STATIC_ARRAY_LENGTH(common)); ++i) {
    PetscCall(PetscStrcmp(((PetscObject)ksp)->type_name, common[i], &flg));
    if (flg) break;
  }
  if (!i) data->cntl[0] = HPDDM_KRYLOV_METHOD_GMRES;
  else if (i == 1) data->cntl[0] = HPDDM_KRYLOV_METHOD_CG;
  else if (i == 2) data->cntl[0] = HPDDM_KRYLOV_METHOD_NONE;
  if (data->cntl[0] != static_cast<char>(PETSC_DECIDE)) PetscCall(PetscInfo(ksp, "Using the previously set KSPType %s\n", common[i]));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPHPDDMSetDeflationMat_C", KSPHPDDMSetDeflationMat_HPDDM));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPHPDDMGetDeflationMat_C", KSPHPDDMGetDeflationMat_HPDDM));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPHPDDMSetType_C", KSPHPDDMSetType_HPDDM));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPHPDDMGetType_C", KSPHPDDMGetType_HPDDM));
#if PetscDefined(HAVE_SLEPC) && PetscDefined(HAVE_DYNAMIC_LIBRARIES) && PetscDefined(USE_SHARED_LIBRARIES)
  if (!loadedDL) PetscCall(HPDDMLoadDL_Private(&loadedDL));
#endif
  data->precision = PETSC_KSPHPDDM_DEFAULT_PRECISION;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#include <petscsf.h>
#include <petsc/private/vecimpl.h>
#include <petsc/private/matimpl.h>
#include <petsc/private/petschpddm.h> /*I "petscpc.h" I*/
#include <petsc/private/pcimpl.h>
#include <petsc/private/dmimpl.h> /* this must be included after petschpddm.h so that DM_MAX_WORK_VECTORS is not defined  */
                                  /* otherwise, it is assumed that one is compiling libhpddm_petsc => circular dependency */

static PetscErrorCode (*loadedSym)(HPDDM::Schwarz<PetscScalar> *const, IS, Mat, Mat, Mat, std::vector<Vec>, PC_HPDDM_Level **const) = nullptr;

static PetscBool PCHPDDMPackageInitialized = PETSC_FALSE;

PetscLogEvent PC_HPDDM_Strc;
PetscLogEvent PC_HPDDM_PtAP;
PetscLogEvent PC_HPDDM_PtBP;
PetscLogEvent PC_HPDDM_Next;
PetscLogEvent PC_HPDDM_SetUp[PETSC_PCHPDDM_MAXLEVELS];
PetscLogEvent PC_HPDDM_Solve[PETSC_PCHPDDM_MAXLEVELS];

const char *const PCHPDDMCoarseCorrectionTypes[] = {"DEFLATED", "ADDITIVE", "BALANCED", "NONE", "PCHPDDMCoarseCorrectionType", "PC_HPDDM_COARSE_CORRECTION_", nullptr};
const char *const PCHPDDMSchurPreTypes[]         = {"LEAST_SQUARES", "GENEO", "PCHPDDMSchurPreType", "PC_HPDDM_SCHUR_PRE", nullptr};

static PetscErrorCode PCReset_HPDDM(PC pc)
{
  PC_HPDDM *data = (PC_HPDDM *)pc->data;

  PetscFunctionBegin;
  if (data->levels) {
    for (PetscInt i = 0; i < PETSC_PCHPDDM_MAXLEVELS && data->levels[i]; ++i) {
      PetscCall(KSPDestroy(&data->levels[i]->ksp));
      PetscCall(PCDestroy(&data->levels[i]->pc));
      PetscCall(PetscFree(data->levels[i]));
    }
    PetscCall(PetscFree(data->levels));
  }
  PetscCall(ISDestroy(&data->is));
  PetscCall(MatDestroy(&data->aux));
  PetscCall(MatDestroy(&data->B));
  PetscCall(VecDestroy(&data->normal));
  data->correction = PC_HPDDM_COARSE_CORRECTION_DEFLATED;
  data->Neumann    = PETSC_BOOL3_UNKNOWN;
  data->deflation  = PETSC_FALSE;
  data->setup      = nullptr;
  data->setup_ctx  = nullptr;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCDestroy_HPDDM(PC pc)
{
  PC_HPDDM *data = (PC_HPDDM *)pc->data;

  PetscFunctionBegin;
  PetscCall(PCReset_HPDDM(pc));
  PetscCall(PetscFree(data));
  PetscCall(PetscObjectChangeTypeName((PetscObject)pc, nullptr));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCHPDDMSetAuxiliaryMat_C", nullptr));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCHPDDMHasNeumannMat_C", nullptr));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCHPDDMSetRHSMat_C", nullptr));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCHPDDMSetCoarseCorrectionType_C", nullptr));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCHPDDMGetCoarseCorrectionType_C", nullptr));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCHPDDMSetSTShareSubKSP_C", nullptr));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCHPDDMGetSTShareSubKSP_C", nullptr));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCHPDDMSetDeflationMat_C", nullptr));
  PetscCall(PetscObjectCompose((PetscObject)pc, "_PCHPDDM_Schur", nullptr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode PCHPDDMSetAuxiliaryMat_Private(PC pc, IS is, Mat A, PetscBool deflation)
{
  PC_HPDDM                   *data = (PC_HPDDM *)pc->data;
  PCHPDDMCoarseCorrectionType type = data->correction;

  PetscFunctionBegin;
  PetscValidLogicalCollectiveBool(pc, deflation, 4);
  if (is && A) {
    PetscInt m[2];

    PetscCall(ISGetLocalSize(is, m));
    PetscCall(MatGetLocalSize(A, m + 1, nullptr));
    PetscCheck(m[0] == m[1], PETSC_COMM_SELF, PETSC_ERR_USER_INPUT, "Inconsistent IS and Mat sizes (%" PetscInt_FMT " v. %" PetscInt_FMT ")", m[0], m[1]);
  }
  if (is) {
    PetscCall(PetscObjectReference((PetscObject)is));
    if (data->is) { /* new overlap definition resets the PC */
      PetscCall(PCReset_HPDDM(pc));
      pc->setfromoptionscalled = 0;
      pc->setupcalled          = PETSC_FALSE;
      data->correction         = type;
    }
    PetscCall(ISDestroy(&data->is));
    data->is = is;
  }
  if (A) {
    PetscCall(PetscObjectReference((PetscObject)A));
    PetscCall(MatDestroy(&data->aux));
    data->aux = A;
  }
  data->deflation = deflation;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode PCHPDDMSplittingMatNormal_Private(Mat A, IS *is, Mat *splitting[])
{
  Mat *sub;
  IS   zero;

  PetscFunctionBegin;
  PetscCall(MatSetOption(A, MAT_SUBMAT_SINGLEIS, PETSC_TRUE));
  PetscCall(MatCreateSubMatrices(A, 1, is + 2, is, MAT_INITIAL_MATRIX, splitting));
  PetscCall(MatCreateSubMatrices(**splitting, 1, is + 2, is + 1, MAT_INITIAL_MATRIX, &sub));
  PetscCall(MatFindZeroRows(*sub, &zero));
  PetscCall(MatDestroySubMatrices(1, &sub));
  PetscCall(MatSetOption(**splitting, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE));
  PetscCall(MatZeroRowsIS(**splitting, zero, 0.0, nullptr, nullptr));
  PetscCall(ISDestroy(&zero));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode PCHPDDMSetAuxiliaryMatNormal_Private(PC pc, Mat A, Mat N, Mat *B, const char *pcpre, Vec *diagonal = nullptr, Mat B01 = nullptr)
{
  PC_HPDDM *data         = (PC_HPDDM *)pc->data;
  Mat      *splitting[2] = {}, aux;
  Vec       d;
  IS        is[3];
  PetscReal norm;
  PetscBool flg;
  char      type[256] = {}; /* same size as in src/ksp/pc/interface/pcset.c */

  PetscFunctionBegin;
  if (!B01) PetscCall(MatConvert(N, MATAIJ, MAT_INITIAL_MATRIX, B));
  else PetscCall(MatTransposeMatMult(B01, A, MAT_INITIAL_MATRIX, PETSC_DETERMINE, B));
  PetscCall(MatEliminateZeros(*B, PETSC_TRUE));
  PetscCall(ISCreateStride(PETSC_COMM_SELF, A->cmap->n, A->cmap->rstart, 1, is));
  PetscCall(MatIncreaseOverlap(*B, 1, is, 1));
  PetscCall(ISCreateStride(PETSC_COMM_SELF, A->cmap->n, A->cmap->rstart, 1, is + 2));
  PetscCall(ISEmbed(is[0], is[2], PETSC_TRUE, is + 1));
  PetscCall(ISDestroy(is + 2));
  PetscCall(ISCreateStride(PETSC_COMM_SELF, A->rmap->N, 0, 1, is + 2));
  PetscCall(PCHPDDMSplittingMatNormal_Private(A, is, &splitting[0]));
  if (B01) {
    PetscCall(PCHPDDMSplittingMatNormal_Private(B01, is, &splitting[1]));
    PetscCall(MatDestroy(&B01));
  }
  PetscCall(ISDestroy(is + 2));
  PetscCall(ISDestroy(is + 1));
  PetscCall(PetscOptionsGetString(nullptr, pcpre, "-pc_hpddm_levels_1_sub_pc_type", type, sizeof(type), nullptr));
  PetscCall(PetscStrcmp(type, PCQR, &flg));
  if (!flg) {
    Mat conjugate = *splitting[splitting[1] ? 1 : 0];

    if (PetscDefined(USE_COMPLEX) && !splitting[1]) {
      PetscCall(MatDuplicate(*splitting[0], MAT_COPY_VALUES, &conjugate));
      PetscCall(MatConjugate(conjugate));
    }
    PetscCall(MatTransposeMatMult(conjugate, *splitting[0], MAT_INITIAL_MATRIX, PETSC_DETERMINE, &aux));
    if (PetscDefined(USE_COMPLEX) && !splitting[1]) PetscCall(MatDestroy(&conjugate));
    else if (splitting[1]) PetscCall(MatDestroySubMatrices(1, &splitting[1]));
    PetscCall(MatNorm(aux, NORM_FROBENIUS, &norm));
    PetscCall(MatSetOption(aux, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
    if (diagonal) {
      PetscReal norm;

      PetscCall(VecScale(*diagonal, -1.0));
      PetscCall(VecNorm(*diagonal, NORM_INFINITY, &norm));
      if (norm > PETSC_SMALL) {
        PetscSF  scatter;
        PetscInt n;

        PetscCall(ISGetLocalSize(*is, &n));
        PetscCall(VecCreateMPI(PetscObjectComm((PetscObject)pc), n, PETSC_DECIDE, &d));
        PetscCall(VecScatterCreate(*diagonal, *is, d, nullptr, &scatter));
        PetscCall(VecScatterBegin(scatter, *diagonal, d, INSERT_VALUES, SCATTER_FORWARD));
        PetscCall(VecScatterEnd(scatter, *diagonal, d, INSERT_VALUES, SCATTER_FORWARD));
        PetscCall(PetscSFDestroy(&scatter));
        PetscCall(MatDiagonalSet(aux, d, ADD_VALUES));
        PetscCall(VecDestroy(&d));
      } else PetscCall(VecDestroy(diagonal));
    }
    if (!diagonal) PetscCall(MatShift(aux, PETSC_SMALL * norm));
  } else {
    PetscBool flg;

    PetscCheck(!splitting[1], PetscObjectComm((PetscObject)pc), PETSC_ERR_SUP, "Cannot use PCQR when A01 != A10^T");
    if (diagonal) {
      PetscCall(VecNorm(*diagonal, NORM_INFINITY, &norm));
      PetscCheck(norm < PETSC_SMALL, PetscObjectComm((PetscObject)pc), PETSC_ERR_SUP, "Nonzero diagonal A11 block");
      PetscCall(VecDestroy(diagonal));
    }
    PetscCall(PetscObjectTypeCompare((PetscObject)N, MATNORMAL, &flg));
    if (flg) PetscCall(MatCreateNormal(*splitting[0], &aux));
    else PetscCall(MatCreateNormalHermitian(*splitting[0], &aux));
  }
  PetscCall(MatDestroySubMatrices(1, &splitting[0]));
  PetscCall(PCHPDDMSetAuxiliaryMat(pc, *is, aux, nullptr, nullptr));
  data->Neumann = PETSC_BOOL3_TRUE;
  PetscCall(ISDestroy(is));
  PetscCall(MatDestroy(&aux));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCHPDDMSetAuxiliaryMat_HPDDM(PC pc, IS is, Mat A, PetscErrorCode (*setup)(Mat, PetscReal, Vec, Vec, PetscReal, IS, void *), void *setup_ctx)
{
  PC_HPDDM *data = (PC_HPDDM *)pc->data;

  PetscFunctionBegin;
  PetscCall(PCHPDDMSetAuxiliaryMat_Private(pc, is, A, PETSC_FALSE));
  if (setup) {
    data->setup     = setup;
    data->setup_ctx = setup_ctx;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PCHPDDMSetAuxiliaryMat - Sets the auxiliary matrix used by `PCHPDDM` for the concurrent GenEO problems at the finest level.

  Input Parameters:
+ pc    - preconditioner context
. is    - index set of the local auxiliary, e.g., Neumann, matrix
. A     - auxiliary sequential matrix
. setup - function for generating the auxiliary matrix entries, may be `NULL`
- ctx   - context for `setup`, may be `NULL`

  Calling sequence of `setup`:
+ J   - matrix whose values are to be set
. t   - time
. X   - linearization point
. X_t - time-derivative of the linearization point
. s   - step
. ovl - index set of the local auxiliary, e.g., Neumann, matrix
- ctx - context for `setup`, may be `NULL`

  Level: intermediate

  Note:
  As an example, in a finite element context with nonoverlapping subdomains plus (overlapping) ghost elements, this could be the unassembled (Neumann)
  local overlapping operator. As opposed to the assembled (Dirichlet) local overlapping operator obtained by summing neighborhood contributions
  at the interface of ghost elements.

  Fortran Notes:
  Only `PETSC_NULL_FUNCTION` is supported for `setup` and `ctx` is never accessed

.seealso: [](ch_ksp), `PCHPDDM`, `PCCreate()`, `PCSetType()`, `PCType`, `PC`, `PCHPDDMSetRHSMat()`, `MATIS`
@*/
PetscErrorCode PCHPDDMSetAuxiliaryMat(PC pc, IS is, Mat A, PetscErrorCode (*setup)(Mat J, PetscReal t, Vec X, Vec X_t, PetscReal s, IS ovl, void *ctx), void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  if (is) PetscValidHeaderSpecific(is, IS_CLASSID, 2);
  if (A) PetscValidHeaderSpecific(A, MAT_CLASSID, 3);
  PetscTryMethod(pc, "PCHPDDMSetAuxiliaryMat_C", (PC, IS, Mat, PetscErrorCode (*)(Mat, PetscReal, Vec, Vec, PetscReal, IS, void *), void *), (pc, is, A, setup, ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCHPDDMHasNeumannMat_HPDDM(PC pc, PetscBool has)
{
  PC_HPDDM *data = (PC_HPDDM *)pc->data;

  PetscFunctionBegin;
  data->Neumann = PetscBoolToBool3(has);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCHPDDMHasNeumannMat - Informs `PCHPDDM` that the `Mat` passed to `PCHPDDMSetAuxiliaryMat()` is the local Neumann matrix.

  Input Parameters:
+ pc  - preconditioner context
- has - Boolean value

  Level: intermediate

  Notes:
  This may be used to bypass a call to `MatCreateSubMatrices()` and to `MatConvert()` for `MATSBAIJ` matrices.

  If a function is composed with DMCreateNeumannOverlap_C implementation is available in the `DM` attached to the Pmat, or the Amat, or the `PC`, the flag is internally set to `PETSC_TRUE`. Its default value is otherwise `PETSC_FALSE`.

.seealso: [](ch_ksp), `PCHPDDM`, `PCHPDDMSetAuxiliaryMat()`
@*/
PetscErrorCode PCHPDDMHasNeumannMat(PC pc, PetscBool has)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscTryMethod(pc, "PCHPDDMHasNeumannMat_C", (PC, PetscBool), (pc, has));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCHPDDMSetRHSMat_HPDDM(PC pc, Mat B)
{
  PC_HPDDM *data = (PC_HPDDM *)pc->data;

  PetscFunctionBegin;
  PetscCall(PetscObjectReference((PetscObject)B));
  PetscCall(MatDestroy(&data->B));
  data->B = B;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCHPDDMSetRHSMat - Sets the right-hand side matrix used by `PCHPDDM` for the concurrent GenEO problems at the finest level.

  Input Parameters:
+ pc - preconditioner context
- B  - right-hand side sequential matrix

  Level: advanced

  Note:
  Must be used in conjunction with `PCHPDDMSetAuxiliaryMat`(N), so that Nv = lambda Bv is solved using `EPSSetOperators`(N, B).
  It is assumed that N and `B` are provided using the same numbering. This provides a means to try more advanced methods such as GenEO-II or H-GenEO.

.seealso: [](ch_ksp), `PCHPDDMSetAuxiliaryMat()`, `PCHPDDM`
@*/
PetscErrorCode PCHPDDMSetRHSMat(PC pc, Mat B)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  if (B) {
    PetscValidHeaderSpecific(B, MAT_CLASSID, 2);
    PetscTryMethod(pc, "PCHPDDMSetRHSMat_C", (PC, Mat), (pc, B));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetFromOptions_HPDDM(PC pc, PetscOptionItems PetscOptionsObject)
{
  PC_HPDDM                   *data   = (PC_HPDDM *)pc->data;
  PC_HPDDM_Level            **levels = data->levels;
  char                        prefix[256], deprecated[256];
  int                         i = 1;
  PetscMPIInt                 size, previous;
  PetscInt                    n, overlap = 1;
  PCHPDDMCoarseCorrectionType type;
  PetscBool                   flg = PETSC_TRUE, set;

  PetscFunctionBegin;
  if (!data->levels) {
    PetscCall(PetscCalloc1(PETSC_PCHPDDM_MAXLEVELS, &levels));
    data->levels = levels;
  }
  PetscOptionsHeadBegin(PetscOptionsObject, "PCHPDDM options");
  PetscCall(PetscOptionsBoundedInt("-pc_hpddm_harmonic_overlap", "Overlap prior to computing local harmonic extensions", "PCHPDDM", overlap, &overlap, &set, 1));
  if (!set) overlap = -1;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)pc), &size));
  previous = size;
  while (i < PETSC_PCHPDDM_MAXLEVELS) {
    PetscInt p = 1;

    if (!data->levels[i - 1]) PetscCall(PetscNew(data->levels + i - 1));
    data->levels[i - 1]->parent = data;
    /* if the previous level has a single process, it is not possible to coarsen further */
    if (previous == 1 || !flg) break;
    data->levels[i - 1]->nu        = 0;
    data->levels[i - 1]->threshold = -1.0;
    PetscCall(PetscSNPrintf(prefix, sizeof(prefix), "-pc_hpddm_levels_%d_eps_nev", i));
    PetscCall(PetscOptionsBoundedInt(prefix, "Local number of deflation vectors computed by SLEPc", "EPSSetDimensions", data->levels[i - 1]->nu, &data->levels[i - 1]->nu, nullptr, 0));
    PetscCall(PetscSNPrintf(deprecated, sizeof(deprecated), "-pc_hpddm_levels_%d_eps_threshold", i));
    PetscCall(PetscSNPrintf(prefix, sizeof(prefix), "-pc_hpddm_levels_%d_eps_threshold_absolute", i));
    PetscCall(PetscOptionsDeprecated(deprecated, prefix, "3.24", nullptr));
    PetscCall(PetscOptionsReal(prefix, "Local absolute threshold for selecting deflation vectors returned by SLEPc", "PCHPDDM", data->levels[i - 1]->threshold, &data->levels[i - 1]->threshold, nullptr));
    if (i == 1) {
      PetscCheck(overlap == -1 || PetscAbsReal(data->levels[i - 1]->threshold + static_cast<PetscReal>(1.0)) < PETSC_MACHINE_EPSILON, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot supply both -pc_hpddm_levels_1_eps_threshold_absolute and -pc_hpddm_harmonic_overlap");
      if (overlap != -1) {
        PetscInt  nsv    = 0;
        PetscBool set[2] = {PETSC_FALSE, PETSC_FALSE};

        PetscCall(PetscSNPrintf(prefix, sizeof(prefix), "-pc_hpddm_levels_%d_svd_nsv", i));
        PetscCall(PetscOptionsBoundedInt(prefix, "Local number of deflation vectors computed by SLEPc", "SVDSetDimensions", nsv, &nsv, nullptr, 0));
        PetscCheck(data->levels[0]->nu == 0 || nsv == 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot supply both -pc_hpddm_levels_1_eps_nev and -pc_hpddm_levels_1_svd_nsv");
        if (data->levels[0]->nu == 0) { /* -eps_nev has not been used, so nu is 0 */
          data->levels[0]->nu = nsv;    /* nu may still be 0 if -svd_nsv has not been used */
          PetscCall(PetscSNPrintf(deprecated, sizeof(deprecated), "-pc_hpddm_levels_%d_svd_relative_threshold", i));
          PetscCall(PetscSNPrintf(prefix, sizeof(prefix), "-pc_hpddm_levels_%d_svd_threshold_relative", i));
          PetscCall(PetscOptionsDeprecated(deprecated, prefix, "3.24", nullptr));
          PetscCall(PetscOptionsReal(prefix, "Local relative threshold for selecting deflation vectors returned by SLEPc", "PCHPDDM", data->levels[0]->threshold, &data->levels[0]->threshold, set)); /* cache whether this option has been used or not to error out in case of exclusive options being used simultaneously later on */
        }
        if (data->levels[0]->nu == 0 || nsv == 0) { /* if neither -eps_nev nor -svd_nsv has been used */
          PetscCall(PetscSNPrintf(deprecated, sizeof(deprecated), "-pc_hpddm_levels_%d_eps_relative_threshold", i));
          PetscCall(PetscSNPrintf(prefix, sizeof(prefix), "-pc_hpddm_levels_%d_eps_threshold_relative", i));
          PetscCall(PetscOptionsDeprecated(deprecated, prefix, "3.24", nullptr));
          PetscCall(PetscOptionsReal(prefix, "Local relative threshold for selecting deflation vectors returned by SLEPc", "PCHPDDM", data->levels[0]->threshold, &data->levels[0]->threshold, set + 1));
          PetscCheck(!set[0] || !set[1], PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot supply both -pc_hpddm_levels_1_eps_threshold_relative and -pc_hpddm_levels_1_svd_threshold_relative");
        }
        PetscCheck(data->levels[0]->nu || PetscAbsReal(data->levels[i - 1]->threshold + static_cast<PetscReal>(1.0)) > PETSC_MACHINE_EPSILON, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Need to supply at least one of 1) -pc_hpddm_levels_1_eps_nev, 2) -pc_hpddm_levels_1_svd_nsv, 3) -pc_hpddm_levels_1_eps_threshold_relative, 4) -pc_hpddm_levels_1_svd_threshold_relative (for nonsymmetric matrices, only option 2 and option 4 are appropriate)");
      }
      PetscCall(PetscSNPrintf(prefix, sizeof(prefix), "-pc_hpddm_levels_1_st_share_sub_ksp"));
      PetscCall(PetscOptionsBool(prefix, "Shared KSP between SLEPc ST and the fine-level subdomain solver", "PCHPDDMSetSTShareSubKSP", PETSC_FALSE, &data->share, nullptr));
    }
    /* if there is no prescribed coarsening, just break out of the loop */
    if (data->levels[i - 1]->threshold <= PetscReal() && data->levels[i - 1]->nu <= 0 && !(data->deflation && i == 1)) break;
    else {
      ++i;
      PetscCall(PetscSNPrintf(prefix, sizeof(prefix), "-pc_hpddm_levels_%d_eps_nev", i));
      PetscCall(PetscOptionsHasName(PetscOptionsObject->options, PetscOptionsObject->prefix, prefix, &flg));
      if (!flg) {
        PetscCall(PetscSNPrintf(prefix, sizeof(prefix), "-pc_hpddm_levels_%d_eps_threshold_absolute", i));
        PetscCall(PetscOptionsHasName(PetscOptionsObject->options, PetscOptionsObject->prefix, prefix, &flg));
      }
      if (flg) {
        /* if there are coarsening options for the next level, then register it  */
        /* otherwise, don't to avoid having both options levels_N_p and coarse_p */
        PetscCall(PetscSNPrintf(prefix, sizeof(prefix), "-pc_hpddm_levels_%d_p", i));
        PetscCall(PetscOptionsRangeInt(prefix, "Number of processes used to assemble the coarse operator at this level", "PCHPDDM", p, &p, &flg, 1, PetscMax(1, previous / 2)));
        previous = p;
      }
    }
  }
  data->N = i;
  n       = 1;
  if (i > 1) {
    PetscCall(PetscSNPrintf(prefix, sizeof(prefix), "-pc_hpddm_coarse_p"));
    PetscCall(PetscOptionsRangeInt(prefix, "Number of processes used to assemble the coarsest operator", "PCHPDDM", n, &n, nullptr, 1, PetscMax(1, previous / 2)));
#if PetscDefined(HAVE_MUMPS)
    PetscCall(PetscSNPrintf(prefix, sizeof(prefix), "pc_hpddm_coarse_"));
    PetscCall(PetscOptionsHasName(nullptr, prefix, "-mat_mumps_use_omp_threads", &flg));
    if (flg) {
      char type[64]; /* same size as in src/ksp/pc/impls/factor/factimpl.c */

      PetscCall(PetscStrncpy(type, n > 1 && PetscDefined(HAVE_MUMPS) ? MATSOLVERMUMPS : MATSOLVERPETSC, sizeof(type))); /* default solver for a MatMPIAIJ or a MatSeqAIJ */
      PetscCall(PetscOptionsGetString(nullptr, prefix, "-pc_factor_mat_solver_type", type, sizeof(type), nullptr));
      PetscCall(PetscStrcmp(type, MATSOLVERMUMPS, &flg));
      PetscCheck(flg, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "-%smat_mumps_use_omp_threads and -%spc_factor_mat_solver_type != %s", prefix, prefix, MATSOLVERMUMPS);
      size = n;
      n    = -1;
      PetscCall(PetscOptionsGetInt(nullptr, prefix, "-mat_mumps_use_omp_threads", &n, nullptr));
      PetscCheck(n >= 1, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Need to specify a positive integer for -%smat_mumps_use_omp_threads", prefix);
      PetscCheck(n * size <= previous, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "%d MPI process%s x %d OpenMP thread%s greater than %d available MPI process%s for the coarsest operator", (int)size, size > 1 ? "es" : "", (int)n, n > 1 ? "s" : "", (int)previous, previous > 1 ? "es" : "");
    }
#endif
    PetscCall(PetscOptionsEnum("-pc_hpddm_coarse_correction", "Type of coarse correction applied each iteration", "PCHPDDMSetCoarseCorrectionType", PCHPDDMCoarseCorrectionTypes, (PetscEnum)data->correction, (PetscEnum *)&type, &flg));
    if (flg) PetscCall(PCHPDDMSetCoarseCorrectionType(pc, type));
    PetscCall(PetscSNPrintf(prefix, sizeof(prefix), "-pc_hpddm_has_neumann"));
    PetscCall(PetscOptionsBool(prefix, "Is the auxiliary Mat the local Neumann matrix?", "PCHPDDMHasNeumannMat", PetscBool3ToBool(data->Neumann), &flg, &set));
    if (set) data->Neumann = PetscBoolToBool3(flg);
    data->log_separate = PETSC_FALSE;
    if (PetscDefined(USE_LOG)) {
      PetscCall(PetscSNPrintf(prefix, sizeof(prefix), "-pc_hpddm_log_separate"));
      PetscCall(PetscOptionsBool(prefix, "Log events level by level instead of inside PCSetUp()/KSPSolve()", nullptr, data->log_separate, &data->log_separate, nullptr));
    }
  }
  PetscOptionsHeadEnd();
  while (i < PETSC_PCHPDDM_MAXLEVELS && data->levels[i]) PetscCall(PetscFree(data->levels[i++]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <bool transpose>
static PetscErrorCode PCApply_HPDDM(PC pc, Vec x, Vec y)
{
  PC_HPDDM *data = (PC_HPDDM *)pc->data;

  PetscFunctionBegin;
  PetscCall(PetscCitationsRegister(HPDDMCitation, &HPDDMCite));
  PetscCheck(data->levels[0]->ksp, PETSC_COMM_SELF, PETSC_ERR_PLIB, "No KSP attached to PCHPDDM");
  if (data->log_separate) PetscCall(PetscLogEventBegin(PC_HPDDM_Solve[0], data->levels[0]->ksp, nullptr, nullptr, nullptr)); /* coarser-level events are directly triggered in HPDDM */
  if (!transpose) PetscCall(KSPSolve(data->levels[0]->ksp, x, y));
  else PetscCall(KSPSolveTranspose(data->levels[0]->ksp, x, y));
  if (data->log_separate) PetscCall(PetscLogEventEnd(PC_HPDDM_Solve[0], data->levels[0]->ksp, nullptr, nullptr, nullptr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <bool transpose>
static PetscErrorCode PCMatApply_HPDDM(PC pc, Mat X, Mat Y)
{
  PC_HPDDM *data = (PC_HPDDM *)pc->data;

  PetscFunctionBegin;
  PetscCall(PetscCitationsRegister(HPDDMCitation, &HPDDMCite));
  PetscCheck(data->levels[0]->ksp, PETSC_COMM_SELF, PETSC_ERR_PLIB, "No KSP attached to PCHPDDM");
  if (!transpose) PetscCall(KSPMatSolve(data->levels[0]->ksp, X, Y));
  else PetscCall(KSPMatSolveTranspose(data->levels[0]->ksp, X, Y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCHPDDMGetComplexities - Computes the grid and operator complexities.

  Collective

  Input Parameter:
. pc - preconditioner context

  Output Parameters:
+ gc - grid complexity $ \sum_i m_i / m_1 $
- oc - operator complexity $ \sum_i nnz_i / nnz_1 $

  Level: advanced

.seealso: [](ch_ksp), `PCMGGetGridComplexity()`, `PCHPDDM`, `PCHYPRE`, `PCGAMG`
@*/
PetscErrorCode PCHPDDMGetComplexities(PC pc, PetscReal *gc, PetscReal *oc)
{
  PC_HPDDM      *data = (PC_HPDDM *)pc->data;
  MatInfo        info;
  PetscLogDouble accumulate[2]{}, nnz1 = 1.0, m1 = 1.0;

  PetscFunctionBegin;
  if (gc) {
    PetscAssertPointer(gc, 2);
    *gc = 0;
  }
  if (oc) {
    PetscAssertPointer(oc, 3);
    *oc = 0;
  }
  for (PetscInt n = 0; n < data->N; ++n) {
    if (data->levels[n]->ksp) {
      Mat       P, A = nullptr;
      PetscInt  m;
      PetscBool flg = PETSC_FALSE;

      PetscCall(KSPGetOperators(data->levels[n]->ksp, nullptr, &P));
      PetscCall(MatGetSize(P, &m, nullptr));
      accumulate[0] += m;
      if (n == 0) {
        PetscCall(PetscObjectTypeCompareAny((PetscObject)P, &flg, MATNORMAL, MATNORMALHERMITIAN, ""));
        if (flg) {
          PetscCall(MatConvert(P, MATAIJ, MAT_INITIAL_MATRIX, &A));
          P = A;
        } else {
          PetscCall(PetscObjectTypeCompare((PetscObject)P, MATSCHURCOMPLEMENT, &flg));
          PetscCall(PetscObjectReference((PetscObject)P));
        }
      }
      if (!A && flg) accumulate[1] += m * m; /* assumption that a MATSCHURCOMPLEMENT is dense if stored explicitly */
      else if (P->ops->getinfo) {
        PetscCall(MatGetInfo(P, MAT_GLOBAL_SUM, &info));
        accumulate[1] += info.nz_used;
      }
      if (n == 0) {
        m1 = m;
        if (!A && flg) nnz1 = m * m;
        else if (P->ops->getinfo) nnz1 = info.nz_used;
        PetscCall(MatDestroy(&P));
      }
    }
  }
  /* only process #0 has access to the full hierarchy by construction, so broadcast to ensure consistent outputs */
  PetscCallMPI(MPI_Bcast(accumulate, 2, MPIU_PETSCLOGDOUBLE, 0, PetscObjectComm((PetscObject)pc)));
  if (gc) *gc = static_cast<PetscReal>(accumulate[0] / m1);
  if (oc) *oc = static_cast<PetscReal>(accumulate[1] / nnz1);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCView_HPDDM(PC pc, PetscViewer viewer)
{
  PC_HPDDM         *data = (PC_HPDDM *)pc->data;
  PetscViewer       subviewer;
  PetscViewerFormat format;
  PetscSubcomm      subcomm;
  PetscReal         oc, gc;
  PetscInt          tabs;
  PetscMPIInt       size, color, rank;
  PetscBool         flg;
  const char       *name;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &flg));
  if (flg) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "level%s: %" PetscInt_FMT "\n", data->N > 1 ? "s" : "", data->N));
    PetscCall(PCHPDDMGetComplexities(pc, &gc, &oc));
    if (data->N > 1) {
      if (!data->deflation) {
        PetscCall(PetscViewerASCIIPrintf(viewer, "Neumann matrix attached? %s\n", PetscBools[PetscBool3ToBool(data->Neumann)]));
        PetscCall(PetscViewerASCIIPrintf(viewer, "shared subdomain KSP between SLEPc and PETSc? %s\n", PetscBools[data->share]));
      } else PetscCall(PetscViewerASCIIPrintf(viewer, "user-supplied deflation matrix\n"));
      PetscCall(PetscViewerASCIIPrintf(viewer, "coarse correction: %s\n", PCHPDDMCoarseCorrectionTypes[data->correction]));
      PetscCall(PetscViewerASCIIPrintf(viewer, "on process #0, value%s (+ threshold%s if available) for selecting deflation vectors:", data->N > 2 ? "s" : "", data->N > 2 ? "s" : ""));
      PetscCall(PetscViewerASCIIGetTab(viewer, &tabs));
      PetscCall(PetscViewerASCIISetTab(viewer, 0));
      for (PetscInt i = 1; i < data->N; ++i) {
        PetscCall(PetscViewerASCIIPrintf(viewer, " %" PetscInt_FMT, data->levels[i - 1]->nu));
        if (data->levels[i - 1]->threshold > static_cast<PetscReal>(-0.1)) PetscCall(PetscViewerASCIIPrintf(viewer, " (%g)", (double)data->levels[i - 1]->threshold));
      }
      PetscCall(PetscViewerASCIIPrintf(viewer, "\n"));
      PetscCall(PetscViewerASCIISetTab(viewer, tabs));
    }
    PetscCall(PetscViewerASCIIPrintf(viewer, "grid and operator complexities: %g %g\n", (double)gc, (double)oc));
    PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)pc), &size));
    if (data->levels[0]->ksp) {
      PetscCall(KSPView(data->levels[0]->ksp, viewer));
      if (data->levels[0]->pc) PetscCall(PCView(data->levels[0]->pc, viewer));
      PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)pc), &rank));
      for (PetscInt i = 1; i < data->N; ++i) {
        if (data->levels[i]->ksp) color = 1;
        else color = 0;
        PetscCall(PetscSubcommCreate(PetscObjectComm((PetscObject)pc), &subcomm));
        PetscCall(PetscSubcommSetNumber(subcomm, PetscMin(size, 2)));
        PetscCall(PetscSubcommSetTypeGeneral(subcomm, color, rank));
        PetscCall(PetscViewerASCIIPushTab(viewer));
        PetscCall(PetscViewerGetSubViewer(viewer, PetscSubcommChild(subcomm), &subviewer));
        if (color == 1) {
          PetscCall(KSPView(data->levels[i]->ksp, subviewer));
          if (data->levels[i]->pc) PetscCall(PCView(data->levels[i]->pc, subviewer));
          PetscCall(PetscViewerFlush(subviewer));
        }
        PetscCall(PetscViewerRestoreSubViewer(viewer, PetscSubcommChild(subcomm), &subviewer));
        PetscCall(PetscViewerASCIIPopTab(viewer));
        PetscCall(PetscSubcommDestroy(&subcomm));
      }
    }
    PetscCall(PetscViewerGetFormat(viewer, &format));
    if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
      PetscCall(PetscViewerFileGetName(viewer, &name));
      if (name) {
        Mat             aux[2];
        IS              is;
        const PetscInt *indices;
        PetscInt        m, n, sizes[5] = {pc->mat->rmap->n, pc->mat->cmap->n, pc->mat->rmap->N, pc->mat->cmap->N, 0};
        char           *tmp;
        std::string     prefix, suffix;
        size_t          pos;

        PetscCall(PetscStrstr(name, ".", &tmp));
        if (tmp) {
          pos    = std::distance(const_cast<char *>(name), tmp);
          prefix = std::string(name, pos);
          suffix = std::string(name + pos + 1);
        } else prefix = name;
        if (data->aux) {
          PetscCall(MatGetSize(data->aux, &m, &n));
          PetscCall(MatCreate(PetscObjectComm((PetscObject)pc), aux));
          PetscCall(MatSetSizes(aux[0], m, n, PETSC_DETERMINE, PETSC_DETERMINE));
          PetscCall(PetscObjectBaseTypeCompare((PetscObject)data->aux, MATSEQAIJ, &flg));
          if (flg) PetscCall(MatSetType(aux[0], MATMPIAIJ));
          else {
            PetscCall(PetscObjectBaseTypeCompare((PetscObject)data->aux, MATSEQBAIJ, &flg));
            if (flg) PetscCall(MatSetType(aux[0], MATMPIBAIJ));
            else {
              PetscCall(PetscObjectBaseTypeCompare((PetscObject)data->aux, MATSEQSBAIJ, &flg));
              PetscCheck(flg, PetscObjectComm((PetscObject)pc), PETSC_ERR_SUP, "MatType of auxiliary Mat (%s) is not any of the following: MATSEQAIJ, MATSEQBAIJ, or MATSEQSBAIJ", ((PetscObject)data->aux)->type_name);
              PetscCall(MatSetType(aux[0], MATMPISBAIJ));
            }
          }
          PetscCall(MatSetBlockSizesFromMats(aux[0], data->aux, data->aux));
          PetscCall(MatAssemblyBegin(aux[0], MAT_FINAL_ASSEMBLY));
          PetscCall(MatAssemblyEnd(aux[0], MAT_FINAL_ASSEMBLY));
          PetscCall(MatGetDiagonalBlock(aux[0], aux + 1));
          PetscCall(MatCopy(data->aux, aux[1], DIFFERENT_NONZERO_PATTERN));
          PetscCall(PetscViewerBinaryOpen(PetscObjectComm((PetscObject)pc), std::string(prefix + "_aux_" + std::to_string(size) + (tmp ? ("." + suffix) : "")).c_str(), FILE_MODE_WRITE, &subviewer));
          PetscCall(MatView(aux[0], subviewer));
          PetscCall(PetscViewerDestroy(&subviewer));
          PetscCall(MatDestroy(aux));
        }
        if (data->is) {
          PetscCall(ISGetIndices(data->is, &indices));
          PetscCall(ISGetSize(data->is, sizes + 4));
          PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)pc), sizes[4], indices, PETSC_USE_POINTER, &is));
          PetscCall(PetscViewerBinaryOpen(PetscObjectComm((PetscObject)pc), std::string(prefix + "_is_" + std::to_string(size) + (tmp ? ("." + suffix) : "")).c_str(), FILE_MODE_WRITE, &subviewer));
          PetscCall(ISView(is, subviewer));
          PetscCall(PetscViewerDestroy(&subviewer));
          PetscCall(ISDestroy(&is));
          PetscCall(ISRestoreIndices(data->is, &indices));
        }
        PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)pc), PETSC_STATIC_ARRAY_LENGTH(sizes), sizes, PETSC_USE_POINTER, &is));
        PetscCall(PetscViewerBinaryOpen(PetscObjectComm((PetscObject)pc), std::string(prefix + "_sizes_" + std::to_string(size) + (tmp ? ("." + suffix) : "")).c_str(), FILE_MODE_WRITE, &subviewer));
        PetscCall(ISView(is, subviewer));
        PetscCall(PetscViewerDestroy(&subviewer));
        PetscCall(ISDestroy(&is));
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCPreSolve_HPDDM(PC pc, KSP ksp, Vec, Vec)
{
  PC_HPDDM *data = (PC_HPDDM *)pc->data;
  Mat       A;
  PetscBool flg;

  PetscFunctionBegin;
  if (ksp) {
    PetscCall(PetscObjectTypeCompare((PetscObject)ksp, KSPLSQR, &flg));
    if (flg && !data->normal) {
      PetscCall(KSPGetOperators(ksp, &A, nullptr));
      PetscCall(MatCreateVecs(A, nullptr, &data->normal)); /* temporary Vec used in PCApply_HPDDMShell() for coarse grid corrections */
    } else if (!flg) {
      PetscCall(PetscObjectTypeCompareAny((PetscObject)ksp, &flg, KSPCG, KSPGROPPCG, KSPPIPECG, KSPPIPECGRR, KSPPIPELCG, KSPPIPEPRCG, KSPPIPECG2, KSPSTCG, KSPFCG, KSPPIPEFCG, KSPMINRES, KSPNASH, KSPSYMMLQ, ""));
      if (!flg) {
        PetscCall(PetscObjectTypeCompare((PetscObject)ksp, KSPHPDDM, &flg));
        if (flg) {
          KSPHPDDMType type;

          PetscCall(KSPHPDDMGetType(ksp, &type));
          flg = (type == KSP_HPDDM_TYPE_CG || type == KSP_HPDDM_TYPE_BCG || type == KSP_HPDDM_TYPE_BFBCG ? PETSC_TRUE : PETSC_FALSE);
        }
      }
    }
    if (flg) {
      if (data->correction == PC_HPDDM_COARSE_CORRECTION_DEFLATED) {
        PetscCall(PetscOptionsHasName(((PetscObject)pc)->options, ((PetscObject)pc)->prefix, "-pc_hpddm_coarse_correction", &flg));
        PetscCheck(flg, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_INCOMP, "PCHPDDMCoarseCorrectionType %s is known to be not symmetric, but KSPType %s requires a symmetric PC, if you insist on using this configuration, use the additional option -%spc_hpddm_coarse_correction %s, or alternatively, switch to a symmetric PCHPDDMCoarseCorrectionType such as %s",
                   PCHPDDMCoarseCorrectionTypes[data->correction], ((PetscObject)ksp)->type_name, ((PetscObject)pc)->prefix ? ((PetscObject)pc)->prefix : "", PCHPDDMCoarseCorrectionTypes[data->correction], PCHPDDMCoarseCorrectionTypes[PC_HPDDM_COARSE_CORRECTION_BALANCED]);
      }
      for (PetscInt n = 0; n < data->N; ++n) {
        if (data->levels[n]->pc) {
          PetscCall(PetscObjectTypeCompare((PetscObject)data->levels[n]->pc, PCASM, &flg));
          if (flg) {
            PCASMType type;

            PetscCall(PCASMGetType(data->levels[n]->pc, &type));
            if (type == PC_ASM_RESTRICT || type == PC_ASM_INTERPOLATE) {
              PetscCall(PetscOptionsHasName(((PetscObject)data->levels[n]->pc)->options, ((PetscObject)data->levels[n]->pc)->prefix, "-pc_asm_type", &flg));
              PetscCheck(flg, PetscObjectComm((PetscObject)data->levels[n]->pc), PETSC_ERR_ARG_INCOMP, "PCASMType %s is known to be not symmetric, but KSPType %s requires a symmetric PC, if you insist on using this configuration, use the additional option -%spc_asm_type %s, or alternatively, switch to a symmetric PCASMType such as %s", PCASMTypes[type],
                         ((PetscObject)ksp)->type_name, ((PetscObject)data->levels[n]->pc)->prefix, PCASMTypes[type], PCASMTypes[PC_ASM_BASIC]);
            }
          }
        }
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetUp_HPDDMShell(PC pc)
{
  PC_HPDDM_Level *ctx;
  Mat             A, P;
  Vec             x;
  const char     *pcpre;

  PetscFunctionBegin;
  PetscCall(PCShellGetContext(pc, &ctx));
  PetscCall(KSPGetOptionsPrefix(ctx->ksp, &pcpre));
  PetscCall(KSPGetOperators(ctx->ksp, &A, &P));
  /* smoother */
  PetscCall(PCSetOptionsPrefix(ctx->pc, pcpre));
  PetscCall(PCSetOperators(ctx->pc, A, P));
  if (!ctx->v[0]) {
    PetscCall(VecDuplicateVecs(ctx->D, 1, &ctx->v[0]));
    if (!std::is_same<PetscScalar, PetscReal>::value) PetscCall(VecDestroy(&ctx->D));
    PetscCall(MatCreateVecs(A, &x, nullptr));
    PetscCall(VecDuplicateVecs(x, 2, &ctx->v[1]));
    PetscCall(VecDestroy(&x));
  }
  std::fill_n(ctx->V, 3, nullptr);
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <bool transpose = false, class Type = Vec, typename std::enable_if<std::is_same<Type, Vec>::value>::type * = nullptr>
static inline PetscErrorCode PCHPDDMDeflate_Private(PC pc, Type x, Type y)
{
  PC_HPDDM_Level *ctx;
  PetscScalar    *out;

  PetscFunctionBegin;
  PetscCall(PCShellGetContext(pc, &ctx));
  /* going from PETSc to HPDDM numbering */
  PetscCall(VecScatterBegin(ctx->scatter, x, ctx->v[0][0], INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(ctx->scatter, x, ctx->v[0][0], INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecGetArrayWrite(ctx->v[0][0], &out));
  ctx->P->deflation<false, transpose>(nullptr, out, 1); /* y = Q x */
  PetscCall(VecRestoreArrayWrite(ctx->v[0][0], &out));
  /* going from HPDDM to PETSc numbering */
  PetscCall(VecScatterBegin(ctx->scatter, ctx->v[0][0], y, INSERT_VALUES, SCATTER_REVERSE));
  PetscCall(VecScatterEnd(ctx->scatter, ctx->v[0][0], y, INSERT_VALUES, SCATTER_REVERSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <bool transpose = false, class Type = Mat, typename std::enable_if<std::is_same<Type, Mat>::value>::type * = nullptr>
static inline PetscErrorCode PCHPDDMDeflate_Private(PC pc, Type X, Type Y)
{
  PC_HPDDM_Level *ctx;
  Vec             vX, vY, vC;
  PetscScalar    *out;
  PetscInt        N;

  PetscFunctionBegin;
  PetscCall(PCShellGetContext(pc, &ctx));
  PetscCall(MatGetSize(X, nullptr, &N));
  /* going from PETSc to HPDDM numbering */
  for (PetscInt i = 0; i < N; ++i) {
    PetscCall(MatDenseGetColumnVecRead(X, i, &vX));
    PetscCall(MatDenseGetColumnVecWrite(ctx->V[0], i, &vC));
    PetscCall(VecScatterBegin(ctx->scatter, vX, vC, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(ctx->scatter, vX, vC, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(MatDenseRestoreColumnVecWrite(ctx->V[0], i, &vC));
    PetscCall(MatDenseRestoreColumnVecRead(X, i, &vX));
  }
  PetscCall(MatDenseGetArrayWrite(ctx->V[0], &out));
  ctx->P->deflation<false, transpose>(nullptr, out, N); /* Y = Q X */
  PetscCall(MatDenseRestoreArrayWrite(ctx->V[0], &out));
  /* going from HPDDM to PETSc numbering */
  for (PetscInt i = 0; i < N; ++i) {
    PetscCall(MatDenseGetColumnVecRead(ctx->V[0], i, &vC));
    PetscCall(MatDenseGetColumnVecWrite(Y, i, &vY));
    PetscCall(VecScatterBegin(ctx->scatter, vC, vY, INSERT_VALUES, SCATTER_REVERSE));
    PetscCall(VecScatterEnd(ctx->scatter, vC, vY, INSERT_VALUES, SCATTER_REVERSE));
    PetscCall(MatDenseRestoreColumnVecWrite(Y, i, &vY));
    PetscCall(MatDenseRestoreColumnVecRead(ctx->V[0], i, &vC));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
     PCApply_HPDDMShell - Applies a (2) deflated, (1) additive, (3) balanced, or (4) no coarse correction. In what follows, E = Z Pmat Z^T and Q = Z^T E^-1 Z.

.vb
   (1) y =                Pmat^-1              x + Q x,
   (2) y =                Pmat^-1 (I - Amat Q) x + Q x (default),
   (3) y = (I - Q Amat^T) Pmat^-1 (I - Amat Q) x + Q x,
   (4) y =                Pmat^-1              x      .
.ve

   Input Parameters:
+     pc - preconditioner context
-     x - input vector

   Output Parameter:
.     y - output vector

   Notes:
     The options of Pmat^1 = pc(Pmat) are prefixed by -pc_hpddm_levels_1_pc_. Z is a tall-and-skiny matrix assembled by HPDDM. The number of processes on which (Z Pmat Z^T) is aggregated is set via -pc_hpddm_coarse_p.
     The options of (Z Pmat Z^T)^-1 = ksp(Z Pmat Z^T) are prefixed by -pc_hpddm_coarse_ (`KSPPREONLY` and `PCCHOLESKY` by default), unless a multilevel correction is turned on, in which case, this function is called recursively at each level except the coarsest one.
     (1) and (2) visit the "next" level (in terms of coarsening) once per application, while (3) visits it twice, so it is asymptotically twice costlier. (2) is not symmetric even if both Amat and Pmat are symmetric.

   Level: advanced

   Developer Note:
   Since this is not an actual manual page the material below should be moved to an appropriate manual page with the appropriate context, i.e. explaining when it is used and how
   to trigger it. Likely the manual page is `PCHPDDM`

.seealso: [](ch_ksp), `PCHPDDM`, `PCHPDDMCoarseCorrectionType`
*/
static PetscErrorCode PCApply_HPDDMShell(PC pc, Vec x, Vec y)
{
  PC_HPDDM_Level *ctx;
  Mat             A;

  PetscFunctionBegin;
  PetscCall(PCShellGetContext(pc, &ctx));
  PetscCheck(ctx->P, PETSC_COMM_SELF, PETSC_ERR_PLIB, "PCSHELL from PCHPDDM called with no HPDDM object");
  PetscCall(KSPGetOperators(ctx->ksp, &A, nullptr));
  if (ctx->parent->correction == PC_HPDDM_COARSE_CORRECTION_NONE) PetscCall(PCApply(ctx->pc, x, y)); /* y = M^-1 x */
  else {
    PetscCall(PCHPDDMDeflate_Private(pc, x, y)); /* y = Q x */
    if (ctx->parent->correction == PC_HPDDM_COARSE_CORRECTION_DEFLATED || ctx->parent->correction == PC_HPDDM_COARSE_CORRECTION_BALANCED) {
      if (!ctx->parent->normal || ctx != ctx->parent->levels[0]) PetscCall(MatMult(A, y, ctx->v[1][0])); /* y = A Q x     */
      else { /* KSPLSQR and finest level */ PetscCall(MatMult(A, y, ctx->parent->normal));               /* y = A Q x     */
        PetscCall(MatMultHermitianTranspose(A, ctx->parent->normal, ctx->v[1][0]));                      /* y = A^T A Q x */
      }
      PetscCall(VecWAXPY(ctx->v[1][1], -1.0, ctx->v[1][0], x)); /* y = (I - A Q) x                             */
      PetscCall(PCApply(ctx->pc, ctx->v[1][1], ctx->v[1][0]));  /* y = M^-1 (I - A Q) x                        */
      if (ctx->parent->correction == PC_HPDDM_COARSE_CORRECTION_BALANCED) {
        if (!ctx->parent->normal || ctx != ctx->parent->levels[0]) PetscCall(MatMultHermitianTranspose(A, ctx->v[1][0], ctx->v[1][1])); /* z = A^T y */
        else {
          PetscCall(MatMult(A, ctx->v[1][0], ctx->parent->normal));
          PetscCall(MatMultHermitianTranspose(A, ctx->parent->normal, ctx->v[1][1])); /* z = A^T A y           */
        }
        PetscCall(PCHPDDMDeflate_Private(pc, ctx->v[1][1], ctx->v[1][1]));     /* z = Q z                      */
        PetscCall(VecAXPBYPCZ(y, -1.0, 1.0, 1.0, ctx->v[1][1], ctx->v[1][0])); /* y = (I - Q A^T) y + Q x      */
      } else PetscCall(VecAXPY(y, 1.0, ctx->v[1][0]));                         /* y = Q M^-1 (I - A Q) x + Q x */
    } else {
      PetscCheck(ctx->parent->correction == PC_HPDDM_COARSE_CORRECTION_ADDITIVE, PetscObjectComm((PetscObject)pc), PETSC_ERR_PLIB, "PCSHELL from PCHPDDM called with an unknown PCHPDDMCoarseCorrectionType %d", ctx->parent->correction);
      PetscCall(PCApply(ctx->pc, x, ctx->v[1][0]));
      PetscCall(VecAXPY(y, 1.0, ctx->v[1][0])); /* y = M^-1 x + Q x */
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <bool transpose>
static PetscErrorCode PCHPDDMMatApply_Private(PC_HPDDM_Level *ctx, Mat Y, PetscBool *reset)
{
  Mat            A, *ptr;
  PetscScalar   *array;
  PetscInt       m, M, N, prev = 0;
  PetscContainer container = nullptr;

  PetscFunctionBegin;
  PetscCall(KSPGetOperators(ctx->ksp, &A, nullptr));
  PetscCall(MatGetSize(Y, nullptr, &N));
  PetscCall(PetscObjectQuery((PetscObject)A, "_HPDDM_MatProduct", (PetscObject *)&container));
  if (container) { /* MatProduct container already attached */
    PetscCall(PetscContainerGetPointer(container, (void **)&ptr));
    if (ptr[1] != ctx->V[2]) /* Mat has changed or may have been set first in KSPHPDDM */
      for (m = 0; m < 2; ++m) {
        PetscCall(MatDestroy(ctx->V + m + 1));
        ctx->V[m + 1] = ptr[m];
        PetscCall(PetscObjectReference((PetscObject)ctx->V[m + 1]));
      }
  }
  if (ctx->V[1]) PetscCall(MatGetSize(ctx->V[1], nullptr, &prev));
  if (N != prev || !ctx->V[0]) {
    PetscCall(MatDestroy(ctx->V));
    PetscCall(VecGetLocalSize(ctx->v[0][0], &m));
    PetscCall(MatCreateDense(PetscObjectComm((PetscObject)Y), m, PETSC_DECIDE, PETSC_DECIDE, N, nullptr, ctx->V));
    if (N != prev) {
      PetscCall(MatDestroy(ctx->V + 1));
      PetscCall(MatDestroy(ctx->V + 2));
      PetscCall(MatGetLocalSize(Y, &m, nullptr));
      PetscCall(MatGetSize(Y, &M, nullptr));
      if (ctx->parent->correction != PC_HPDDM_COARSE_CORRECTION_BALANCED) PetscCall(MatDenseGetArrayWrite(ctx->V[0], &array));
      else array = nullptr;
      PetscCall(MatCreateDense(PetscObjectComm((PetscObject)Y), m, PETSC_DECIDE, M, N, array, ctx->V + 1));
      if (ctx->parent->correction != PC_HPDDM_COARSE_CORRECTION_BALANCED) PetscCall(MatDenseRestoreArrayWrite(ctx->V[0], &array));
      PetscCall(MatCreateDense(PetscObjectComm((PetscObject)Y), m, PETSC_DECIDE, M, N, nullptr, ctx->V + 2));
      PetscCall(MatProductCreateWithMat(A, !transpose ? Y : ctx->V[2], nullptr, ctx->V[1]));
      PetscCall(MatProductSetType(ctx->V[1], !transpose ? MATPRODUCT_AB : MATPRODUCT_AtB));
      PetscCall(MatProductSetFromOptions(ctx->V[1]));
      PetscCall(MatProductSymbolic(ctx->V[1]));
      if (!container) PetscCall(PetscObjectContainerCompose((PetscObject)A, "_HPDDM_MatProduct", ctx->V + 1, nullptr)); /* no MatProduct container attached, create one to be queried in KSPHPDDM or at the next call to PCMatApply() */
      else PetscCall(PetscContainerSetPointer(container, ctx->V + 1));                                                  /* need to compose B and D from MatProductCreateWithMat(A, B, NULL, D), which are stored in the contiguous array ctx->V */
    }
    if (ctx->parent->correction == PC_HPDDM_COARSE_CORRECTION_BALANCED) {
      PetscCall(MatProductCreateWithMat(A, !transpose ? ctx->V[1] : Y, nullptr, ctx->V[2]));
      PetscCall(MatProductSetType(ctx->V[2], !transpose ? MATPRODUCT_AtB : MATPRODUCT_AB));
      PetscCall(MatProductSetFromOptions(ctx->V[2]));
      PetscCall(MatProductSymbolic(ctx->V[2]));
    }
    ctx->P->start(N);
  }
  if (N == prev || container) { /* when MatProduct container is attached, always need to MatProductReplaceMats() since KSPHPDDM may have replaced the Mat as well */
    PetscCall(MatProductReplaceMats(nullptr, !transpose ? Y : ctx->V[2], nullptr, ctx->V[1]));
    if (container && ctx->parent->correction != PC_HPDDM_COARSE_CORRECTION_BALANCED) {
      PetscCall(MatDenseGetArrayWrite(ctx->V[0], &array));
      PetscCall(MatDensePlaceArray(ctx->V[1], array));
      PetscCall(MatDenseRestoreArrayWrite(ctx->V[0], &array));
      *reset = PETSC_TRUE;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
     PCMatApply_HPDDMShell - Variant of PCApply_HPDDMShell() for blocks of vectors.

   Input Parameters:
+     pc - preconditioner context
-     X - block of input vectors

   Output Parameter:
.     Y - block of output vectors

   Level: advanced

.seealso: [](ch_ksp), `PCHPDDM`, `PCApply_HPDDMShell()`, `PCHPDDMCoarseCorrectionType`
*/
static PetscErrorCode PCMatApply_HPDDMShell(PC pc, Mat X, Mat Y)
{
  PC_HPDDM_Level *ctx;
  PetscBool       reset = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(PCShellGetContext(pc, &ctx));
  PetscCheck(ctx->P, PETSC_COMM_SELF, PETSC_ERR_PLIB, "PCSHELL from PCHPDDM called with no HPDDM object");
  if (ctx->parent->correction == PC_HPDDM_COARSE_CORRECTION_NONE) PetscCall(PCMatApply(ctx->pc, X, Y));
  else {
    PetscCall(PCHPDDMMatApply_Private<false>(ctx, Y, &reset));
    PetscCall(PCHPDDMDeflate_Private(pc, X, Y));
    if (ctx->parent->correction == PC_HPDDM_COARSE_CORRECTION_DEFLATED || ctx->parent->correction == PC_HPDDM_COARSE_CORRECTION_BALANCED) {
      PetscCall(MatProductNumeric(ctx->V[1]));
      PetscCall(MatCopy(ctx->V[1], ctx->V[2], SAME_NONZERO_PATTERN));
      PetscCall(MatAXPY(ctx->V[2], -1.0, X, SAME_NONZERO_PATTERN));
      PetscCall(PCMatApply(ctx->pc, ctx->V[2], ctx->V[1]));
      if (ctx->parent->correction == PC_HPDDM_COARSE_CORRECTION_BALANCED) {
        PetscCall(MatProductNumeric(ctx->V[2]));
        PetscCall(PCHPDDMDeflate_Private(pc, ctx->V[2], ctx->V[2]));
        PetscCall(MatAXPY(ctx->V[1], -1.0, ctx->V[2], SAME_NONZERO_PATTERN));
      }
      PetscCall(MatAXPY(Y, -1.0, ctx->V[1], SAME_NONZERO_PATTERN));
    } else {
      PetscCheck(ctx->parent->correction == PC_HPDDM_COARSE_CORRECTION_ADDITIVE, PetscObjectComm((PetscObject)pc), PETSC_ERR_PLIB, "PCSHELL from PCHPDDM called with an unknown PCHPDDMCoarseCorrectionType %d", ctx->parent->correction);
      PetscCall(PCMatApply(ctx->pc, X, ctx->V[1]));
      PetscCall(MatAXPY(Y, 1.0, ctx->V[1], SAME_NONZERO_PATTERN));
    }
    if (reset) PetscCall(MatDenseResetArray(ctx->V[1]));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
     PCApplyTranspose_HPDDMShell - Applies the transpose of a (2) deflated, (1) additive, (3) balanced, or (4) no coarse correction. In what follows, E = Z Pmat Z^T and Q = Z^T E^-1 Z.

.vb
   (1) y =                  Pmat^-T                x + Q^T x,
   (2) y = (I - Q^T Amat^T) Pmat^-T                x + Q^T x (default),
   (3) y = (I - Q^T Amat^T) Pmat^-T (I - Amat Q^T) x + Q^T x,
   (4) y =                  Pmat^-T                x        .
.ve

   Input Parameters:
+     pc - preconditioner context
-     x - input vector

   Output Parameter:
.     y - output vector

   Level: advanced

   Developer Note:
   Since this is not an actual manual page the material below should be moved to an appropriate manual page with the appropriate context, i.e. explaining when it is used and how
   to trigger it. Likely the manual page is `PCHPDDM`

.seealso: [](ch_ksp), `PCHPDDM`, `PCApply_HPDDMShell()`, `PCHPDDMCoarseCorrectionType`
*/
static PetscErrorCode PCApplyTranspose_HPDDMShell(PC pc, Vec x, Vec y)
{
  PC_HPDDM_Level *ctx;
  Mat             A;

  PetscFunctionBegin;
  PetscCall(PCShellGetContext(pc, &ctx));
  PetscCheck(ctx->P, PETSC_COMM_SELF, PETSC_ERR_PLIB, "PCSHELL from PCHPDDM called with no HPDDM object");
  PetscCheck(!ctx->parent->normal, PetscObjectComm((PetscObject)pc), PETSC_ERR_SUP, "Not implemented for the normal equations");
  PetscCall(KSPGetOperators(ctx->ksp, &A, nullptr));
  if (ctx->parent->correction == PC_HPDDM_COARSE_CORRECTION_NONE) PetscCall(PCApplyTranspose(ctx->pc, x, y)); /* y = M^-T x */
  else {
    PetscCall(PCHPDDMDeflate_Private<true>(pc, x, y)); /* y = Q^T x */
    if (ctx->parent->correction == PC_HPDDM_COARSE_CORRECTION_DEFLATED || ctx->parent->correction == PC_HPDDM_COARSE_CORRECTION_BALANCED) {
      if (ctx->parent->correction == PC_HPDDM_COARSE_CORRECTION_BALANCED) {
        PetscCall(MatMult(A, y, ctx->v[1][0]));                                /* y = A Q^T x                 */
        PetscCall(VecWAXPY(ctx->v[1][1], -1.0, ctx->v[1][0], x));              /* y = (I - A Q^T) x           */
        PetscCall(PCApplyTranspose(ctx->pc, ctx->v[1][1], ctx->v[1][0]));      /* y = M^-T (I - A Q^T) x      */
      } else PetscCall(PCApplyTranspose(ctx->pc, x, ctx->v[1][0]));            /* y = M^-T x                  */
      PetscCall(MatMultHermitianTranspose(A, ctx->v[1][0], ctx->v[1][1]));     /* z = A^T y                   */
      PetscCall(PCHPDDMDeflate_Private<true>(pc, ctx->v[1][1], ctx->v[1][1])); /* z = Q^T z                   */
      PetscCall(VecAXPBYPCZ(y, -1.0, 1.0, 1.0, ctx->v[1][1], ctx->v[1][0]));   /* y = (I - Q^T A^T) y + Q^T x */
    } else {
      PetscCheck(ctx->parent->correction == PC_HPDDM_COARSE_CORRECTION_ADDITIVE, PetscObjectComm((PetscObject)pc), PETSC_ERR_PLIB, "PCSHELL from PCHPDDM called with an unknown PCHPDDMCoarseCorrectionType %d", ctx->parent->correction);
      PetscCall(PCApplyTranspose(ctx->pc, x, ctx->v[1][0]));
      PetscCall(VecAXPY(y, 1.0, ctx->v[1][0])); /* y = M^-T x + Q^T x */
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
     PCMatApplyTranspose_HPDDMShell - Variant of PCApplyTranspose_HPDDMShell() for blocks of vectors.

   Input Parameters:
+     pc - preconditioner context
-     X - block of input vectors

   Output Parameter:
.     Y - block of output vectors

   Level: advanced

.seealso: [](ch_ksp), `PCHPDDM`, `PCApplyTranspose_HPDDMShell()`, `PCHPDDMCoarseCorrectionType`
*/
static PetscErrorCode PCMatApplyTranspose_HPDDMShell(PC pc, Mat X, Mat Y)
{
  PC_HPDDM_Level *ctx;
  PetscScalar    *array;
  PetscBool       reset = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(PCShellGetContext(pc, &ctx));
  PetscCheck(ctx->P, PETSC_COMM_SELF, PETSC_ERR_PLIB, "PCSHELL from PCHPDDM called with no HPDDM object");
  if (ctx->parent->correction == PC_HPDDM_COARSE_CORRECTION_NONE) PetscCall(PCMatApplyTranspose(ctx->pc, X, Y));
  else {
    PetscCall(PCHPDDMMatApply_Private<true>(ctx, Y, &reset));
    PetscCall(PCHPDDMDeflate_Private<true>(pc, X, Y));
    if (ctx->parent->correction == PC_HPDDM_COARSE_CORRECTION_DEFLATED || ctx->parent->correction == PC_HPDDM_COARSE_CORRECTION_BALANCED) {
      if (ctx->parent->correction == PC_HPDDM_COARSE_CORRECTION_BALANCED) {
        PetscCall(MatProductNumeric(ctx->V[2]));
        PetscCall(MatCopy(ctx->V[2], ctx->V[1], SAME_NONZERO_PATTERN));
        PetscCall(MatAYPX(ctx->V[1], -1.0, X, SAME_NONZERO_PATTERN));
        PetscCall(PCMatApplyTranspose(ctx->pc, ctx->V[1], ctx->V[2]));
      } else PetscCall(PCMatApplyTranspose(ctx->pc, X, ctx->V[2]));
      PetscCall(MatAXPY(Y, 1.0, ctx->V[2], SAME_NONZERO_PATTERN));
      PetscCall(MatProductNumeric(ctx->V[1]));
      /* ctx->V[0] and ctx->V[1] memory regions overlap, so need to copy to ctx->V[2] and switch array */
      PetscCall(MatCopy(ctx->V[1], ctx->V[2], SAME_NONZERO_PATTERN));
      if (reset) PetscCall(MatDenseResetArray(ctx->V[1]));
      PetscCall(MatDenseGetArrayWrite(ctx->V[2], &array));
      PetscCall(MatDensePlaceArray(ctx->V[1], array));
      PetscCall(MatDenseRestoreArrayWrite(ctx->V[2], &array));
      reset = PETSC_TRUE;
      PetscCall(PCHPDDMDeflate_Private<true>(pc, ctx->V[1], ctx->V[1]));
      PetscCall(MatAXPY(Y, -1.0, ctx->V[1], SAME_NONZERO_PATTERN));
    } else {
      PetscCheck(ctx->parent->correction == PC_HPDDM_COARSE_CORRECTION_ADDITIVE, PetscObjectComm((PetscObject)pc), PETSC_ERR_PLIB, "PCSHELL from PCHPDDM called with an unknown PCHPDDMCoarseCorrectionType %d", ctx->parent->correction);
      PetscCall(PCMatApplyTranspose(ctx->pc, X, ctx->V[1]));
      PetscCall(MatAXPY(Y, 1.0, ctx->V[1], SAME_NONZERO_PATTERN));
    }
    if (reset) PetscCall(MatDenseResetArray(ctx->V[1]));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCDestroy_HPDDMShell(PC pc)
{
  PC_HPDDM_Level *ctx;

  PetscFunctionBegin;
  PetscCall(PCShellGetContext(pc, &ctx));
  PetscCall(HPDDM::Schwarz<PetscScalar>::destroy(ctx, PETSC_TRUE));
  PetscCall(VecDestroyVecs(1, &ctx->v[0]));
  PetscCall(VecDestroyVecs(2, &ctx->v[1]));
  PetscCall(PetscObjectCompose((PetscObject)ctx->pc->mat, "_HPDDM_MatProduct", nullptr));
  PetscCall(MatDestroy(ctx->V));
  PetscCall(MatDestroy(ctx->V + 1));
  PetscCall(MatDestroy(ctx->V + 2));
  PetscCall(VecDestroy(&ctx->D));
  PetscCall(PetscSFDestroy(&ctx->scatter));
  PetscCall(PCDestroy(&ctx->pc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <class Type, bool T = false, typename std::enable_if<std::is_same<Type, Vec>::value>::type * = nullptr>
static inline PetscErrorCode PCApply_Schur_Private(std::tuple<KSP, IS, Vec[2]> *p, PC factor, Type x, Type y)
{
  PetscFunctionBegin;
  PetscCall(VecISCopy(std::get<2>(*p)[0], std::get<1>(*p), SCATTER_FORWARD, x));
  if (!T) PetscCall(PCApply(factor, std::get<2>(*p)[0], std::get<2>(*p)[1]));
  else PetscCall(PCApplyTranspose(factor, std::get<2>(*p)[0], std::get<2>(*p)[1]));
  PetscCall(VecISCopy(std::get<2>(*p)[1], std::get<1>(*p), SCATTER_REVERSE, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <class Type, bool = false, typename std::enable_if<std::is_same<Type, Mat>::value>::type * = nullptr>
static inline PetscErrorCode PCApply_Schur_Private(std::tuple<KSP, IS, Vec[2]> *p, PC factor, Type X, Type Y)
{
  Mat B[2];
  Vec x, y;

  PetscFunctionBegin;
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, factor->mat->rmap->n, X->cmap->n, nullptr, B));
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, factor->mat->rmap->n, X->cmap->n, nullptr, B + 1));
  for (PetscInt i = 0; i < X->cmap->n; ++i) {
    PetscCall(MatDenseGetColumnVecRead(X, i, &x));
    PetscCall(MatDenseGetColumnVecWrite(B[0], i, &y));
    PetscCall(VecISCopy(y, std::get<1>(*p), SCATTER_FORWARD, x));
    PetscCall(MatDenseRestoreColumnVecWrite(B[0], i, &y));
    PetscCall(MatDenseRestoreColumnVecRead(X, i, &x));
  }
  PetscCall(PCMatApply(factor, B[0], B[1]));
  PetscCall(MatDestroy(B));
  for (PetscInt i = 0; i < X->cmap->n; ++i) {
    PetscCall(MatDenseGetColumnVecRead(B[1], i, &x));
    PetscCall(MatDenseGetColumnVecWrite(Y, i, &y));
    PetscCall(VecISCopy(x, std::get<1>(*p), SCATTER_REVERSE, y));
    PetscCall(MatDenseRestoreColumnVecWrite(Y, i, &y));
    PetscCall(MatDenseRestoreColumnVecRead(B[1], i, &x));
  }
  PetscCall(MatDestroy(B + 1));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <class Type = Vec, bool T = false>
static PetscErrorCode PCApply_Schur(PC pc, Type x, Type y)
{
  PC                           factor;
  Mat                          A;
  MatSolverType                type;
  PetscBool                    flg;
  std::tuple<KSP, IS, Vec[2]> *p;

  PetscFunctionBegin;
  PetscCall(PCShellGetContext(pc, &p));
  PetscCall(KSPGetPC(std::get<0>(*p), &factor));
  PetscCall(PCFactorGetMatSolverType(factor, &type));
  PetscCall(PCFactorGetMatrix(factor, &A));
  PetscCall(PetscStrcmp(type, MATSOLVERMUMPS, &flg));
  if (flg) {
    PetscCheck(PetscDefined(HAVE_MUMPS), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Inconsistent MatSolverType");
    PetscCall(MatMumpsSetIcntl(A, 26, 0));
  } else {
    PetscCall(PetscStrcmp(type, MATSOLVERMKL_PARDISO, &flg));
    PetscCheck(flg && PetscDefined(HAVE_MKL_PARDISO), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Inconsistent MatSolverType");
    flg = PETSC_FALSE;
#if PetscDefined(HAVE_MKL_PARDISO)
    PetscCall(MatMkl_PardisoSetCntl(A, 70, 1));
#endif
  }
  PetscCall(PCApply_Schur_Private<Type, T>(p, factor, x, y));
  if (flg) {
    PetscCall(MatMumpsSetIcntl(A, 26, -1));
  } else {
#if PetscDefined(HAVE_MKL_PARDISO)
    PetscCall(MatMkl_PardisoSetCntl(A, 70, 0));
#endif
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCDestroy_Schur(PC pc)
{
  std::tuple<KSP, IS, Vec[2]> *p;

  PetscFunctionBegin;
  PetscCall(PCShellGetContext(pc, &p));
  PetscCall(ISDestroy(&std::get<1>(*p)));
  PetscCall(VecDestroy(std::get<2>(*p)));
  PetscCall(VecDestroy(std::get<2>(*p) + 1));
  PetscCall(PetscFree(p));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <bool transpose>
static PetscErrorCode PCHPDDMSolve_Private(const PC_HPDDM_Level *ctx, PetscScalar *rhs, const unsigned short &mu)
{
  Mat      B, X;
  PetscInt n, N, j = 0;

  PetscFunctionBegin;
  PetscCall(KSPGetOperators(ctx->ksp, &B, nullptr));
  PetscCall(MatGetLocalSize(B, &n, nullptr));
  PetscCall(MatGetSize(B, &N, nullptr));
  if (ctx->parent->log_separate) {
    j = std::distance(ctx->parent->levels, std::find(ctx->parent->levels, ctx->parent->levels + ctx->parent->N, ctx));
    PetscCall(PetscLogEventBegin(PC_HPDDM_Solve[j], ctx->ksp, nullptr, nullptr, nullptr));
  }
  if (mu == 1) {
    if (!ctx->ksp->vec_rhs) {
      PetscCall(VecCreateMPIWithArray(PetscObjectComm((PetscObject)ctx->ksp), 1, n, N, nullptr, &ctx->ksp->vec_rhs));
      PetscCall(VecCreateMPI(PetscObjectComm((PetscObject)ctx->ksp), n, N, &ctx->ksp->vec_sol));
    }
    PetscCall(VecPlaceArray(ctx->ksp->vec_rhs, rhs));
    if (!transpose) PetscCall(KSPSolve(ctx->ksp, nullptr, nullptr));
    else PetscCall(KSPSolveTranspose(ctx->ksp, nullptr, nullptr));
    PetscCall(VecCopy(ctx->ksp->vec_sol, ctx->ksp->vec_rhs));
    PetscCall(VecResetArray(ctx->ksp->vec_rhs));
  } else {
    PetscCall(MatCreateDense(PetscObjectComm((PetscObject)ctx->ksp), n, PETSC_DECIDE, N, mu, rhs, &B));
    PetscCall(MatCreateDense(PetscObjectComm((PetscObject)ctx->ksp), n, PETSC_DECIDE, N, mu, nullptr, &X));
    PetscCall(KSPMatSolve(ctx->ksp, B, X));
    PetscCall(MatCopy(X, B, SAME_NONZERO_PATTERN));
    PetscCall(MatDestroy(&X));
    PetscCall(MatDestroy(&B));
  }
  if (ctx->parent->log_separate) PetscCall(PetscLogEventEnd(PC_HPDDM_Solve[j], ctx->ksp, nullptr, nullptr, nullptr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCHPDDMSetUpNeumannOverlap_Private(PC pc)
{
  PC_HPDDM *data = (PC_HPDDM *)pc->data;

  PetscFunctionBegin;
  if (data->setup) {
    Mat       P;
    Vec       x, xt = nullptr;
    PetscReal t = 0.0, s = 0.0;

    PetscCall(PCGetOperators(pc, nullptr, &P));
    PetscCall(PetscObjectQuery((PetscObject)P, "__SNES_latest_X", (PetscObject *)&x));
    PetscCallBack("PCHPDDM Neumann callback", (*data->setup)(data->aux, t, x, xt, s, data->is, data->setup_ctx));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCHPDDMCreateSubMatrices_Private(Mat mat, PetscInt n, const IS *, const IS *, MatReuse scall, Mat *submat[])
{
  Mat       A;
  PetscBool flg;

  PetscFunctionBegin;
  PetscCheck(n == 1, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "MatCreateSubMatrices() called to extract %" PetscInt_FMT " submatrices, which is different than 1", n);
  /* previously composed Mat */
  PetscCall(PetscObjectQuery((PetscObject)mat, "_PCHPDDM_SubMatrices", (PetscObject *)&A));
  PetscCheck(A, PETSC_COMM_SELF, PETSC_ERR_PLIB, "SubMatrices not found in Mat");
  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATSCHURCOMPLEMENT, &flg)); /* MATSCHURCOMPLEMENT has neither a MatDuplicate() nor a MatCopy() implementation */
  if (scall == MAT_INITIAL_MATRIX) {
    PetscCall(PetscCalloc1(2, submat)); /* allocate an extra Mat to avoid errors in MatDestroySubMatrices_Dummy() */
    if (!flg) PetscCall(MatDuplicate(A, MAT_COPY_VALUES, *submat));
  } else if (!flg) PetscCall(MatCopy(A, (*submat)[0], SAME_NONZERO_PATTERN));
  if (flg) {
    PetscCall(MatDestroy(*submat)); /* previously created Mat has to be destroyed */
    (*submat)[0] = A;
    PetscCall(PetscObjectReference((PetscObject)A));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCHPDDMCommunicationAvoidingPCASM_Private(PC pc, Mat C, PetscBool sorted)
{
  PetscErrorCodeFn *op;

  PetscFunctionBegin;
  /* previously-composed Mat */
  PetscCall(PetscObjectCompose((PetscObject)pc->pmat, "_PCHPDDM_SubMatrices", (PetscObject)C));
  PetscCall(MatGetOperation(pc->pmat, MATOP_CREATE_SUBMATRICES, &op));
  /* trick suggested by Barry https://lists.mcs.anl.gov/pipermail/petsc-dev/2020-January/025491.html */
  PetscCall(MatSetOperation(pc->pmat, MATOP_CREATE_SUBMATRICES, (PetscErrorCodeFn *)PCHPDDMCreateSubMatrices_Private));
  if (sorted) PetscCall(PCASMSetSortIndices(pc, PETSC_FALSE)); /* everything is already sorted */
  PetscCall(PCSetFromOptions(pc));                             /* otherwise -pc_hpddm_levels_1_pc_asm_sub_mat_type is not used */
  PetscCall(PCSetUp(pc));
  /* reset MatCreateSubMatrices() */
  PetscCall(MatSetOperation(pc->pmat, MATOP_CREATE_SUBMATRICES, op));
  PetscCall(PetscObjectCompose((PetscObject)pc->pmat, "_PCHPDDM_SubMatrices", nullptr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCHPDDMPermute_Private(IS is, IS in_is, IS *out_is, Mat in_C, Mat *out_C, IS *p)
{
  IS                           perm;
  const PetscInt              *ptr;
  PetscInt                    *concatenate, size, bs;
  std::map<PetscInt, PetscInt> order;
  PetscBool                    sorted;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is, IS_CLASSID, 1);
  PetscValidHeaderSpecific(in_C, MAT_CLASSID, 4);
  PetscCall(ISSorted(is, &sorted));
  if (!sorted) {
    PetscCall(ISGetLocalSize(is, &size));
    PetscCall(ISGetIndices(is, &ptr));
    PetscCall(ISGetBlockSize(is, &bs));
    /* MatCreateSubMatrices(), called by PCASM, follows the global numbering of Pmat */
    for (PetscInt n = 0; n < size; n += bs) order.insert(std::make_pair(ptr[n] / bs, n / bs));
    PetscCall(ISRestoreIndices(is, &ptr));
    size /= bs;
    if (out_C) {
      PetscCall(PetscMalloc1(size, &concatenate));
      for (const std::pair<const PetscInt, PetscInt> &i : order) *concatenate++ = i.second;
      concatenate -= size;
      PetscCall(ISCreateBlock(PetscObjectComm((PetscObject)in_C), bs, size, concatenate, PETSC_OWN_POINTER, &perm));
      PetscCall(ISSetPermutation(perm));
      /* permute user-provided Mat so that it matches with MatCreateSubMatrices() numbering */
      PetscCall(MatPermute(in_C, perm, perm, out_C));
      if (p) *p = perm;
      else PetscCall(ISDestroy(&perm)); /* no need to save the permutation */
    }
    if (out_is) {
      PetscCall(PetscMalloc1(size, &concatenate));
      for (const std::pair<const PetscInt, PetscInt> &i : order) *concatenate++ = i.first;
      concatenate -= size;
      /* permute user-provided IS so that it matches with MatCreateSubMatrices() numbering */
      PetscCall(ISCreateBlock(PetscObjectComm((PetscObject)in_is), bs, size, concatenate, PETSC_OWN_POINTER, out_is));
    }
  } else { /* input IS is sorted, nothing to permute, simply duplicate inputs when needed */
    if (out_C) PetscCall(MatDuplicate(in_C, MAT_COPY_VALUES, out_C));
    if (out_is) PetscCall(ISDuplicate(in_is, out_is));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCHPDDMCheckSymmetry_Private(PC pc, Mat A01, Mat A10, Mat *B01 = nullptr)
{
  Mat       T, U = nullptr, B = nullptr;
  IS        z;
  PetscBool flg, conjugate = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)A10, MATTRANSPOSEVIRTUAL, &flg));
  if (B01) *B01 = nullptr;
  if (flg) {
    PetscCall(MatShellGetScalingShifts(A10, (PetscScalar *)MAT_SHELL_NOT_ALLOWED, (PetscScalar *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Mat *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED));
    PetscCall(MatTransposeGetMat(A10, &U));
  } else {
    PetscCall(PetscObjectTypeCompare((PetscObject)A10, MATHERMITIANTRANSPOSEVIRTUAL, &flg));
    if (flg) {
      PetscCall(MatShellGetScalingShifts(A10, (PetscScalar *)MAT_SHELL_NOT_ALLOWED, (PetscScalar *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Mat *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED));
      PetscCall(MatHermitianTransposeGetMat(A10, &U));
      conjugate = PETSC_TRUE;
    }
  }
  if (U) PetscCall(MatDuplicate(U, MAT_COPY_VALUES, &T));
  else PetscCall(MatHermitianTranspose(A10, MAT_INITIAL_MATRIX, &T));
  PetscCall(PetscObjectTypeCompare((PetscObject)A01, MATTRANSPOSEVIRTUAL, &flg));
  if (flg) {
    PetscCall(MatShellGetScalingShifts(A01, (PetscScalar *)MAT_SHELL_NOT_ALLOWED, (PetscScalar *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Mat *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED));
    PetscCall(MatTransposeGetMat(A01, &A01));
    PetscCall(MatTranspose(A01, MAT_INITIAL_MATRIX, &B));
    A01 = B;
  } else {
    PetscCall(PetscObjectTypeCompare((PetscObject)A01, MATHERMITIANTRANSPOSEVIRTUAL, &flg));
    if (flg) {
      PetscCall(MatShellGetScalingShifts(A01, (PetscScalar *)MAT_SHELL_NOT_ALLOWED, (PetscScalar *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Mat *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED));
      PetscCall(MatHermitianTransposeGetMat(A01, &A01));
      PetscCall(MatHermitianTranspose(A01, MAT_INITIAL_MATRIX, &B));
      A01 = B;
    }
  }
  PetscCall(PetscLayoutCompare(T->rmap, A01->rmap, &flg));
  if (flg) {
    PetscCall(PetscLayoutCompare(T->cmap, A01->cmap, &flg));
    if (flg) {
      PetscCall(MatFindZeroRows(A01, &z)); /* for essential boundary conditions, some implementations will */
      if (z) {                             /*  zero rows in [P00 A01] except for the diagonal of P00       */
        if (B01) PetscCall(MatDuplicate(T, MAT_COPY_VALUES, B01));
        PetscCall(MatSetOption(T, MAT_NO_OFF_PROC_ZERO_ROWS, PETSC_TRUE));
        PetscCall(MatZeroRowsIS(T, z, 0.0, nullptr, nullptr)); /* corresponding zero rows from A01 */
      }
      PetscCall(MatMultEqual(A01, T, 20, &flg));
      if (!B01) PetscCheck(flg, PetscObjectComm((PetscObject)pc), PETSC_ERR_SUP, "A01 != A10^T");
      else {
        PetscCall(PetscInfo(pc, "A01 and A10^T are equal? %s\n", PetscBools[flg]));
        if (!flg) {
          if (z) PetscCall(MatDestroy(&T));
          else *B01 = T;
          flg = PETSC_TRUE;
        } else PetscCall(MatDestroy(B01));
      }
      PetscCall(ISDestroy(&z));
    }
  }
  if (!flg) PetscCall(PetscInfo(pc, "A01 and A10^T have non-congruent layouts, cannot test for equality\n"));
  if (!B01 || !*B01) PetscCall(MatDestroy(&T));
  else if (conjugate) PetscCall(MatConjugate(T));
  PetscCall(MatDestroy(&B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCHPDDMCheckInclusion_Private(PC pc, IS is, IS is_local, PetscBool check)
{
  IS          intersect;
  const char *str = "IS of the auxiliary Mat does not include all local rows of A";
  PetscBool   equal;

  PetscFunctionBegin;
  PetscCall(ISIntersect(is, is_local, &intersect));
  PetscCall(ISEqualUnsorted(is_local, intersect, &equal));
  PetscCall(ISDestroy(&intersect));
  if (check) PetscCheck(equal, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "%s", str);
  else if (!equal) PetscCall(PetscInfo(pc, "%s\n", str));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCHPDDMDestroySubMatrices_Private(PetscBool flg, PetscBool algebraic, Mat *sub)
{
  IS is;

  PetscFunctionBegin;
  if (!flg) {
    if (algebraic) {
      PetscCall(PetscObjectQuery((PetscObject)sub[0], "_PCHPDDM_Embed", (PetscObject *)&is));
      PetscCall(ISDestroy(&is));
      PetscCall(PetscObjectCompose((PetscObject)sub[0], "_PCHPDDM_Embed", nullptr));
      PetscCall(PetscObjectCompose((PetscObject)sub[0], "_PCHPDDM_Compact", nullptr));
    }
    PetscCall(MatDestroySubMatrices(algebraic ? 2 : 1, &sub));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCHPDDMAlgebraicAuxiliaryMat_Private(Mat P, IS *is, Mat *sub[], PetscBool block)
{
  IS         icol[3], irow[2];
  Mat       *M, Q;
  PetscReal *ptr;
  PetscInt  *idx, p = 0, bs = P->cmap->bs;
  PetscBool  flg;

  PetscFunctionBegin;
  PetscCall(ISCreateStride(PETSC_COMM_SELF, P->cmap->N, 0, 1, icol + 2));
  PetscCall(ISSetBlockSize(icol[2], bs));
  PetscCall(ISSetIdentity(icol[2]));
  PetscCall(PetscObjectTypeCompare((PetscObject)P, MATMPISBAIJ, &flg));
  if (flg) {
    /* MatCreateSubMatrices() does not handle MATMPISBAIJ properly when iscol != isrow, so convert first to MATMPIBAIJ */
    PetscCall(MatConvert(P, MATMPIBAIJ, MAT_INITIAL_MATRIX, &Q));
    std::swap(P, Q);
  }
  PetscCall(MatCreateSubMatrices(P, 1, is, icol + 2, MAT_INITIAL_MATRIX, &M));
  if (flg) {
    std::swap(P, Q);
    PetscCall(MatDestroy(&Q));
  }
  PetscCall(ISDestroy(icol + 2));
  PetscCall(ISCreateStride(PETSC_COMM_SELF, M[0]->rmap->N, 0, 1, irow));
  PetscCall(ISSetBlockSize(irow[0], bs));
  PetscCall(ISSetIdentity(irow[0]));
  if (!block) {
    PetscCall(PetscMalloc2(P->cmap->N, &ptr, P->cmap->N / bs, &idx));
    PetscCall(MatGetColumnNorms(M[0], NORM_INFINITY, ptr));
    /* check for nonzero columns so that M[0] may be expressed in compact form */
    for (PetscInt n = 0; n < P->cmap->N; n += bs) {
      if (std::find_if(ptr + n, ptr + n + bs, [](PetscReal v) { return v > PETSC_MACHINE_EPSILON; }) != ptr + n + bs) idx[p++] = n / bs;
    }
    PetscCall(ISCreateBlock(PETSC_COMM_SELF, bs, p, idx, PETSC_USE_POINTER, icol + 1));
    PetscCall(ISSetInfo(icol[1], IS_SORTED, IS_GLOBAL, PETSC_TRUE, PETSC_TRUE));
    PetscCall(ISEmbed(*is, icol[1], PETSC_FALSE, icol + 2));
    irow[1] = irow[0];
    /* first Mat will be used in PCASM (if it is used as a PC on this level) and as the left-hand side of GenEO */
    icol[0] = is[0];
    PetscCall(MatCreateSubMatrices(M[0], 2, irow, icol, MAT_INITIAL_MATRIX, sub));
    PetscCall(ISDestroy(icol + 1));
    PetscCall(PetscFree2(ptr, idx));
    /* IS used to go back and forth between the augmented and the original local linear system, see eq. (3.4) of [2022b] */
    PetscCall(PetscObjectCompose((PetscObject)(*sub)[0], "_PCHPDDM_Embed", (PetscObject)icol[2]));
    /* Mat used in eq. (3.1) of [2022b] */
    PetscCall(PetscObjectCompose((PetscObject)(*sub)[0], "_PCHPDDM_Compact", (PetscObject)(*sub)[1]));
  } else {
    Mat aux;

    PetscCall(MatSetOption(M[0], MAT_SUBMAT_SINGLEIS, PETSC_TRUE));
    /* diagonal block of the overlapping rows */
    PetscCall(MatCreateSubMatrices(M[0], 1, irow, is, MAT_INITIAL_MATRIX, sub));
    PetscCall(MatDuplicate((*sub)[0], MAT_COPY_VALUES, &aux));
    PetscCall(MatSetOption(aux, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
    if (bs == 1) { /* scalar case */
      Vec sum[2];

      PetscCall(MatCreateVecs(aux, sum, sum + 1));
      PetscCall(MatGetRowSum(M[0], sum[0]));
      PetscCall(MatGetRowSum(aux, sum[1]));
      /* off-diagonal block row sum (full rows - diagonal block rows) */
      PetscCall(VecAXPY(sum[0], -1.0, sum[1]));
      /* subdomain matrix plus off-diagonal block row sum */
      PetscCall(MatDiagonalSet(aux, sum[0], ADD_VALUES));
      PetscCall(VecDestroy(sum));
      PetscCall(VecDestroy(sum + 1));
    } else { /* vectorial case */
      /* TODO: missing MatGetValuesBlocked(), so the code below is     */
      /* an extension of the scalar case for when bs > 1, but it could */
      /* be more efficient by avoiding all these MatMatMult()          */
      Mat          sum[2], ones;
      PetscScalar *ptr;

      PetscCall(PetscCalloc1(M[0]->cmap->n * bs, &ptr));
      PetscCall(MatCreateDense(PETSC_COMM_SELF, M[0]->cmap->n, bs, M[0]->cmap->n, bs, ptr, &ones));
      for (PetscInt n = 0; n < M[0]->cmap->n; n += bs) {
        for (p = 0; p < bs; ++p) ptr[n + p * (M[0]->cmap->n + 1)] = 1.0;
      }
      PetscCall(MatMatMult(M[0], ones, MAT_INITIAL_MATRIX, PETSC_CURRENT, sum));
      PetscCall(MatDestroy(&ones));
      PetscCall(MatCreateDense(PETSC_COMM_SELF, aux->cmap->n, bs, aux->cmap->n, bs, ptr, &ones));
      PetscCall(MatDenseSetLDA(ones, M[0]->cmap->n));
      PetscCall(MatMatMult(aux, ones, MAT_INITIAL_MATRIX, PETSC_CURRENT, sum + 1));
      PetscCall(MatDestroy(&ones));
      PetscCall(PetscFree(ptr));
      /* off-diagonal block row sum (full rows - diagonal block rows) */
      PetscCall(MatAXPY(sum[0], -1.0, sum[1], SAME_NONZERO_PATTERN));
      PetscCall(MatDestroy(sum + 1));
      /* re-order values to be consistent with MatSetValuesBlocked()           */
      /* equivalent to MatTranspose() which does not truly handle              */
      /* MAT_INPLACE_MATRIX in the rectangular case, as it calls PetscMalloc() */
      PetscCall(MatDenseGetArrayWrite(sum[0], &ptr));
      HPDDM::Wrapper<PetscScalar>::imatcopy<'T'>(bs, sum[0]->rmap->n, ptr, sum[0]->rmap->n, bs);
      /* subdomain matrix plus off-diagonal block row sum */
      for (PetscInt n = 0; n < aux->cmap->n / bs; ++n) PetscCall(MatSetValuesBlocked(aux, 1, &n, 1, &n, ptr + n * bs * bs, ADD_VALUES));
      PetscCall(MatAssemblyBegin(aux, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(aux, MAT_FINAL_ASSEMBLY));
      PetscCall(MatDenseRestoreArrayWrite(sum[0], &ptr));
      PetscCall(MatDestroy(sum));
    }
    PetscCall(MatSetOption(aux, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));
    /* left-hand side of GenEO, with the same sparsity pattern as PCASM subdomain solvers  */
    PetscCall(PetscObjectCompose((PetscObject)(*sub)[0], "_PCHPDDM_Neumann_Mat", (PetscObject)aux));
  }
  PetscCall(ISDestroy(irow));
  PetscCall(MatDestroySubMatrices(1, &M));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCApply_Nest(PC pc, Vec x, Vec y)
{
  Mat                    A;
  MatSolverType          type;
  IS                     is[2];
  PetscBool              flg;
  std::pair<PC, Vec[2]> *p;

  PetscFunctionBegin;
  PetscCall(PCShellGetContext(pc, &p));
  if (p->second[0]) { /* in case of a centralized Schur complement, some processes may have no local operator */
    PetscCall(PCGetOperators(p->first, &A, nullptr));
    PetscCall(MatNestGetISs(A, is, nullptr));
    PetscCall(PetscObjectTypeCompareAny((PetscObject)p->first, &flg, PCLU, PCCHOLESKY, ""));
    if (flg) { /* partial solve currently only makes sense with exact factorizations */
      PetscCall(PCFactorGetMatSolverType(p->first, &type));
      PetscCall(PCFactorGetMatrix(p->first, &A));
      if (A->schur) {
        PetscCall(PetscStrcmp(type, MATSOLVERMUMPS, &flg));
        if (flg) PetscCall(MatMumpsSetIcntl(A, 26, 1)); /* reduction/condensation phase followed by Schur complement solve */
      } else flg = PETSC_FALSE;
    }
    PetscCall(VecISCopy(p->second[0], is[1], SCATTER_FORWARD, x)); /* assign the RHS associated to the Schur complement */
    PetscCall(PCApply(p->first, p->second[0], p->second[1]));
    PetscCall(VecISCopy(p->second[1], is[1], SCATTER_REVERSE, y)); /* retrieve the partial solution associated to the Schur complement */
    if (flg) PetscCall(MatMumpsSetIcntl(A, 26, -1));               /* default ICNTL(26) value in PETSc */
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCView_Nest(PC pc, PetscViewer viewer)
{
  std::pair<PC, Vec[2]> *p;

  PetscFunctionBegin;
  PetscCall(PCShellGetContext(pc, &p));
  PetscCall(PCView(p->first, viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCDestroy_Nest(PC pc)
{
  std::pair<PC, Vec[2]> *p;

  PetscFunctionBegin;
  PetscCall(PCShellGetContext(pc, &p));
  PetscCall(VecDestroy(p->second));
  PetscCall(VecDestroy(p->second + 1));
  PetscCall(PCDestroy(&p->first));
  PetscCall(PetscFree(p));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <bool T = false>
static PetscErrorCode MatMult_Schur(Mat A, Vec x, Vec y)
{
  std::tuple<Mat, PetscSF, Vec[2]> *ctx;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A, &ctx));
  PetscCall(VecScatterBegin(std::get<1>(*ctx), x, std::get<2>(*ctx)[0], INSERT_VALUES, SCATTER_FORWARD)); /* local Vec with overlap */
  PetscCall(VecScatterEnd(std::get<1>(*ctx), x, std::get<2>(*ctx)[0], INSERT_VALUES, SCATTER_FORWARD));
  if (!T) PetscCall(MatMult(std::get<0>(*ctx), std::get<2>(*ctx)[0], std::get<2>(*ctx)[1])); /* local Schur complement */
  else PetscCall(MatMultTranspose(std::get<0>(*ctx), std::get<2>(*ctx)[0], std::get<2>(*ctx)[1]));
  PetscCall(VecSet(y, 0.0));
  PetscCall(VecScatterBegin(std::get<1>(*ctx), std::get<2>(*ctx)[1], y, ADD_VALUES, SCATTER_REVERSE)); /* global Vec with summed up contributions on the overlap */
  PetscCall(VecScatterEnd(std::get<1>(*ctx), std::get<2>(*ctx)[1], y, ADD_VALUES, SCATTER_REVERSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDestroy_Schur(Mat A)
{
  std::tuple<Mat, PetscSF, Vec[2]> *ctx;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A, &ctx));
  PetscCall(VecDestroy(std::get<2>(*ctx)));
  PetscCall(VecDestroy(std::get<2>(*ctx) + 1));
  PetscCall(PetscFree(ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMult_SchurCorrection(Mat A, Vec x, Vec y)
{
  PC                                         pc;
  std::tuple<PC[2], Mat[2], PCSide, Vec[3]> *ctx;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A, &ctx));
  pc = ((PC_HPDDM *)std::get<0>(*ctx)[0]->data)->levels[0]->ksp->pc;
  if (std::get<2>(*ctx) == PC_LEFT || std::get<2>(*ctx) == PC_SIDE_DEFAULT) {             /* Q_0 is the coarse correction associated to the A00 block from PCFIELDSPLIT */
    PetscCall(MatMult(std::get<1>(*ctx)[0], x, std::get<3>(*ctx)[1]));                    /*     A_01 x                 */
    PetscCall(PCHPDDMDeflate_Private(pc, std::get<3>(*ctx)[1], std::get<3>(*ctx)[1]));    /*     Q_0 A_01 x             */
    PetscCall(MatMult(std::get<1>(*ctx)[1], std::get<3>(*ctx)[1], std::get<3>(*ctx)[0])); /*     A_10 Q_0 A_01 x        */
    PetscCall(PCApply(std::get<0>(*ctx)[1], std::get<3>(*ctx)[0], y));                    /* y = M_S^-1 A_10 Q_0 A_01 x */
  } else {
    PetscCall(PCApply(std::get<0>(*ctx)[1], x, std::get<3>(*ctx)[0]));                    /*     M_S^-1 x               */
    PetscCall(MatMult(std::get<1>(*ctx)[0], std::get<3>(*ctx)[0], std::get<3>(*ctx)[1])); /*     A_01 M_S^-1 x          */
    PetscCall(PCHPDDMDeflate_Private(pc, std::get<3>(*ctx)[1], std::get<3>(*ctx)[1]));    /*     Q_0 A_01 M_S^-1 x      */
    PetscCall(MatMult(std::get<1>(*ctx)[1], std::get<3>(*ctx)[1], y));                    /* y = A_10 Q_0 A_01 M_S^-1 x */
  }
  PetscCall(VecAXPY(y, -1.0, x)); /* y -= x, preconditioned eq. (24) of https://hal.science/hal-02343808v6/document (with a sign flip) */
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatView_SchurCorrection(Mat A, PetscViewer viewer)
{
  PetscBool                                  ascii;
  std::tuple<PC[2], Mat[2], PCSide, Vec[3]> *ctx;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &ascii));
  if (ascii) {
    PetscCall(MatShellGetContext(A, &ctx));
    PetscCall(PetscViewerASCIIPrintf(viewer, "action of %s\n", std::get<2>(*ctx) == PC_LEFT || std::get<2>(*ctx) == PC_SIDE_DEFAULT ? "(I - M_S^-1 A_10 Q_0 A_01)" : "(I - A_10 Q_0 A_01 M_S^-1)"));
    PetscCall(PCView(std::get<0>(*ctx)[1], viewer)); /* no need to PCView(Q_0) since it will be done by PCFIELDSPLIT */
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDestroy_SchurCorrection(Mat A)
{
  std::tuple<PC[2], Mat[2], PCSide, Vec[3]> *ctx;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A, &ctx));
  PetscCall(VecDestroy(std::get<3>(*ctx)));
  PetscCall(VecDestroy(std::get<3>(*ctx) + 1));
  PetscCall(VecDestroy(std::get<3>(*ctx) + 2));
  PetscCall(PCDestroy(std::get<0>(*ctx) + 1));
  PetscCall(PetscFree(ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCPostSolve_SchurPreLeastSquares(PC, KSP, Vec, Vec x)
{
  PetscFunctionBegin;
  PetscCall(VecScale(x, -1.0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPPreSolve_SchurCorrection(KSP, Vec b, Vec, void *context)
{
  std::tuple<PC[2], Mat[2], PCSide, Vec[3]> *ctx = reinterpret_cast<std::tuple<PC[2], Mat[2], PCSide, Vec[3]> *>(context);

  PetscFunctionBegin;
  if (std::get<2>(*ctx) == PC_LEFT || std::get<2>(*ctx) == PC_SIDE_DEFAULT) {
    PetscCall(PCApply(std::get<0>(*ctx)[1], b, std::get<3>(*ctx)[2]));
    std::swap(*b, *std::get<3>(*ctx)[2]); /* replace b by M^-1 b, but need to keep a copy of the original RHS, so swap it with the work Vec */
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPPostSolve_SchurCorrection(KSP, Vec b, Vec x, void *context)
{
  std::tuple<PC[2], Mat[2], PCSide, Vec[3]> *ctx = reinterpret_cast<std::tuple<PC[2], Mat[2], PCSide, Vec[3]> *>(context);

  PetscFunctionBegin;
  if (std::get<2>(*ctx) == PC_LEFT || std::get<2>(*ctx) == PC_SIDE_DEFAULT) std::swap(*b, *std::get<3>(*ctx)[2]); /* put back the original RHS where it belongs */
  else {
    PetscCall(PCApply(std::get<0>(*ctx)[1], x, std::get<3>(*ctx)[2]));
    PetscCall(VecCopy(std::get<3>(*ctx)[2], x)); /* replace x by M^-1 x */
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMult_Harmonic(Mat, Vec, Vec);
static PetscErrorCode MatMultTranspose_Harmonic(Mat, Vec, Vec);
static PetscErrorCode MatProduct_AB_Harmonic(Mat, Mat, Mat, void *);
static PetscErrorCode MatDestroy_Harmonic(Mat);

static PetscErrorCode PCSetUp_HPDDM(PC pc)
{
  PC_HPDDM                                  *data = (PC_HPDDM *)pc->data;
  PC                                         inner;
  KSP                                       *ksp;
  Mat                                       *sub, A, P, N, C = nullptr, uaux = nullptr, weighted, subA[2], S;
  Vec                                        xin, v;
  std::vector<Vec>                           initial;
  IS                                         is[1], loc, uis = data->is, unsorted = nullptr;
  ISLocalToGlobalMapping                     l2g;
  char                                       prefix[256];
  const char                                *pcpre;
  const PetscScalar *const                  *ev;
  PetscInt                                   n, requested = data->N, reused = 0, overlap = -1;
  MatStructure                               structure  = UNKNOWN_NONZERO_PATTERN;
  PetscBool                                  subdomains = PETSC_FALSE, flg = PETSC_FALSE, ismatis, swap = PETSC_FALSE, algebraic = PETSC_FALSE, block = PETSC_FALSE;
  DM                                         dm;
  std::tuple<PC[2], Mat[2], PCSide, Vec[3]> *ctx = nullptr;
#if PetscDefined(USE_DEBUG)
  IS  dis  = nullptr;
  Mat daux = nullptr;
#endif

  PetscFunctionBegin;
  PetscCheck(data->levels && data->levels[0], PETSC_COMM_SELF, PETSC_ERR_PLIB, "Not a single level allocated");
  PetscCall(PCGetOptionsPrefix(pc, &pcpre));
  PetscCall(PCGetOperators(pc, &A, &P));
  if (!data->levels[0]->ksp) {
    PetscCall(KSPCreate(PetscObjectComm((PetscObject)pc), &data->levels[0]->ksp));
    PetscCall(KSPSetNestLevel(data->levels[0]->ksp, pc->kspnestlevel));
    PetscCall(PetscSNPrintf(prefix, sizeof(prefix), "%spc_hpddm_%s_", pcpre ? pcpre : "", data->N > 1 ? "levels_1" : "coarse"));
    PetscCall(KSPSetOptionsPrefix(data->levels[0]->ksp, prefix));
    PetscCall(KSPSetType(data->levels[0]->ksp, KSPPREONLY));
  } else if (data->levels[0]->ksp->pc && data->levels[0]->ksp->pc->setupcalled && data->levels[0]->ksp->pc->reusepreconditioner) {
    /* if the fine-level PCSHELL exists, its setup has succeeded, and one wants to reuse it, */
    /* then just propagate the appropriate flag to the coarser levels                        */
    for (n = 0; n < PETSC_PCHPDDM_MAXLEVELS && data->levels[n]; ++n) {
      /* the following KSP and PC may be NULL for some processes, hence the check            */
      if (data->levels[n]->ksp) PetscCall(KSPSetReusePreconditioner(data->levels[n]->ksp, PETSC_TRUE));
      if (data->levels[n]->pc) PetscCall(PCSetReusePreconditioner(data->levels[n]->pc, PETSC_TRUE));
    }
    /* early bail out because there is nothing to do */
    PetscFunctionReturn(PETSC_SUCCESS);
  } else {
    /* reset coarser levels */
    for (n = 1; n < PETSC_PCHPDDM_MAXLEVELS && data->levels[n]; ++n) {
      if (data->levels[n]->ksp && data->levels[n]->ksp->pc && data->levels[n]->ksp->pc->setupcalled && data->levels[n]->ksp->pc->reusepreconditioner && n < data->N) {
        reused = data->N - n;
        break;
      }
      PetscCall(KSPDestroy(&data->levels[n]->ksp));
      PetscCall(PCDestroy(&data->levels[n]->pc));
    }
    /* check if some coarser levels are being reused */
    PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &reused, 1, MPIU_INT, MPI_MAX, PetscObjectComm((PetscObject)pc)));
    const int *addr = data->levels[0]->P ? data->levels[0]->P->getAddrLocal() : &HPDDM::i__0;

    if (addr != &HPDDM::i__0 && reused != data->N - 1) {
      /* reuse previously computed eigenvectors */
      ev = data->levels[0]->P->getVectors();
      if (ev) {
        initial.reserve(*addr);
        PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF, 1, data->levels[0]->P->getDof(), ev[0], &xin));
        for (n = 0; n < *addr; ++n) {
          PetscCall(VecDuplicate(xin, &v));
          PetscCall(VecPlaceArray(xin, ev[n]));
          PetscCall(VecCopy(xin, v));
          initial.emplace_back(v);
          PetscCall(VecResetArray(xin));
        }
        PetscCall(VecDestroy(&xin));
      }
    }
  }
  data->N -= reused;
  PetscCall(KSPSetOperators(data->levels[0]->ksp, A, P));

  PetscCall(PetscObjectTypeCompare((PetscObject)P, MATIS, &ismatis));
  if (!data->is && !ismatis) {
    PetscErrorCode (*create)(DM, IS *, Mat *, PetscErrorCode (**)(Mat, PetscReal, Vec, Vec, PetscReal, IS, void *), void **) = nullptr;
    PetscErrorCode (*usetup)(Mat, PetscReal, Vec, Vec, PetscReal, IS, void *)                                                = nullptr;
    void *uctx                                                                                                               = nullptr;

    /* first see if we can get the data from the DM */
    PetscCall(MatGetDM(P, &dm));
    if (!dm) PetscCall(MatGetDM(A, &dm));
    if (!dm) PetscCall(PCGetDM(pc, &dm));
    if (dm) { /* this is the hook for DMPLEX for which the auxiliary Mat is the local Neumann matrix */
      PetscCall(PetscObjectQueryFunction((PetscObject)dm, "DMCreateNeumannOverlap_C", &create));
      if (create) {
        PetscCall((*create)(dm, &uis, &uaux, &usetup, &uctx));
        if (data->Neumann == PETSC_BOOL3_UNKNOWN) data->Neumann = PETSC_BOOL3_TRUE; /* set the value only if it was not already provided by the user */
      }
    }
    if (!create) {
      if (!uis) {
        PetscCall(PetscObjectQuery((PetscObject)pc, "_PCHPDDM_Neumann_IS", (PetscObject *)&uis));
        PetscCall(PetscObjectReference((PetscObject)uis));
      }
      if (!uaux) {
        PetscCall(PetscObjectQuery((PetscObject)pc, "_PCHPDDM_Neumann_Mat", (PetscObject *)&uaux));
        PetscCall(PetscObjectReference((PetscObject)uaux));
      }
      /* look inside the Pmat instead of the PC, needed for MatSchurComplementComputeExplicitOperator() */
      if (!uis) {
        PetscCall(PetscObjectQuery((PetscObject)P, "_PCHPDDM_Neumann_IS", (PetscObject *)&uis));
        PetscCall(PetscObjectReference((PetscObject)uis));
      }
      if (!uaux) {
        PetscCall(PetscObjectQuery((PetscObject)P, "_PCHPDDM_Neumann_Mat", (PetscObject *)&uaux));
        PetscCall(PetscObjectReference((PetscObject)uaux));
      }
    }
    PetscCall(PCHPDDMSetAuxiliaryMat(pc, uis, uaux, usetup, uctx));
    PetscCall(MatDestroy(&uaux));
    PetscCall(ISDestroy(&uis));
  }

  if (!ismatis) {
    PetscCall(PCHPDDMSetUpNeumannOverlap_Private(pc));
    PetscCall(PetscOptionsGetBool(nullptr, pcpre, "-pc_hpddm_block_splitting", &block, nullptr));
    PetscCall(PetscOptionsGetInt(nullptr, pcpre, "-pc_hpddm_harmonic_overlap", &overlap, nullptr));
    PetscCall(PetscObjectTypeCompare((PetscObject)P, MATSCHURCOMPLEMENT, &flg));
    if (data->is || flg) {
      if (block || overlap != -1) {
        PetscCall(ISDestroy(&data->is));
        PetscCall(MatDestroy(&data->aux));
      } else if (flg) {
        PCHPDDMSchurPreType type = PC_HPDDM_SCHUR_PRE_GENEO;

        PetscCall(PetscOptionsGetEnum(nullptr, pcpre, "-pc_hpddm_schur_precondition", PCHPDDMSchurPreTypes, (PetscEnum *)&type, &flg));
        if (type == PC_HPDDM_SCHUR_PRE_LEAST_SQUARES) {
          PetscCall(ISDestroy(&data->is)); /* destroy any previously user-set objects since they will be set automatically */
          PetscCall(MatDestroy(&data->aux));
        } else if (type == PC_HPDDM_SCHUR_PRE_GENEO) {
          PetscContainer container = nullptr;

          PetscCall(PetscObjectQuery((PetscObject)pc, "_PCHPDDM_Schur", (PetscObject *)&container));
          if (!container) { /* first call to PCSetUp() on the PC associated to the Schur complement */
            PC_HPDDM       *data_00;
            KSP             ksp, inner_ksp;
            PC              pc_00;
            Mat             A11 = nullptr;
            Vec             d   = nullptr;
            const PetscInt *ranges;
            PetscMPIInt     size;
            char           *prefix;

            PetscCall(MatSchurComplementGetKSP(P, &ksp));
            PetscCall(KSPGetPC(ksp, &pc_00));
            PetscCall(PetscObjectTypeCompare((PetscObject)pc_00, PCHPDDM, &flg));
            PetscCheck(flg, PetscObjectComm((PetscObject)P), PETSC_ERR_ARG_INCOMP, "-%spc_hpddm_schur_precondition %s and -%spc_type %s (!= %s)", pcpre ? pcpre : "", PCHPDDMSchurPreTypes[type], ((PetscObject)pc_00)->prefix ? ((PetscObject)pc_00)->prefix : "",
                       ((PetscObject)pc_00)->type_name, PCHPDDM);
            data_00 = (PC_HPDDM *)pc_00->data;
            PetscCheck(data_00->N == 2, PetscObjectComm((PetscObject)P), PETSC_ERR_ARG_INCOMP, "-%spc_hpddm_schur_precondition %s and %" PetscInt_FMT " level%s instead of 2 for the A00 block -%s", pcpre ? pcpre : "", PCHPDDMSchurPreTypes[type],
                       data_00->N, data_00->N > 1 ? "s" : "", ((PetscObject)pc_00)->prefix);
            PetscCheck(data_00->levels[0]->pc, PetscObjectComm((PetscObject)P), PETSC_ERR_ORDER, "PC of the first block%s not setup yet", ((PetscObject)pc_00)->prefix ? std::string(std::string(" (") + ((PetscObject)pc_00)->prefix + std::string(")")).c_str() : "");
            PetscCall(PetscObjectTypeCompare((PetscObject)data_00->levels[0]->pc, PCASM, &flg));
            PetscCheck(flg, PetscObjectComm((PetscObject)P), PETSC_ERR_ARG_INCOMP, "-%spc_hpddm_schur_precondition %s and -%spc_type %s (!= %s)", pcpre ? pcpre : "", PCHPDDMSchurPreTypes[type], ((PetscObject)data_00->levels[0]->pc)->prefix,
                       ((PetscObject)data_00->levels[0]->pc)->type_name, PCASM);
            PetscCall(PetscNew(&ctx)); /* context to pass data around for the inner-most PC, which will be a proper PCHPDDM (or a dummy variable if the Schur complement is centralized on a single process)  */
            PetscCall(MatSchurComplementGetSubMatrices(P, nullptr, nullptr, nullptr, nullptr, &A11));
            PetscCall(MatGetOwnershipRanges(A11, &ranges));
            PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)A11), &size));
            flg = PetscBool(std::find_if(ranges, ranges + size + 1, [&](PetscInt v) { return v != ranges[0] && v != ranges[size]; }) == ranges + size + 1); /* are all local matrices but one of dimension 0 (centralized Schur complement)? */
            if (!flg) {
              if (PetscDefined(USE_DEBUG) || !data->is) {
                Mat A01, A10, B = nullptr, C = nullptr, *sub;

                PetscCall(MatSchurComplementGetSubMatrices(P, &A, nullptr, &A01, &A10, nullptr));
                PetscCall(PetscObjectTypeCompare((PetscObject)A10, MATTRANSPOSEVIRTUAL, &flg));
                if (flg) {
                  PetscCall(MatTransposeGetMat(A10, &C));
                  PetscCall(MatTranspose(C, MAT_INITIAL_MATRIX, &B));
                } else {
                  PetscCall(PetscObjectTypeCompare((PetscObject)A10, MATHERMITIANTRANSPOSEVIRTUAL, &flg));
                  if (flg) {
                    PetscCall(MatHermitianTransposeGetMat(A10, &C));
                    PetscCall(MatHermitianTranspose(C, MAT_INITIAL_MATRIX, &B));
                  }
                }
                if (flg)
                  PetscCall(MatShellGetScalingShifts(A10, (PetscScalar *)MAT_SHELL_NOT_ALLOWED, (PetscScalar *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Mat *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED));
                if (!B) {
                  B = A10;
                  PetscCall(PetscObjectReference((PetscObject)B));
                } else if (!data->is) {
                  PetscCall(PetscObjectTypeCompareAny((PetscObject)A01, &flg, MATTRANSPOSEVIRTUAL, MATHERMITIANTRANSPOSEVIRTUAL, ""));
                  if (!flg) C = A01;
                  else
                    PetscCall(MatShellGetScalingShifts(A01, (PetscScalar *)MAT_SHELL_NOT_ALLOWED, (PetscScalar *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Mat *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED));
                }
                PetscCall(ISCreateStride(PETSC_COMM_SELF, B->rmap->N, 0, 1, &uis));
                PetscCall(ISSetIdentity(uis));
                if (!data->is) {
                  if (C) PetscCall(PetscObjectReference((PetscObject)C));
                  else PetscCall(MatTranspose(B, MAT_INITIAL_MATRIX, &C));
                  PetscCall(ISDuplicate(data_00->is, is));
                  PetscCall(MatIncreaseOverlap(A, 1, is, 1));
                  PetscCall(MatSetOption(C, MAT_SUBMAT_SINGLEIS, PETSC_TRUE));
                  PetscCall(MatCreateSubMatrices(C, 1, is, &uis, MAT_INITIAL_MATRIX, &sub));
                  PetscCall(MatDestroy(&C));
                  PetscCall(MatTranspose(sub[0], MAT_INITIAL_MATRIX, &C));
                  PetscCall(MatDestroySubMatrices(1, &sub));
                  PetscCall(MatFindNonzeroRows(C, &data->is));
                  PetscCheck(data->is, PetscObjectComm((PetscObject)C), PETSC_ERR_SUP, "No empty row, which likely means that some rows of A_10 are dense");
                  PetscCall(MatDestroy(&C));
                  PetscCall(ISDestroy(is));
                  PetscCall(ISCreateStride(PetscObjectComm((PetscObject)data->is), A11->rmap->n, A11->rmap->rstart, 1, &loc));
                  if (PetscDefined(USE_DEBUG)) PetscCall(PCHPDDMCheckInclusion_Private(pc, data->is, loc, PETSC_FALSE));
                  PetscCall(ISExpand(data->is, loc, is));
                  PetscCall(ISDestroy(&loc));
                  PetscCall(ISDestroy(&data->is));
                  data->is = is[0];
                  is[0]    = nullptr;
                }
                if (PetscDefined(USE_DEBUG)) {
                  PetscCall(PCHPDDMCheckSymmetry_Private(pc, A01, A10));
                  PetscCall(MatCreateSubMatrices(B, 1, &uis, &data_00->is, MAT_INITIAL_MATRIX, &sub)); /* expensive check since all processes fetch all rows (but only some columns) of the constraint matrix */
                  PetscCall(ISDestroy(&uis));
                  PetscCall(ISDuplicate(data->is, &uis));
                  PetscCall(ISSort(uis));
                  PetscCall(ISComplement(uis, 0, B->rmap->N, is));
                  PetscCall(MatDuplicate(sub[0], MAT_COPY_VALUES, &C));
                  PetscCall(MatZeroRowsIS(C, is[0], 0.0, nullptr, nullptr));
                  PetscCall(ISDestroy(is));
                  PetscCall(MatMultEqual(sub[0], C, 20, &flg));
                  PetscCheck(flg, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "The image of A_10 (R_i^p)^T from the local primal (e.g., velocity) space to the full dual (e.g., pressure) space is not restricted to the local dual space: A_10 (R_i^p)^T != R_i^d (R_i^d)^T A_10 (R_i^p)^T"); /* cf. eq. (9) of https://hal.science/hal-02343808v6/document */
                  PetscCall(MatDestroy(&C));
                  PetscCall(MatDestroySubMatrices(1, &sub));
                }
                PetscCall(ISDestroy(&uis));
                PetscCall(MatDestroy(&B));
              }
              flg = PETSC_FALSE;
              if (!data->aux) {
                Mat D;

                PetscCall(MatCreateVecs(A11, &d, nullptr));
                PetscCall(MatGetDiagonal(A11, d));
                PetscCall(PetscObjectTypeCompareAny((PetscObject)A11, &flg, MATDIAGONAL, MATCONSTANTDIAGONAL, ""));
                if (!flg) {
                  PetscCall(MatCreateDiagonal(d, &D));
                  PetscCall(MatMultEqual(A11, D, 20, &flg));
                  PetscCall(MatDestroy(&D));
                }
                if (flg) PetscCall(PetscInfo(pc, "A11 block is likely diagonal so the PC will build an auxiliary Mat (which was not initially provided by the user)\n"));
              }
              if (data->Neumann != PETSC_BOOL3_TRUE && !flg && A11) {
                PetscReal norm;

                PetscCall(MatNorm(A11, NORM_INFINITY, &norm));
                PetscCheck(norm < PETSC_MACHINE_EPSILON * static_cast<PetscReal>(10.0), PetscObjectComm((PetscObject)P), PETSC_ERR_ARG_INCOMP, "-%spc_hpddm_schur_precondition geneo and -%spc_hpddm_has_neumann != true with a nonzero or non-diagonal A11 block", pcpre ? pcpre : "", pcpre ? pcpre : "");
                PetscCall(PetscInfo(pc, "A11 block is likely zero so the PC will build an auxiliary Mat (which was%s initially provided by the user)\n", data->aux ? "" : " not"));
                PetscCall(MatDestroy(&data->aux));
                flg = PETSC_TRUE;
              }
              if (!data->aux) { /* if A11 is near zero, e.g., Stokes equation, or diagonal, build an auxiliary (Neumann) Mat which is a (possibly slightly shifted) diagonal weighted by the inverse of the multiplicity */
                PetscSF            scatter;
                const PetscScalar *read;
                PetscScalar       *write, *diagonal = nullptr;

                PetscCall(MatDestroy(&data->aux));
                PetscCall(ISGetLocalSize(data->is, &n));
                PetscCall(VecCreateMPI(PetscObjectComm((PetscObject)P), n, PETSC_DECIDE, &xin));
                PetscCall(VecDuplicate(xin, &v));
                PetscCall(VecScatterCreate(xin, data->is, v, nullptr, &scatter));
                PetscCall(VecSet(v, 1.0));
                PetscCall(VecSet(xin, 1.0));
                PetscCall(VecScatterBegin(scatter, v, xin, ADD_VALUES, SCATTER_REVERSE));
                PetscCall(VecScatterEnd(scatter, v, xin, ADD_VALUES, SCATTER_REVERSE)); /* v has the multiplicity of all unknowns on the overlap */
                PetscCall(PetscSFDestroy(&scatter));
                if (d) {
                  PetscCall(VecScatterCreate(d, data->is, v, nullptr, &scatter));
                  PetscCall(VecScatterBegin(scatter, d, v, INSERT_VALUES, SCATTER_FORWARD));
                  PetscCall(VecScatterEnd(scatter, d, v, INSERT_VALUES, SCATTER_FORWARD));
                  PetscCall(PetscSFDestroy(&scatter));
                  PetscCall(VecDestroy(&d));
                  PetscCall(PetscMalloc1(n, &diagonal));
                  PetscCall(VecGetArrayRead(v, &read));
                  PetscCallCXX(std::copy_n(read, n, diagonal));
                  PetscCall(VecRestoreArrayRead(v, &read));
                }
                PetscCall(VecDestroy(&v));
                PetscCall(VecCreateSeq(PETSC_COMM_SELF, n, &v));
                PetscCall(VecGetArrayRead(xin, &read));
                PetscCall(VecGetArrayWrite(v, &write));
                for (PetscInt i = 0; i < n; ++i) write[i] = (!diagonal || std::abs(diagonal[i]) < PETSC_MACHINE_EPSILON) ? PETSC_SMALL / (static_cast<PetscReal>(1000.0) * read[i]) : diagonal[i] / read[i];
                PetscCall(PetscFree(diagonal));
                PetscCall(VecRestoreArrayRead(xin, &read));
                PetscCall(VecRestoreArrayWrite(v, &write));
                PetscCall(VecDestroy(&xin));
                PetscCall(MatCreateDiagonal(v, &data->aux));
                PetscCall(VecDestroy(&v));
              }
              uis  = data->is;
              uaux = data->aux;
              PetscCall(PetscObjectReference((PetscObject)uis));
              PetscCall(PetscObjectReference((PetscObject)uaux));
              PetscCall(PetscStrallocpy(pcpre, &prefix));
              PetscCall(PCSetOptionsPrefix(pc, nullptr));
              PetscCall(PCSetType(pc, PCKSP));                                    /* replace the PC associated to the Schur complement by PCKSP */
              PetscCall(KSPCreate(PetscObjectComm((PetscObject)pc), &inner_ksp)); /* new KSP that will be attached to the previously set PC */
              PetscCall(PetscObjectGetTabLevel((PetscObject)pc, &n));
              PetscCall(PetscObjectSetTabLevel((PetscObject)inner_ksp, n + 2));
              PetscCall(KSPSetOperators(inner_ksp, pc->mat, pc->pmat));
              PetscCall(KSPSetOptionsPrefix(inner_ksp, std::string(std::string(prefix) + "pc_hpddm_").c_str()));
              PetscCall(KSPSetSkipPCSetFromOptions(inner_ksp, PETSC_TRUE));
              PetscCall(KSPSetFromOptions(inner_ksp));
              PetscCall(KSPGetPC(inner_ksp, &inner));
              PetscCall(PCSetOptionsPrefix(inner, nullptr));
              PetscCall(PCSetType(inner, PCNONE)); /* no preconditioner since the action of M^-1 A or A M^-1 will be computed by the Amat */
              PetscCall(PCKSPSetKSP(pc, inner_ksp));
              std::get<0>(*ctx)[0] = pc_00; /* for coarse correction on the primal (e.g., velocity) space */
              PetscCall(PCCreate(PetscObjectComm((PetscObject)pc), &std::get<0>(*ctx)[1]));
              PetscCall(PCSetOptionsPrefix(pc, prefix)); /* both PC share the same prefix so that the outer PC can be reset with PCSetFromOptions() */
              PetscCall(PCSetOptionsPrefix(std::get<0>(*ctx)[1], prefix));
              PetscCall(PetscFree(prefix));
              PetscCall(PCSetOperators(std::get<0>(*ctx)[1], pc->mat, pc->pmat));
              PetscCall(PCSetType(std::get<0>(*ctx)[1], PCHPDDM));
              PetscCall(PCHPDDMSetAuxiliaryMat(std::get<0>(*ctx)[1], uis, uaux, nullptr, nullptr)); /* transfer ownership of the auxiliary inputs from the inner (PCKSP) to the inner-most (PCHPDDM) PC */
              if (flg) static_cast<PC_HPDDM *>(std::get<0>(*ctx)[1]->data)->Neumann = PETSC_BOOL3_TRUE;
              PetscCall(PCSetFromOptions(std::get<0>(*ctx)[1]));
              PetscCall(PetscObjectDereference((PetscObject)uis));
              PetscCall(PetscObjectDereference((PetscObject)uaux));
              PetscCall(MatCreateShell(PetscObjectComm((PetscObject)pc), inner->mat->rmap->n, inner->mat->cmap->n, inner->mat->rmap->N, inner->mat->cmap->N, ctx, &S)); /* MatShell computing the action of M^-1 A or A M^-1 */
              PetscCall(MatShellSetOperation(S, MATOP_MULT, (PetscErrorCodeFn *)MatMult_SchurCorrection));
              PetscCall(MatShellSetOperation(S, MATOP_VIEW, (PetscErrorCodeFn *)MatView_SchurCorrection));
              PetscCall(MatShellSetOperation(S, MATOP_DESTROY, (PetscErrorCodeFn *)MatDestroy_SchurCorrection));
              PetscCall(KSPGetPCSide(inner_ksp, &(std::get<2>(*ctx))));
              if (std::get<2>(*ctx) == PC_LEFT || std::get<2>(*ctx) == PC_SIDE_DEFAULT) {
                PetscCall(KSPSetPreSolve(inner_ksp, KSPPreSolve_SchurCorrection, ctx));
              } else { /* no support for PC_SYMMETRIC */
                PetscCheck(std::get<2>(*ctx) == PC_RIGHT, PetscObjectComm((PetscObject)pc), PETSC_ERR_SUP, "PCSide %s (!= %s or %s or %s)", PCSides[std::get<2>(*ctx)], PCSides[PC_SIDE_DEFAULT], PCSides[PC_LEFT], PCSides[PC_RIGHT]);
              }
              PetscCall(KSPSetPostSolve(inner_ksp, KSPPostSolve_SchurCorrection, ctx));
              PetscCall(PetscObjectContainerCompose((PetscObject)std::get<0>(*ctx)[1], "_PCHPDDM_Schur", ctx, nullptr));
              PetscCall(PCSetUp(std::get<0>(*ctx)[1]));
              PetscCall(KSPSetOperators(inner_ksp, S, S));
              PetscCall(MatCreateVecs(std::get<1>(*ctx)[0], std::get<3>(*ctx), std::get<3>(*ctx) + 1));
              PetscCall(VecDuplicate(std::get<3>(*ctx)[0], std::get<3>(*ctx) + 2));
              PetscCall(PetscObjectDereference((PetscObject)inner_ksp));
              PetscCall(PetscObjectDereference((PetscObject)S));
            } else {
              std::get<0>(*ctx)[0] = pc_00;
              PetscCall(PetscObjectContainerCompose((PetscObject)pc, "_PCHPDDM_Schur", ctx, nullptr));
              PetscCall(ISCreateStride(PetscObjectComm((PetscObject)data_00->is), A11->rmap->n, A11->rmap->rstart, 1, &data->is)); /* dummy variables in the case of a centralized Schur complement */
              PetscCall(MatGetDiagonalBlock(A11, &data->aux));
              PetscCall(PetscObjectReference((PetscObject)data->aux));
              PetscCall(PCSetUp(pc));
            }
            for (std::vector<Vec>::iterator it = initial.begin(); it != initial.end(); ++it) PetscCall(VecDestroy(&*it));
            PetscFunctionReturn(PETSC_SUCCESS);
          } else { /* second call to PCSetUp() on the PC associated to the Schur complement, retrieve previously set context */
            PetscCall(PetscContainerGetPointer(container, (void **)&ctx));
          }
        }
      }
    }
    if (!data->is && data->N > 1) {
      char type[256] = {}; /* same size as in src/ksp/pc/interface/pcset.c */

      PetscCall(PetscObjectTypeCompareAny((PetscObject)P, &flg, MATNORMAL, MATNORMALHERMITIAN, ""));
      if (flg || (A->rmap->N != A->cmap->N && P->rmap->N == P->cmap->N && P->rmap->N == A->cmap->N)) {
        Mat B;

        PetscCall(PCHPDDMSetAuxiliaryMatNormal_Private(pc, A, P, &B, pcpre));
        if (data->correction == PC_HPDDM_COARSE_CORRECTION_DEFLATED) data->correction = PC_HPDDM_COARSE_CORRECTION_BALANCED;
        PetscCall(MatDestroy(&B));
      } else {
        PetscCall(PetscObjectTypeCompare((PetscObject)P, MATSCHURCOMPLEMENT, &flg));
        if (flg) {
          Mat                 A00, P00, A01, A10, A11, B, N;
          PCHPDDMSchurPreType type = PC_HPDDM_SCHUR_PRE_LEAST_SQUARES;

          PetscCall(MatSchurComplementGetSubMatrices(P, &A00, &P00, &A01, &A10, &A11));
          PetscCall(PetscOptionsGetEnum(nullptr, pcpre, "-pc_hpddm_schur_precondition", PCHPDDMSchurPreTypes, (PetscEnum *)&type, &flg));
          if (type == PC_HPDDM_SCHUR_PRE_LEAST_SQUARES) {
            Mat                        B01;
            Vec                        diagonal = nullptr;
            const PetscScalar         *array;
            MatSchurComplementAinvType type;

            PetscCall(PCHPDDMCheckSymmetry_Private(pc, A01, A10, &B01));
            if (A11) {
              PetscCall(MatCreateVecs(A11, &diagonal, nullptr));
              PetscCall(MatGetDiagonal(A11, diagonal));
            }
            PetscCall(MatCreateVecs(P00, &v, nullptr));
            PetscCall(MatSchurComplementGetAinvType(P, &type));
            PetscCheck(type == MAT_SCHUR_COMPLEMENT_AINV_DIAG || type == MAT_SCHUR_COMPLEMENT_AINV_LUMP, PetscObjectComm((PetscObject)P), PETSC_ERR_SUP, "-%smat_schur_complement_ainv_type %s", ((PetscObject)P)->prefix ? ((PetscObject)P)->prefix : "", MatSchurComplementAinvTypes[type]);
            if (type == MAT_SCHUR_COMPLEMENT_AINV_LUMP) {
              PetscCall(MatGetRowSum(P00, v));
              if (A00 == P00) PetscCall(PetscObjectReference((PetscObject)A00));
              PetscCall(MatDestroy(&P00));
              PetscCall(VecGetArrayRead(v, &array));
              PetscCall(MatCreateAIJ(PetscObjectComm((PetscObject)A00), A00->rmap->n, A00->cmap->n, A00->rmap->N, A00->cmap->N, 1, nullptr, 0, nullptr, &P00));
              PetscCall(MatSetOption(P00, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE));
              for (n = A00->rmap->rstart; n < A00->rmap->rend; ++n) PetscCall(MatSetValue(P00, n, n, array[n - A00->rmap->rstart], INSERT_VALUES));
              PetscCall(MatAssemblyBegin(P00, MAT_FINAL_ASSEMBLY));
              PetscCall(MatAssemblyEnd(P00, MAT_FINAL_ASSEMBLY));
              PetscCall(VecRestoreArrayRead(v, &array));
              PetscCall(MatSchurComplementUpdateSubMatrices(P, A00, P00, A01, A10, A11)); /* replace P00 by diag(sum of each row of P00) */
              PetscCall(MatDestroy(&P00));
            } else PetscCall(MatGetDiagonal(P00, v));
            PetscCall(VecReciprocal(v)); /* inv(diag(P00))       */
            PetscCall(VecSqrtAbs(v));    /* sqrt(inv(diag(P00))) */
            PetscCall(MatDuplicate(A01, MAT_COPY_VALUES, &B));
            PetscCall(MatDiagonalScale(B, v, nullptr));
            if (B01) PetscCall(MatDiagonalScale(B01, v, nullptr));
            PetscCall(VecDestroy(&v));
            PetscCall(MatCreateNormalHermitian(B, &N));
            PetscCall(PCHPDDMSetAuxiliaryMatNormal_Private(pc, B, N, &P, pcpre, &diagonal, B01));
            PetscCall(PetscObjectTypeCompare((PetscObject)data->aux, MATSEQAIJ, &flg));
            if (!flg) {
              PetscCall(MatDestroy(&P));
              P = N;
              PetscCall(PetscObjectReference((PetscObject)P));
            }
            if (diagonal) {
              PetscCall(MatDiagonalSet(P, diagonal, ADD_VALUES));
              PetscCall(PCSetOperators(pc, P, P)); /* replace P by A01^T inv(diag(P00)) A01 - diag(P11) */
              PetscCall(VecDestroy(&diagonal));
            } else PetscCall(PCSetOperators(pc, B01 ? P : N, P));  /* replace P by A01^T inv(diag(P00)) A01                         */
            pc->ops->postsolve = PCPostSolve_SchurPreLeastSquares; /*  PCFIELDSPLIT expect a KSP for (P11 - A10 inv(diag(P00)) A01) */
            PetscCall(MatDestroy(&N));                             /*  but a PC for (A10 inv(diag(P00)) A10 - P11) is setup instead */
            PetscCall(MatDestroy(&P));                             /*  so the sign of the solution must be flipped                  */
            PetscCall(MatDestroy(&B));
          } else
            PetscCheck(type != PC_HPDDM_SCHUR_PRE_GENEO, PetscObjectComm((PetscObject)P), PETSC_ERR_ARG_INCOMP, "-%spc_hpddm_schur_precondition %s without a prior call to PCHPDDMSetAuxiliaryMat() on the A11 block%s%s", pcpre ? pcpre : "", PCHPDDMSchurPreTypes[type], pcpre ? " -" : "", pcpre ? pcpre : "");
          for (std::vector<Vec>::iterator it = initial.begin(); it != initial.end(); ++it) PetscCall(VecDestroy(&*it));
          PetscFunctionReturn(PETSC_SUCCESS);
        } else {
          PetscCall(PetscOptionsGetString(nullptr, pcpre, "-pc_hpddm_levels_1_st_pc_type", type, sizeof(type), nullptr));
          PetscCall(PetscStrcmp(type, PCMAT, &algebraic));
          PetscCheck(!algebraic || !block, PetscObjectComm((PetscObject)P), PETSC_ERR_ARG_INCOMP, "-pc_hpddm_levels_1_st_pc_type mat and -pc_hpddm_block_splitting");
          if (overlap != -1) {
            PetscCheck(!block && !algebraic, PetscObjectComm((PetscObject)P), PETSC_ERR_ARG_INCOMP, "-pc_hpddm_%s and -pc_hpddm_harmonic_overlap", block ? "block_splitting" : "levels_1_st_pc_type mat");
            PetscCheck(overlap >= 1, PetscObjectComm((PetscObject)P), PETSC_ERR_ARG_WRONG, "-pc_hpddm_harmonic_overlap %" PetscInt_FMT " < 1", overlap);
          }
          if (block || overlap != -1) algebraic = PETSC_TRUE;
          if (algebraic) {
            PetscCall(ISCreateStride(PETSC_COMM_SELF, P->rmap->n, P->rmap->rstart, 1, &data->is));
            PetscCall(MatIncreaseOverlap(P, 1, &data->is, 1));
            PetscCall(ISSort(data->is));
          } else
            PetscCall(PetscInfo(pc, "Cannot assemble a fully-algebraic coarse operator with an assembled Pmat and -%spc_hpddm_levels_1_st_pc_type != mat and -%spc_hpddm_block_splitting != true and -%spc_hpddm_harmonic_overlap < 1\n", pcpre ? pcpre : "", pcpre ? pcpre : "", pcpre ? pcpre : ""));
        }
      }
    }
  }
#if PetscDefined(USE_DEBUG)
  if (data->is) PetscCall(ISDuplicate(data->is, &dis));
  if (data->aux) PetscCall(MatDuplicate(data->aux, MAT_COPY_VALUES, &daux));
#endif
  if (data->is || (ismatis && data->N > 1)) {
    if (ismatis) {
      std::initializer_list<std::string> list = {MATSEQBAIJ, MATSEQSBAIJ};
      PetscCall(MatISGetLocalMat(P, &N));
      std::initializer_list<std::string>::const_iterator it = std::find(list.begin(), list.end(), ((PetscObject)N)->type_name);
      PetscCall(MatISRestoreLocalMat(P, &N));
      switch (std::distance(list.begin(), it)) {
      case 0:
        PetscCall(MatConvert(P, MATMPIBAIJ, MAT_INITIAL_MATRIX, &C));
        break;
      case 1:
        /* MatCreateSubMatrices() does not work with MATSBAIJ and unsorted ISes, so convert to MPIBAIJ */
        PetscCall(MatConvert(P, MATMPIBAIJ, MAT_INITIAL_MATRIX, &C));
        PetscCall(MatSetOption(C, MAT_SYMMETRIC, PETSC_TRUE));
        break;
      default:
        PetscCall(MatConvert(P, MATMPIAIJ, MAT_INITIAL_MATRIX, &C));
      }
      PetscCall(MatISGetLocalToGlobalMapping(P, &l2g, nullptr));
      PetscCall(PetscObjectReference((PetscObject)P));
      PetscCall(KSPSetOperators(data->levels[0]->ksp, A, C));
      std::swap(C, P);
      PetscCall(ISLocalToGlobalMappingGetSize(l2g, &n));
      PetscCall(ISCreateStride(PETSC_COMM_SELF, n, 0, 1, &loc));
      PetscCall(ISLocalToGlobalMappingApplyIS(l2g, loc, &is[0]));
      PetscCall(ISDestroy(&loc));
      /* the auxiliary Mat is _not_ the local Neumann matrix                                */
      /* it is the local Neumann matrix augmented (with zeros) through MatIncreaseOverlap() */
      data->Neumann = PETSC_BOOL3_FALSE;
      structure     = SAME_NONZERO_PATTERN;
    } else {
      is[0] = data->is;
      if (algebraic || ctx) subdomains = PETSC_TRUE;
      PetscCall(PetscOptionsGetBool(nullptr, pcpre, "-pc_hpddm_define_subdomains", &subdomains, nullptr));
      if (ctx) PetscCheck(subdomains, PetscObjectComm((PetscObject)P), PETSC_ERR_ARG_INCOMP, "-%spc_hpddm_schur_precondition geneo and -%spc_hpddm_define_subdomains false", pcpre, pcpre);
      if (PetscBool3ToBool(data->Neumann)) {
        PetscCheck(!block, PetscObjectComm((PetscObject)P), PETSC_ERR_ARG_INCOMP, "-pc_hpddm_block_splitting and -pc_hpddm_has_neumann");
        PetscCheck(overlap == -1, PetscObjectComm((PetscObject)P), PETSC_ERR_ARG_INCOMP, "-pc_hpddm_harmonic_overlap %" PetscInt_FMT " and -pc_hpddm_has_neumann", overlap);
        PetscCheck(!algebraic, PetscObjectComm((PetscObject)P), PETSC_ERR_ARG_INCOMP, "-pc_hpddm_levels_1_st_pc_type mat and -pc_hpddm_has_neumann");
      }
      if (PetscBool3ToBool(data->Neumann) || block) structure = SAME_NONZERO_PATTERN;
      PetscCall(ISCreateStride(PetscObjectComm((PetscObject)data->is), P->rmap->n, P->rmap->rstart, 1, &loc));
    }
    PetscCall(PetscSNPrintf(prefix, sizeof(prefix), "%spc_hpddm_levels_1_", pcpre ? pcpre : ""));
    PetscCall(PetscOptionsGetEnum(nullptr, prefix, "-st_matstructure", MatStructures, (PetscEnum *)&structure, &flg)); /* if not user-provided, force its value when possible */
    if (!flg && structure == SAME_NONZERO_PATTERN) {                                                                   /* cannot call STSetMatStructure() yet, insert the appropriate option in the database, parsed by STSetFromOptions() */
      PetscCall(PetscSNPrintf(prefix, sizeof(prefix), "-%spc_hpddm_levels_1_st_matstructure", pcpre ? pcpre : ""));
      PetscCall(PetscOptionsSetValue(nullptr, prefix, MatStructures[structure]));
    }
    flg = PETSC_FALSE;
    if (data->share) {
      data->share = PETSC_FALSE; /* will be reset to PETSC_TRUE if none of the conditions below are true */
      if (!subdomains) PetscCall(PetscInfo(pc, "Cannot share subdomain KSP between SLEPc and PETSc since -%spc_hpddm_define_subdomains is not true\n", pcpre ? pcpre : ""));
      else if (data->deflation) PetscCall(PetscInfo(pc, "Nothing to share since PCHPDDMSetDeflationMat() has been called\n"));
      else if (ismatis) PetscCall(PetscInfo(pc, "Cannot share subdomain KSP between SLEPc and PETSc with a Pmat of type MATIS\n"));
      else if (!algebraic && structure != SAME_NONZERO_PATTERN)
        PetscCall(PetscInfo(pc, "Cannot share subdomain KSP between SLEPc and PETSc since -%spc_hpddm_levels_1_st_matstructure %s (!= %s)\n", pcpre ? pcpre : "", MatStructures[structure], MatStructures[SAME_NONZERO_PATTERN]));
      else data->share = PETSC_TRUE;
    }
    if (!ismatis) {
      if (data->share || (!PetscBool3ToBool(data->Neumann) && subdomains)) PetscCall(ISDuplicate(is[0], &unsorted));
      else unsorted = is[0];
    }
    if ((ctx || data->N > 1) && (data->aux || ismatis || algebraic)) {
      PetscCheck(loadedSym, PETSC_COMM_SELF, PETSC_ERR_PLIB, "HPDDM library not loaded, cannot use more than one level");
      PetscCall(MatSetOption(P, MAT_SUBMAT_SINGLEIS, PETSC_TRUE));
      if (ismatis) {
        /* needed by HPDDM (currently) so that the partition of unity is 0 on subdomain interfaces */
        PetscCall(MatIncreaseOverlap(P, 1, is, 1));
        PetscCall(ISDestroy(&data->is));
        data->is = is[0];
      } else {
        if (PetscDefined(USE_DEBUG)) PetscCall(PCHPDDMCheckInclusion_Private(pc, data->is, loc, PETSC_TRUE));
        if (!ctx && overlap == -1) PetscCall(PetscObjectComposeFunction((PetscObject)pc->pmat, "PCHPDDMAlgebraicAuxiliaryMat_Private_C", PCHPDDMAlgebraicAuxiliaryMat_Private));
        if (!PetscBool3ToBool(data->Neumann) && (!algebraic || overlap != -1)) {
          PetscCall(PetscObjectTypeCompare((PetscObject)P, MATMPISBAIJ, &flg));
          if (flg) {
            /* maybe better to ISSort(is[0]), MatCreateSubMatrices(), and then MatPermute() */
            /* but there is no MatPermute_SeqSBAIJ(), so as before, just use MATMPIBAIJ     */
            PetscCall(MatConvert(P, MATMPIBAIJ, MAT_INITIAL_MATRIX, &uaux));
            flg = PETSC_FALSE;
          }
        }
      }
      if (algebraic && overlap == -1) {
        PetscUseMethod(pc->pmat, "PCHPDDMAlgebraicAuxiliaryMat_Private_C", (Mat, IS *, Mat *[], PetscBool), (P, is, &sub, block));
        if (block) {
          PetscCall(PetscObjectQuery((PetscObject)sub[0], "_PCHPDDM_Neumann_Mat", (PetscObject *)&data->aux));
          PetscCall(PetscObjectCompose((PetscObject)sub[0], "_PCHPDDM_Neumann_Mat", nullptr));
        }
      } else if (!uaux || overlap != -1) {
        if (!ctx) {
          if (PetscBool3ToBool(data->Neumann)) sub = &data->aux;
          else {
            PetscBool flg;
            if (overlap != -1) {
              Harmonic              h;
              Mat                   A0, *a;                    /* with an SVD: [ A_00  A_01       ] */
              IS                    ov[2], rows, cols, stride; /*              [ A_10  A_11  A_12 ] */
              const PetscInt       *i[2], bs = P->cmap->bs;    /* with a GEVP: [ A_00  A_01       ] */
              PetscInt              n[2];                      /*              [ A_10  A_11  A_12 ] */
              std::vector<PetscInt> v[2];                      /*              [       A_21  A_22 ] */

              do {
                PetscCall(ISDuplicate(data->is, ov));
                if (overlap > 1) PetscCall(MatIncreaseOverlap(P, 1, ov, overlap - 1));
                PetscCall(ISDuplicate(ov[0], ov + 1));
                PetscCall(MatIncreaseOverlap(P, 1, ov + 1, 1));
                PetscCall(ISGetLocalSize(ov[0], n));
                PetscCall(ISGetLocalSize(ov[1], n + 1));
                flg = PetscBool(n[0] == n[1] && n[0] != P->rmap->n);
                PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &flg, 1, MPI_C_BOOL, MPI_LOR, PetscObjectComm((PetscObject)pc)));
                if (flg) {
                  PetscCall(ISDestroy(ov));
                  PetscCall(ISDestroy(ov + 1));
                  PetscCheck(--overlap, PetscObjectComm((PetscObject)pc), PETSC_ERR_SUP, "No oversampling possible");
                  PetscCall(PetscInfo(pc, "Supplied -%spc_hpddm_harmonic_overlap parameter is too large, it has been decreased to %" PetscInt_FMT "\n", pcpre ? pcpre : "", overlap));
                } else break;
              } while (1);
              PetscCall(PetscNew(&h));
              h->ksp = nullptr;
              PetscCall(PetscCalloc1(2, &h->A));
              PetscCall(PetscOptionsHasName(nullptr, prefix, "-eps_nev", &flg));
              if (!flg) {
                PetscCall(PetscOptionsHasName(nullptr, prefix, "-svd_nsv", &flg));
                if (!flg) PetscCall(PetscOptionsHasName(nullptr, prefix, "-svd_threshold_relative", &flg));
              } else flg = PETSC_FALSE;
              PetscCall(ISSort(ov[0]));
              if (!flg) PetscCall(ISSort(ov[1]));
              PetscCall(PetscCalloc1(5, &h->is));
              PetscCall(MatCreateSubMatrices(uaux ? uaux : P, 1, ov + !flg, ov + 1, MAT_INITIAL_MATRIX, &a)); /* submatrix from above, either square (!flg) or rectangular (flg) */
              for (PetscInt j = 0; j < 2; ++j) PetscCall(ISGetIndices(ov[j], i + j));
              v[1].reserve((n[1] - n[0]) / bs);
              for (PetscInt j = 0; j < n[1]; j += bs) { /* indices of the (2,2) block */
                PetscInt location;
                PetscCall(ISLocate(ov[0], i[1][j], &location));
                if (location < 0) v[1].emplace_back(j / bs);
              }
              if (!flg) {
                h->A[1] = a[0];
                PetscCall(PetscObjectReference((PetscObject)h->A[1]));
                v[0].reserve((n[0] - P->rmap->n) / bs);
                for (PetscInt j = 0; j < n[1]; j += bs) { /* row indices of the (1,2) block */
                  PetscInt location;
                  PetscCall(ISLocate(loc, i[1][j], &location));
                  if (location < 0) {
                    PetscCall(ISLocate(ov[0], i[1][j], &location));
                    if (location >= 0) v[0].emplace_back(j / bs);
                  }
                }
                PetscCall(ISCreateBlock(PETSC_COMM_SELF, bs, v[0].size(), v[0].data(), PETSC_USE_POINTER, &rows));
                PetscCall(ISCreateBlock(PETSC_COMM_SELF, bs, v[1].size(), v[1].data(), PETSC_COPY_VALUES, h->is + 4));
                PetscCall(MatCreateSubMatrix(a[0], rows, h->is[4], MAT_INITIAL_MATRIX, h->A)); /* A_12 submatrix from above */
                PetscCall(ISDestroy(&rows));
                if (uaux) PetscCall(MatConvert(a[0], MATSEQSBAIJ, MAT_INPLACE_MATRIX, a)); /* initial Pmat was MATSBAIJ, convert back to the same format since the rectangular A_12 submatrix has been created */
                PetscCall(ISEmbed(ov[0], ov[1], PETSC_TRUE, &rows));
                PetscCall(MatCreateSubMatrix(a[0], rows, cols = rows, MAT_INITIAL_MATRIX, &A0)); /* [ A_00  A_01 ; A_10  A_11 ] submatrix from above */
                PetscCall(ISDestroy(&rows));
                v[0].clear();
                PetscCall(ISEmbed(loc, ov[1], PETSC_TRUE, h->is + 3));
                PetscCall(ISEmbed(data->is, ov[1], PETSC_TRUE, h->is + 2));
              }
              v[0].reserve((n[0] - P->rmap->n) / bs);
              for (PetscInt j = 0; j < n[0]; j += bs) {
                PetscInt location;
                PetscCall(ISLocate(loc, i[0][j], &location));
                if (location < 0) v[0].emplace_back(j / bs);
              }
              PetscCall(ISCreateBlock(PETSC_COMM_SELF, bs, v[0].size(), v[0].data(), PETSC_USE_POINTER, &rows));
              for (PetscInt j = 0; j < 2; ++j) PetscCall(ISRestoreIndices(ov[j], i + j));
              if (flg) {
                IS is;
                PetscCall(ISCreateStride(PETSC_COMM_SELF, a[0]->rmap->n, 0, 1, &is));
                PetscCall(ISEmbed(ov[0], ov[1], PETSC_TRUE, &cols));
                PetscCall(MatCreateSubMatrix(a[0], is, cols, MAT_INITIAL_MATRIX, &A0)); /* [ A_00  A_01 ; A_10  A_11 ] submatrix from above */
                PetscCall(ISDestroy(&cols));
                PetscCall(ISDestroy(&is));
                if (uaux) PetscCall(MatConvert(A0, MATSEQSBAIJ, MAT_INPLACE_MATRIX, &A0)); /* initial Pmat was MATSBAIJ, convert back to the same format since this submatrix is square */
                PetscCall(ISEmbed(loc, data->is, PETSC_TRUE, h->is + 2));
                PetscCall(ISCreateBlock(PETSC_COMM_SELF, bs, v[1].size(), v[1].data(), PETSC_USE_POINTER, &cols));
                PetscCall(MatCreateSubMatrix(a[0], rows, cols, MAT_INITIAL_MATRIX, h->A)); /* A_12 submatrix from above */
                PetscCall(ISDestroy(&cols));
              }
              PetscCall(ISCreateStride(PETSC_COMM_SELF, A0->rmap->n, 0, 1, &stride));
              PetscCall(ISEmbed(rows, stride, PETSC_TRUE, h->is));
              PetscCall(ISDestroy(&stride));
              PetscCall(ISDestroy(&rows));
              PetscCall(ISEmbed(loc, ov[0], PETSC_TRUE, h->is + 1));
              if (subdomains) {
                if (!data->levels[0]->pc) {
                  PetscCall(PCCreate(PetscObjectComm((PetscObject)pc), &data->levels[0]->pc));
                  PetscCall(PetscSNPrintf(prefix, sizeof(prefix), "%spc_hpddm_levels_1_", pcpre ? pcpre : ""));
                  PetscCall(PCSetOptionsPrefix(data->levels[0]->pc, prefix));
                  PetscCall(PCSetOperators(data->levels[0]->pc, A, P));
                }
                PetscCall(PCSetType(data->levels[0]->pc, PCASM));
                if (!data->levels[0]->pc->setupcalled) PetscCall(PCASMSetLocalSubdomains(data->levels[0]->pc, 1, ov + !flg, &loc));
                PetscCall(PCHPDDMCommunicationAvoidingPCASM_Private(data->levels[0]->pc, flg ? A0 : a[0], PETSC_TRUE));
                if (!flg) ++overlap;
                if (data->share) {
                  PetscInt n = -1;
                  PetscTryMethod(data->levels[0]->pc, "PCASMGetSubKSP_C", (PC, PetscInt *, PetscInt *, KSP **), (data->levels[0]->pc, &n, nullptr, &ksp));
                  PetscCheck(n == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of subdomain solver %" PetscInt_FMT " != 1", n);
                  if (flg) {
                    h->ksp = ksp[0];
                    PetscCall(PetscObjectReference((PetscObject)h->ksp));
                  }
                }
              }
              if (!h->ksp) {
                PetscBool share = data->share;
                PetscCall(KSPCreate(PETSC_COMM_SELF, &h->ksp));
                PetscCall(KSPSetType(h->ksp, KSPPREONLY));
                PetscCall(KSPSetOperators(h->ksp, A0, A0));
                do {
                  if (!data->share) {
                    share = PETSC_FALSE;
                    PetscCall(PetscSNPrintf(prefix, sizeof(prefix), "%spc_hpddm_levels_1_%s", pcpre ? pcpre : "", flg ? "svd_" : "eps_"));
                    PetscCall(KSPSetOptionsPrefix(h->ksp, prefix));
                    PetscCall(KSPSetFromOptions(h->ksp));
                  } else {
                    MatSolverType type;
                    PetscCall(KSPGetPC(ksp[0], &pc));
                    PetscCall(PetscObjectTypeCompareAny((PetscObject)pc, &data->share, PCLU, PCCHOLESKY, ""));
                    if (data->share) {
                      PetscCall(PCFactorGetMatSolverType(pc, &type));
                      if (!type) {
                        if (PetscDefined(HAVE_MUMPS)) PetscCall(PCFactorSetMatSolverType(pc, MATSOLVERMUMPS));
                        else if (PetscDefined(HAVE_MKL_PARDISO)) PetscCall(PCFactorSetMatSolverType(pc, MATSOLVERMKL_PARDISO));
                        else data->share = PETSC_FALSE;
                        if (data->share) PetscCall(PCSetFromOptions(pc));
                      } else {
                        PetscCall(PetscStrcmp(type, MATSOLVERMUMPS, &data->share));
                        if (!data->share) PetscCall(PetscStrcmp(type, MATSOLVERMKL_PARDISO, &data->share));
                      }
                      if (data->share) {
                        std::tuple<KSP, IS, Vec[2]> *p;
                        PetscCall(PCFactorGetMatrix(pc, &A));
                        PetscCall(MatFactorSetSchurIS(A, h->is[4]));
                        PetscCall(KSPSetUp(ksp[0]));
                        PetscCall(PetscSNPrintf(prefix, sizeof(prefix), "%spc_hpddm_levels_1_eps_shell_", pcpre ? pcpre : ""));
                        PetscCall(KSPSetOptionsPrefix(h->ksp, prefix));
                        PetscCall(KSPSetFromOptions(h->ksp));
                        PetscCall(KSPGetPC(h->ksp, &pc));
                        PetscCall(PCSetType(pc, PCSHELL));
                        PetscCall(PetscNew(&p));
                        std::get<0>(*p) = ksp[0];
                        PetscCall(ISEmbed(ov[0], ov[1], PETSC_TRUE, &std::get<1>(*p)));
                        PetscCall(MatCreateVecs(A, std::get<2>(*p), std::get<2>(*p) + 1));
                        PetscCall(PCShellSetContext(pc, p));
                        PetscCall(PCShellSetApply(pc, PCApply_Schur));
                        PetscCall(PCShellSetApplyTranspose(pc, PCApply_Schur<Vec, true>));
                        PetscCall(PCShellSetMatApply(pc, PCApply_Schur<Mat>));
                        PetscCall(PCShellSetDestroy(pc, PCDestroy_Schur));
                      }
                    }
                    if (!data->share) PetscCall(PetscInfo(pc, "Cannot share subdomain KSP between SLEPc and PETSc since neither MUMPS nor MKL PARDISO is used\n"));
                  }
                } while (!share != !data->share); /* if data->share is initially PETSC_TRUE, but then reset to PETSC_FALSE, then go back to the beginning of the do loop */
              }
              PetscCall(ISDestroy(ov));
              PetscCall(ISDestroy(ov + 1));
              if (overlap == 1 && subdomains && flg) {
                *subA = A0;
                sub   = subA;
                if (uaux) PetscCall(MatDestroy(&uaux));
              } else PetscCall(MatDestroy(&A0));
              PetscCall(MatCreateShell(PETSC_COMM_SELF, P->rmap->n, n[1] - n[0], P->rmap->n, n[1] - n[0], h, &data->aux));
              PetscCall(KSPSetErrorIfNotConverged(h->ksp, PETSC_TRUE)); /* bail out as early as possible to avoid (apparently) unrelated error messages */
              PetscCall(MatCreateVecs(h->ksp->pc->pmat, &h->v, nullptr));
              PetscCall(MatShellSetOperation(data->aux, MATOP_MULT, (PetscErrorCodeFn *)MatMult_Harmonic));
              PetscCall(MatShellSetOperation(data->aux, MATOP_MULT_TRANSPOSE, (PetscErrorCodeFn *)MatMultTranspose_Harmonic));
              PetscCall(MatShellSetMatProductOperation(data->aux, MATPRODUCT_AB, nullptr, MatProduct_AB_Harmonic, nullptr, MATDENSE, MATDENSE));
              PetscCall(MatShellSetOperation(data->aux, MATOP_DESTROY, (PetscErrorCodeFn *)MatDestroy_Harmonic));
              PetscCall(MatDestroySubMatrices(1, &a));
            }
            if (overlap != 1 || !subdomains) {
              PetscCall(MatCreateSubMatrices(uaux ? uaux : P, 1, is, is, MAT_INITIAL_MATRIX, &sub));
              if (ismatis) {
                PetscCall(MatISGetLocalMat(C, &N));
                PetscCall(PetscObjectTypeCompare((PetscObject)N, MATSEQSBAIJ, &flg));
                if (flg) PetscCall(MatConvert(sub[0], MATSEQSBAIJ, MAT_INPLACE_MATRIX, sub));
                PetscCall(MatISRestoreLocalMat(C, &N));
              }
            }
            if (uaux) {
              PetscCall(MatDestroy(&uaux));
              PetscCall(MatConvert(sub[0], MATSEQSBAIJ, MAT_INPLACE_MATRIX, sub));
            }
          }
        }
      } else if (!ctx) {
        PetscCall(MatCreateSubMatrices(uaux, 1, is, is, MAT_INITIAL_MATRIX, &sub));
        PetscCall(MatDestroy(&uaux));
        PetscCall(MatConvert(sub[0], MATSEQSBAIJ, MAT_INPLACE_MATRIX, sub));
      }
      if (data->N > 1) {
        /* Vec holding the partition of unity */
        if (!data->levels[0]->D) {
          PetscCall(ISGetLocalSize(data->is, &n));
          PetscCall(VecCreateMPI(PETSC_COMM_SELF, n, PETSC_DETERMINE, &data->levels[0]->D));
        }
        if (data->share && overlap == -1) {
          Mat      D;
          IS       perm = nullptr;
          PetscInt size = -1;

          if (!data->levels[0]->pc) {
            PetscCall(PetscSNPrintf(prefix, sizeof(prefix), "%spc_hpddm_levels_1_", pcpre ? pcpre : ""));
            PetscCall(PCCreate(PetscObjectComm((PetscObject)pc), &data->levels[0]->pc));
            PetscCall(PCSetOptionsPrefix(data->levels[0]->pc, prefix));
            PetscCall(PCSetOperators(data->levels[0]->pc, A, P));
          }
          PetscCall(PCSetType(data->levels[0]->pc, PCASM));
          if (!ctx) {
            if (!data->levels[0]->pc->setupcalled) {
              IS sorted; /* PCASM will sort the input IS, duplicate it to return an unmodified (PCHPDDM) input IS */
              PetscCall(ISDuplicate(is[0], &sorted));
              PetscCall(PCASMSetLocalSubdomains(data->levels[0]->pc, 1, &sorted, &loc));
              PetscCall(PetscObjectDereference((PetscObject)sorted));
            }
            PetscCall(PCSetFromOptions(data->levels[0]->pc));
            if (block) {
              PetscCall(PCHPDDMPermute_Private(unsorted, data->is, &uis, sub[0], &C, &perm));
              PetscCall(PCHPDDMCommunicationAvoidingPCASM_Private(data->levels[0]->pc, C, algebraic));
            } else PetscCall(PCSetUp(data->levels[0]->pc));
            PetscTryMethod(data->levels[0]->pc, "PCASMGetSubKSP_C", (PC, PetscInt *, PetscInt *, KSP **), (data->levels[0]->pc, &size, nullptr, &ksp));
            if (size != 1) {
              data->share = PETSC_FALSE;
              PetscCheck(size == -1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of subdomain solver %" PetscInt_FMT " != 1", size);
              PetscCall(PetscInfo(pc, "Cannot share subdomain KSP between SLEPc and PETSc since PCASMGetSubKSP() not found in fine-level PC\n"));
              PetscCall(ISDestroy(&unsorted));
              unsorted = is[0];
            } else {
              const char *matpre;
              PetscBool   cmp[4];

              if (!block && !ctx) PetscCall(PCHPDDMPermute_Private(unsorted, data->is, &uis, PetscBool3ToBool(data->Neumann) ? sub[0] : data->aux, &C, &perm));
              if (perm) { /* unsorted input IS */
                if (!PetscBool3ToBool(data->Neumann) && !block) {
                  PetscCall(MatPermute(sub[0], perm, perm, &D)); /* permute since PCASM will call ISSort() */
                  PetscCall(MatHeaderReplace(sub[0], &D));
                }
                if (data->B) { /* see PCHPDDMSetRHSMat() */
                  PetscCall(MatPermute(data->B, perm, perm, &D));
                  PetscCall(MatHeaderReplace(data->B, &D));
                }
                PetscCall(ISDestroy(&perm));
              }
              PetscCall(KSPGetOperators(ksp[0], subA, subA + 1));
              PetscCall(PetscObjectReference((PetscObject)subA[0]));
              PetscCall(MatDuplicate(subA[1], MAT_SHARE_NONZERO_PATTERN, &D));
              PetscCall(MatGetOptionsPrefix(subA[1], &matpre));
              PetscCall(MatSetOptionsPrefix(D, matpre));
              PetscCall(PetscObjectTypeCompare((PetscObject)D, MATNORMAL, cmp));
              PetscCall(PetscObjectTypeCompare((PetscObject)C, MATNORMAL, cmp + 1));
              if (!cmp[0]) PetscCall(PetscObjectTypeCompare((PetscObject)D, MATNORMALHERMITIAN, cmp + 2));
              else cmp[2] = PETSC_FALSE;
              if (!cmp[1]) PetscCall(PetscObjectTypeCompare((PetscObject)C, MATNORMALHERMITIAN, cmp + 3));
              else cmp[3] = PETSC_FALSE;
              PetscCheck(cmp[0] == cmp[1] && cmp[2] == cmp[3], PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "-%spc_hpddm_levels_1_pc_asm_sub_mat_type %s and auxiliary Mat of type %s", pcpre ? pcpre : "", ((PetscObject)D)->type_name, ((PetscObject)C)->type_name);
              if (!cmp[0] && !cmp[2]) {
                if (!block) PetscCall(MatAXPY(D, 1.0, C, SUBSET_NONZERO_PATTERN));
                else {
                  PetscCall(MatMissingDiagonal(D, cmp, nullptr));
                  if (cmp[0]) structure = DIFFERENT_NONZERO_PATTERN; /* data->aux has no missing diagonal entry */
                  PetscCall(MatAXPY(D, 1.0, data->aux, structure));
                }
              } else {
                Mat mat[2];

                if (cmp[0]) {
                  PetscCall(MatNormalGetMat(D, mat));
                  PetscCall(MatNormalGetMat(C, mat + 1));
                } else {
                  PetscCall(MatNormalHermitianGetMat(D, mat));
                  PetscCall(MatNormalHermitianGetMat(C, mat + 1));
                }
                PetscCall(MatAXPY(mat[0], 1.0, mat[1], SUBSET_NONZERO_PATTERN));
              }
              PetscCall(MatPropagateSymmetryOptions(C, D));
              PetscCall(MatDestroy(&C));
              C = D;
              /* swap pointers so that variables stay consistent throughout PCSetUp() */
              std::swap(C, data->aux);
              std::swap(uis, data->is);
              swap = PETSC_TRUE;
            }
          }
        }
      }
      if (ctx) {
        PC_HPDDM              *data_00 = (PC_HPDDM *)std::get<0>(*ctx)[0]->data;
        PC                     s;
        Mat                    A00, P00, A01 = nullptr, A10, A11, N, b[4];
        IS                     sorted, is[2], *is_00;
        MatSolverType          type;
        std::pair<PC, Vec[2]> *p;

        n = -1;
        PetscTryMethod(data_00->levels[0]->pc, "PCASMGetSubKSP_C", (PC, PetscInt *, PetscInt *, KSP **), (data_00->levels[0]->pc, &n, nullptr, &ksp));
        PetscCheck(n == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of subdomain solver %" PetscInt_FMT " != 1", n);
        PetscCall(KSPGetOperators(ksp[0], subA, subA + 1));
        PetscCall(ISGetLocalSize(data_00->is, &n));
        if (n != subA[0]->rmap->n || n != subA[0]->cmap->n) {
          PetscCall(PCASMGetLocalSubdomains(data_00->levels[0]->pc, &n, &is_00, nullptr));
          PetscCall(ISGetLocalSize(*is_00, &n));
          PetscCheck(n == subA[0]->rmap->n && n == subA[0]->cmap->n, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "-%spc_hpddm_schur_precondition geneo and -%spc_hpddm_define_subdomains false", pcpre ? pcpre : "", ((PetscObject)pc)->prefix);
        } else is_00 = &data_00->is;
        PetscCall(PCHPDDMPermute_Private(unsorted, data->is, &uis, data->aux, &C, nullptr)); /* permute since PCASM works with a sorted IS */
        std::swap(C, data->aux);
        std::swap(uis, data->is);
        swap = PETSC_TRUE;
        PetscCall(MatSchurComplementGetSubMatrices(P, &A00, &P00, std::get<1>(*ctx), &A10, &A11));
        std::get<1>(*ctx)[1] = A10;
        PetscCall(PetscObjectTypeCompare((PetscObject)A10, MATTRANSPOSEVIRTUAL, &flg));
        if (flg) PetscCall(MatTransposeGetMat(A10, &A01));
        else {
          PetscBool flg;

          PetscCall(PetscObjectTypeCompare((PetscObject)A10, MATHERMITIANTRANSPOSEVIRTUAL, &flg));
          if (flg) PetscCall(MatHermitianTransposeGetMat(A10, &A01));
        }
        PetscCall(ISDuplicate(*is_00, &sorted)); /* during setup of the PC associated to the A00 block, this IS has already been sorted, but it's put back to its original state at the end of PCSetUp_HPDDM(), which may be unsorted */
        PetscCall(ISSort(sorted));               /* this is to avoid changing users inputs, but it requires a new call to ISSort() here                                                                                               */
        if (!A01) {
          PetscCall(MatSetOption(A10, MAT_SUBMAT_SINGLEIS, PETSC_TRUE));
          PetscCall(MatCreateSubMatrices(A10, 1, &data->is, &sorted, MAT_INITIAL_MATRIX, &sub));
          b[2] = sub[0];
          PetscCall(PetscObjectReference((PetscObject)sub[0]));
          PetscCall(MatDestroySubMatrices(1, &sub));
          PetscCall(PetscObjectTypeCompare((PetscObject)std::get<1>(*ctx)[0], MATTRANSPOSEVIRTUAL, &flg));
          A10 = nullptr;
          if (flg) PetscCall(MatTransposeGetMat(std::get<1>(*ctx)[0], &A10));
          else {
            PetscBool flg;

            PetscCall(PetscObjectTypeCompare((PetscObject)std::get<1>(*ctx)[0], MATHERMITIANTRANSPOSEVIRTUAL, &flg));
            if (flg) PetscCall(MatHermitianTransposeGetMat(std::get<1>(*ctx)[0], &A10));
          }
          if (!A10) PetscCall(MatCreateSubMatrices(std::get<1>(*ctx)[0], 1, &sorted, &data->is, MAT_INITIAL_MATRIX, &sub));
          else {
            if (flg) PetscCall(MatCreateTranspose(b[2], b + 1));
            else PetscCall(MatCreateHermitianTranspose(b[2], b + 1));
          }
        } else {
          PetscCall(MatSetOption(A01, MAT_SUBMAT_SINGLEIS, PETSC_TRUE));
          PetscCall(MatCreateSubMatrices(A01, 1, &sorted, &data->is, MAT_INITIAL_MATRIX, &sub));
          if (flg) PetscCall(MatCreateTranspose(*sub, b + 2));
          else PetscCall(MatCreateHermitianTranspose(*sub, b + 2));
        }
        if (A01 || !A10) {
          b[1] = sub[0];
          PetscCall(PetscObjectReference((PetscObject)sub[0]));
        }
        PetscCall(MatDestroySubMatrices(1, &sub));
        PetscCall(ISDestroy(&sorted));
        b[3] = data->aux;
        PetscCall(MatCreateSchurComplement(subA[0], subA[1], b[1], b[2], b[3], &S));
        PetscCall(MatSchurComplementSetKSP(S, ksp[0]));
        if (data->N != 1) {
          PetscCall(PCASMSetType(data->levels[0]->pc, PC_ASM_NONE)); /* "Neumann--Neumann" preconditioning with overlap and a Boolean partition of unity */
          PetscCall(PCASMSetLocalSubdomains(data->levels[0]->pc, 1, &data->is, &loc));
          PetscCall(PCSetFromOptions(data->levels[0]->pc)); /* action of eq. (15) of https://hal.science/hal-02343808v6/document (with a sign flip) */
          s = data->levels[0]->pc;
        } else {
          is[0] = data->is;
          PetscCall(PetscObjectReference((PetscObject)is[0]));
          PetscCall(PetscObjectReference((PetscObject)b[3]));
          PetscCall(PCSetType(pc, PCASM));                          /* change the type of the current PC */
          data = nullptr;                                           /* destroyed in the previous PCSetType(), so reset to NULL to avoid any faulty use */
          PetscCall(PCAppendOptionsPrefix(pc, "pc_hpddm_coarse_")); /* same prefix as when using PCHPDDM with a single level */
          PetscCall(PCASMSetLocalSubdomains(pc, 1, is, &loc));
          PetscCall(ISDestroy(is));
          PetscCall(ISDestroy(&loc));
          s = pc;
        }
        PetscCall(PCHPDDMCommunicationAvoidingPCASM_Private(s, S, PETSC_TRUE)); /* the subdomain Mat is already known and the input IS of PCASMSetLocalSubdomains() is already sorted */
        PetscTryMethod(s, "PCASMGetSubKSP_C", (PC, PetscInt *, PetscInt *, KSP **), (s, &n, nullptr, &ksp));
        PetscCheck(n == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of subdomain solver %" PetscInt_FMT " != 1", n);
        PetscCall(KSPGetPC(ksp[0], &inner));
        PetscCall(PCSetType(inner, PCSHELL)); /* compute the action of the inverse of the local Schur complement with a PCSHELL */
        b[0] = subA[0];
        PetscCall(MatCreateNest(PETSC_COMM_SELF, 2, nullptr, 2, nullptr, b, &N)); /* instead of computing inv(A11 - A10 inv(A00) A01), compute inv([A00, A01; A10, A11]) followed by a partial solution associated to the A11 block */
        if (!data) PetscCall(PetscObjectDereference((PetscObject)b[3]));
        PetscCall(PetscObjectDereference((PetscObject)b[1]));
        PetscCall(PetscObjectDereference((PetscObject)b[2]));
        PetscCall(PCCreate(PETSC_COMM_SELF, &s));
        PetscCall(PCSetOptionsPrefix(s, ((PetscObject)inner)->prefix));
        PetscCall(PCSetOptionsPrefix(inner, nullptr));
        PetscCall(KSPSetSkipPCSetFromOptions(ksp[0], PETSC_TRUE));
        PetscCall(PCSetType(s, PCLU));
        if (PetscDefined(HAVE_MUMPS)) PetscCall(PCFactorSetMatSolverType(s, MATSOLVERMUMPS)); /* only MATSOLVERMUMPS handles MATNEST, so for the others, e.g., MATSOLVERPETSC or MATSOLVERMKL_PARDISO, convert to plain MATAIJ */
        PetscCall(PCSetFromOptions(s));
        PetscCall(PCFactorGetMatSolverType(s, &type));
        PetscCall(PetscStrcmp(type, MATSOLVERMUMPS, &flg));
        PetscCall(MatGetLocalSize(A11, &n, nullptr));
        if (flg || n == 0) {
          PetscCall(PCSetOperators(s, N, N));
          if (n) {
            PetscCall(PCFactorGetMatrix(s, b));
            PetscCall(MatSetOptionsPrefix(*b, ((PetscObject)s)->prefix));
            n = -1;
            PetscCall(PetscOptionsGetInt(nullptr, ((PetscObject)s)->prefix, "-mat_mumps_icntl_26", &n, nullptr));
            if (n == 1) {                                /* allocates a square MatDense of size is[1]->map->n, so one */
              PetscCall(MatNestGetISs(N, is, nullptr));  /*  needs to be able to deactivate this path when dealing    */
              PetscCall(MatFactorSetSchurIS(*b, is[1])); /*  with a large constraint space in order to avoid OOM      */
            }
          } else PetscCall(PCSetType(s, PCNONE)); /* empty local Schur complement (e.g., centralized on another process) */
        } else {
          PetscCall(MatConvert(N, MATAIJ, MAT_INITIAL_MATRIX, b));
          PetscCall(PCSetOperators(s, N, *b));
          PetscCall(PetscObjectDereference((PetscObject)*b));
          PetscCall(PetscObjectTypeCompareAny((PetscObject)s, &flg, PCLU, PCCHOLESKY, PCILU, PCICC, PCQR, ""));
          if (flg) PetscCall(PCFactorGetMatrix(s, b)); /* MATSOLVERMKL_PARDISO cannot compute in PETSc (yet) a partial solution associated to the A11 block, only partial solution associated to the A00 block or full solution */
        }
        PetscCall(PetscNew(&p));
        p->first = s;
        if (n != 0) PetscCall(MatCreateVecs(*b, p->second, p->second + 1));
        else p->second[0] = p->second[1] = nullptr;
        PetscCall(PCShellSetContext(inner, p));
        PetscCall(PCShellSetApply(inner, PCApply_Nest));
        PetscCall(PCShellSetView(inner, PCView_Nest));
        PetscCall(PCShellSetDestroy(inner, PCDestroy_Nest));
        PetscCall(PetscObjectDereference((PetscObject)N));
        if (!data) {
          PetscCall(MatDestroy(&S));
          PetscCall(ISDestroy(&unsorted));
          PetscCall(MatDestroy(&C));
          PetscCall(ISDestroy(&uis));
          PetscCall(PetscFree(ctx));
#if PetscDefined(USE_DEBUG)
          PetscCall(ISDestroy(&dis));
          PetscCall(MatDestroy(&daux));
#endif
          PetscFunctionReturn(PETSC_SUCCESS);
        }
      }
      if (!data->levels[0]->scatter) {
        PetscCall(MatCreateVecs(P, &xin, nullptr));
        if (ismatis) PetscCall(MatDestroy(&P));
        PetscCall(VecScatterCreate(xin, data->is, data->levels[0]->D, nullptr, &data->levels[0]->scatter));
        PetscCall(VecDestroy(&xin));
      }
      if (data->levels[0]->P) {
        /* if the pattern is the same and PCSetUp() has previously succeeded, reuse HPDDM buffers and connectivity */
        PetscCall(HPDDM::Schwarz<PetscScalar>::destroy(data->levels[0], !pc->setupcalled || pc->flag == DIFFERENT_NONZERO_PATTERN ? PETSC_TRUE : PETSC_FALSE));
      }
      if (!data->levels[0]->P) data->levels[0]->P = new HPDDM::Schwarz<PetscScalar>();
      if (data->log_separate) PetscCall(PetscLogEventBegin(PC_HPDDM_SetUp[0], data->levels[0]->ksp, nullptr, nullptr, nullptr));
      else PetscCall(PetscLogEventBegin(PC_HPDDM_Strc, data->levels[0]->ksp, nullptr, nullptr, nullptr));
      /* HPDDM internal data structure */
      PetscCall(data->levels[0]->P->structure(loc, data->is, !ctx ? sub[0] : nullptr, ismatis ? C : data->aux, data->levels));
      if (!data->log_separate) PetscCall(PetscLogEventEnd(PC_HPDDM_Strc, data->levels[0]->ksp, nullptr, nullptr, nullptr));
      /* matrix pencil of the generalized eigenvalue problem on the overlap (GenEO) */
      if (!ctx) {
        if (data->deflation || overlap != -1) weighted = data->aux;
        else if (!data->B) {
          PetscBool cmp;

          PetscCall(MatDuplicate(sub[0], MAT_COPY_VALUES, &weighted));
          PetscCall(PetscObjectTypeCompareAny((PetscObject)weighted, &cmp, MATNORMAL, MATNORMALHERMITIAN, ""));
          if (cmp) flg = PETSC_FALSE;
          PetscCall(MatDiagonalScale(weighted, data->levels[0]->D, data->levels[0]->D));
          /* neither MatDuplicate() nor MatDiagonalScale() handles the symmetry options, so propagate the options explicitly */
          /* only useful for -mat_type baij -pc_hpddm_levels_1_st_pc_type cholesky (no problem with MATAIJ or MATSBAIJ)      */
          PetscCall(MatPropagateSymmetryOptions(sub[0], weighted));
          if (PetscDefined(USE_DEBUG) && PetscBool3ToBool(data->Neumann)) {
            Mat      *sub, A[3];
            PetscReal norm[2];
            PetscBool flg;

            PetscCall(PetscObjectTypeCompare((PetscObject)P, MATMPISBAIJ, &flg)); /* MatCreateSubMatrices() does not work with MATSBAIJ and unsorted ISes, so convert to MPIAIJ */
            if (flg) PetscCall(MatConvert(P, MATMPIAIJ, MAT_INITIAL_MATRIX, A));
            else {
              A[0] = P;
              PetscCall(PetscObjectReference((PetscObject)P));
            }
            PetscCall(MatCreateSubMatrices(A[0], 1, &data->is, &data->is, MAT_INITIAL_MATRIX, &sub));
            PetscCall(MatDiagonalScale(sub[0], data->levels[0]->D, data->levels[0]->D));
            PetscCall(MatConvert(sub[0], MATSEQAIJ, MAT_INITIAL_MATRIX, A + 1)); /* too many corner cases to handle (MATNORMAL, MATNORMALHERMITIAN, MATBAIJ with different block sizes...), so just MatConvert() to MATSEQAIJ since this is just for debugging */
            PetscCall(MatConvert(weighted, MATSEQAIJ, MAT_INITIAL_MATRIX, A + 2));
            PetscCall(MatAXPY(A[1], -1.0, A[2], UNKNOWN_NONZERO_PATTERN));
            PetscCall(MatNorm(A[1], NORM_FROBENIUS, norm));
            if (norm[0]) {
              PetscCall(MatNorm(A[2], NORM_FROBENIUS, norm + 1));
              PetscCheck(PetscAbsReal(norm[0] / norm[1]) < PetscSqrtReal(PETSC_SMALL), PETSC_COMM_SELF, PETSC_ERR_USER_INPUT, "Auxiliary Mat is different from the (assembled) subdomain Mat for the interior unknowns, so it cannot be the Neumann matrix, remove -%spc_hpddm_has_neumann", pcpre ? pcpre : "");
            }
            PetscCall(MatDestroySubMatrices(1, &sub));
            for (PetscInt i = 0; i < 3; ++i) PetscCall(MatDestroy(A + i));
          }
        } else weighted = data->B;
      } else weighted = nullptr;
      /* SLEPc is used inside the loaded symbol */
      PetscCall((*loadedSym)(data->levels[0]->P, data->is, ismatis ? C : (algebraic && !block && overlap == -1 ? sub[0] : (!ctx ? data->aux : S)), weighted, data->B, initial, data->levels));
      if (!ctx && data->share && overlap == -1) {
        Mat st[2];

        PetscCall(KSPGetOperators(ksp[0], st, st + 1));
        PetscCall(MatCopy(subA[0], st[0], structure));
        if (subA[1] != subA[0] || st[1] != st[0]) PetscCall(MatCopy(subA[1], st[1], SAME_NONZERO_PATTERN));
        PetscCall(PetscObjectDereference((PetscObject)subA[0]));
      }
      if (data->log_separate) PetscCall(PetscLogEventEnd(PC_HPDDM_SetUp[0], data->levels[0]->ksp, nullptr, nullptr, nullptr));
      if (ismatis) PetscCall(MatISGetLocalMat(C, &N));
      else N = data->aux;
      if (!ctx) P = sub[0];
      else P = S;
      /* going through the grid hierarchy */
      for (n = 1; n < data->N; ++n) {
        if (data->log_separate) PetscCall(PetscLogEventBegin(PC_HPDDM_SetUp[n], data->levels[n]->ksp, nullptr, nullptr, nullptr));
        /* method composed in the loaded symbol since there, SLEPc is used as well */
        PetscTryMethod(data->levels[0]->ksp, "PCHPDDMSetUp_Private_C", (Mat *, Mat *, PetscInt, PetscInt *const, PC_HPDDM_Level **const), (&P, &N, n, &data->N, data->levels));
        if (data->log_separate) PetscCall(PetscLogEventEnd(PC_HPDDM_SetUp[n], data->levels[n]->ksp, nullptr, nullptr, nullptr));
      }
      /* reset to NULL to avoid any faulty use */
      PetscCall(PetscObjectComposeFunction((PetscObject)data->levels[0]->ksp, "PCHPDDMSetUp_Private_C", nullptr));
      if (!ismatis) PetscCall(PetscObjectComposeFunction((PetscObject)pc->pmat, "PCHPDDMAlgebraicAuxiliaryMat_C", nullptr));
      else PetscCall(PetscObjectDereference((PetscObject)C)); /* matching PetscObjectReference() above */
      for (n = 0; n < data->N - 1; ++n)
        if (data->levels[n]->P) {
          /* HPDDM internal work buffers */
          data->levels[n]->P->setBuffer();
          data->levels[n]->P->super::start();
        }
      if (ismatis || !subdomains) PetscCall(PCHPDDMDestroySubMatrices_Private(PetscBool3ToBool(data->Neumann), PetscBool(algebraic && !block && overlap == -1), sub));
      if (ismatis) data->is = nullptr;
      for (n = 0; n < data->N - 1 + (reused > 0); ++n) {
        if (data->levels[n]->P) {
          PC spc;

          /* force the PC to be PCSHELL to do the coarse grid corrections */
          PetscCall(KSPSetSkipPCSetFromOptions(data->levels[n]->ksp, PETSC_TRUE));
          PetscCall(KSPGetPC(data->levels[n]->ksp, &spc));
          PetscCall(PCSetType(spc, PCSHELL));
          PetscCall(PCShellSetContext(spc, data->levels[n]));
          PetscCall(PCShellSetSetUp(spc, PCSetUp_HPDDMShell));
          PetscCall(PCShellSetApply(spc, PCApply_HPDDMShell));
          PetscCall(PCShellSetMatApply(spc, PCMatApply_HPDDMShell));
          PetscCall(PCShellSetApplyTranspose(spc, PCApplyTranspose_HPDDMShell));
          PetscCall(PCShellSetMatApplyTranspose(spc, PCMatApplyTranspose_HPDDMShell));
          if (ctx && n == 0) {
            Mat                               Amat, Pmat;
            PetscInt                          m, M;
            std::tuple<Mat, PetscSF, Vec[2]> *ctx;

            PetscCall(KSPGetOperators(data->levels[n]->ksp, nullptr, &Pmat));
            PetscCall(MatGetLocalSize(Pmat, &m, nullptr));
            PetscCall(MatGetSize(Pmat, &M, nullptr));
            PetscCall(PetscNew(&ctx));
            std::get<0>(*ctx) = S;
            std::get<1>(*ctx) = data->levels[n]->scatter;
            PetscCall(MatCreateShell(PetscObjectComm((PetscObject)data->levels[n]->ksp), m, m, M, M, ctx, &Amat));
            PetscCall(MatShellSetOperation(Amat, MATOP_MULT, (PetscErrorCodeFn *)MatMult_Schur<false>));
            PetscCall(MatShellSetOperation(Amat, MATOP_MULT_TRANSPOSE, (PetscErrorCodeFn *)MatMult_Schur<true>));
            PetscCall(MatShellSetOperation(Amat, MATOP_DESTROY, (PetscErrorCodeFn *)MatDestroy_Schur));
            PetscCall(MatCreateVecs(S, std::get<2>(*ctx), std::get<2>(*ctx) + 1));
            PetscCall(KSPSetOperators(data->levels[n]->ksp, Amat, Pmat));
            PetscCall(PetscObjectDereference((PetscObject)Amat));
          }
          PetscCall(PCShellSetDestroy(spc, PCDestroy_HPDDMShell));
          if (!data->levels[n]->pc) PetscCall(PCCreate(PetscObjectComm((PetscObject)data->levels[n]->ksp), &data->levels[n]->pc));
          if (n < reused) {
            PetscCall(PCSetReusePreconditioner(spc, PETSC_TRUE));
            PetscCall(PCSetReusePreconditioner(data->levels[n]->pc, PETSC_TRUE));
          }
          PetscCall(PCSetUp(spc));
        }
      }
      if (ctx) PetscCall(MatDestroy(&S));
      if (overlap == -1) PetscCall(PetscObjectComposeFunction((PetscObject)pc->pmat, "PCHPDDMAlgebraicAuxiliaryMat_Private_C", nullptr));
    } else flg = reused ? PETSC_FALSE : PETSC_TRUE;
    if (!ismatis && subdomains) {
      if (flg) PetscCall(KSPGetPC(data->levels[0]->ksp, &inner));
      else inner = data->levels[0]->pc;
      if (inner) {
        if (!inner->setupcalled) PetscCall(PCSetType(inner, PCASM));
        PetscCall(PCSetFromOptions(inner));
        PetscCall(PetscStrcmp(((PetscObject)inner)->type_name, PCASM, &flg));
        if (flg) {
          if (!inner->setupcalled) { /* evaluates to PETSC_FALSE when -pc_hpddm_block_splitting */
            IS sorted;               /* PCASM will sort the input IS, duplicate it to return an unmodified (PCHPDDM) input IS */

            PetscCall(ISDuplicate(is[0], &sorted));
            PetscCall(PCASMSetLocalSubdomains(inner, 1, &sorted, &loc));
            PetscCall(PetscObjectDereference((PetscObject)sorted));
          }
          if (!PetscBool3ToBool(data->Neumann) && data->N > 1) { /* subdomain matrices are already created for the eigenproblem, reuse them for the fine-level PC */
            PetscCall(PCHPDDMPermute_Private(*is, nullptr, nullptr, sub[0], &P, nullptr));
            PetscCall(PCHPDDMCommunicationAvoidingPCASM_Private(inner, P, algebraic));
            PetscCall(PetscObjectDereference((PetscObject)P));
          }
        }
      }
      if (data->N > 1) {
        if (overlap != 1) PetscCall(PCHPDDMDestroySubMatrices_Private(PetscBool3ToBool(data->Neumann), PetscBool(algebraic && !block && overlap == -1), sub));
        if (overlap == 1) PetscCall(MatDestroy(subA));
      }
    }
    PetscCall(ISDestroy(&loc));
  } else data->N = 1 + reused; /* enforce this value to 1 + reused if there is no way to build another level */
  if (requested != data->N + reused) {
    PetscCall(PetscInfo(pc, "%" PetscInt_FMT " levels requested, only %" PetscInt_FMT " built + %" PetscInt_FMT " reused. Options for level(s) > %" PetscInt_FMT ", including -%spc_hpddm_coarse_ will not be taken into account\n", requested, data->N, reused,
                        data->N, pcpre ? pcpre : ""));
    PetscCall(PetscInfo(pc, "It is best to tune parameters, e.g., a higher value for -%spc_hpddm_levels_%" PetscInt_FMT "_eps_threshold_absolute or a lower value for -%spc_hpddm_levels_%" PetscInt_FMT "_svd_threshold_relative, so that at least one local deflation vector will be selected\n", pcpre ? pcpre : "",
                        data->N, pcpre ? pcpre : "", data->N));
    /* cannot use PCDestroy_HPDDMShell() because PCSHELL not set for unassembled levels */
    for (n = data->N - 1; n < requested - 1; ++n) {
      if (data->levels[n]->P) {
        PetscCall(HPDDM::Schwarz<PetscScalar>::destroy(data->levels[n], PETSC_TRUE));
        PetscCall(VecDestroyVecs(1, &data->levels[n]->v[0]));
        PetscCall(VecDestroyVecs(2, &data->levels[n]->v[1]));
        PetscCall(MatDestroy(data->levels[n]->V));
        PetscCall(MatDestroy(data->levels[n]->V + 1));
        PetscCall(MatDestroy(data->levels[n]->V + 2));
        PetscCall(VecDestroy(&data->levels[n]->D));
        PetscCall(PetscSFDestroy(&data->levels[n]->scatter));
      }
    }
    if (reused) {
      for (n = reused; n < PETSC_PCHPDDM_MAXLEVELS && data->levels[n]; ++n) {
        PetscCall(KSPDestroy(&data->levels[n]->ksp));
        PetscCall(PCDestroy(&data->levels[n]->pc));
      }
    }
    PetscCheck(!PetscDefined(USE_DEBUG), PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONG, "%" PetscInt_FMT " levels requested, only %" PetscInt_FMT " built + %" PetscInt_FMT " reused. Options for level(s) > %" PetscInt_FMT ", including -%spc_hpddm_coarse_ will not be taken into account. It is best to tune parameters, e.g., a higher value for -%spc_hpddm_levels_%" PetscInt_FMT "_eps_threshold or a lower value for -%spc_hpddm_levels_%" PetscInt_FMT "_svd_threshold_relative, so that at least one local deflation vector will be selected. If you don't want this to error out, compile --with-debugging=0", requested,
               data->N, reused, data->N, pcpre ? pcpre : "", pcpre ? pcpre : "", data->N, pcpre ? pcpre : "", data->N);
  }
  /* these solvers are created after PCSetFromOptions() is called */
  if (pc->setfromoptionscalled) {
    for (n = 0; n < data->N; ++n) {
      if (data->levels[n]->ksp) PetscCall(KSPSetFromOptions(data->levels[n]->ksp));
      if (data->levels[n]->pc) PetscCall(PCSetFromOptions(data->levels[n]->pc));
    }
    pc->setfromoptionscalled = 0;
  }
  data->N += reused;
  if (data->share && swap) {
    /* swap back pointers so that variables follow the user-provided numbering */
    std::swap(C, data->aux);
    std::swap(uis, data->is);
    PetscCall(MatDestroy(&C));
    PetscCall(ISDestroy(&uis));
  }
  if (algebraic) PetscCall(MatDestroy(&data->aux));
  if (unsorted && unsorted != is[0]) {
    PetscCall(ISCopy(unsorted, data->is));
    PetscCall(ISDestroy(&unsorted));
  }
#if PetscDefined(USE_DEBUG)
  PetscCheck((data->is && dis) || (!data->is && !dis), PETSC_COMM_SELF, PETSC_ERR_PLIB, "An IS pointer is NULL but not the other: input IS pointer (%p) v. output IS pointer (%p)", (void *)dis, (void *)data->is);
  if (data->is) {
    PetscCall(ISEqualUnsorted(data->is, dis, &flg));
    PetscCall(ISDestroy(&dis));
    PetscCheck(flg, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Input IS and output IS are not equal");
  }
  PetscCheck((data->aux && daux) || (!data->aux && !daux), PETSC_COMM_SELF, PETSC_ERR_PLIB, "A Mat pointer is NULL but not the other: input Mat pointer (%p) v. output Mat pointer (%p)", (void *)daux, (void *)data->aux);
  if (data->aux) {
    PetscCall(MatMultEqual(data->aux, daux, 20, &flg));
    PetscCall(MatDestroy(&daux));
    PetscCheck(flg, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Input Mat and output Mat are not equal");
  }
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCHPDDMSetCoarseCorrectionType - Sets the coarse correction type.

  Collective

  Input Parameters:
+ pc   - preconditioner context
- type - `PC_HPDDM_COARSE_CORRECTION_DEFLATED`, `PC_HPDDM_COARSE_CORRECTION_ADDITIVE`, `PC_HPDDM_COARSE_CORRECTION_BALANCED`, or `PC_HPDDM_COARSE_CORRECTION_NONE`

  Options Database Key:
. -pc_hpddm_coarse_correction <deflated, additive, balanced, none> - type of coarse correction to apply

  Level: intermediate

.seealso: [](ch_ksp), `PCHPDDMGetCoarseCorrectionType()`, `PCHPDDM`, `PCHPDDMCoarseCorrectionType`
@*/
PetscErrorCode PCHPDDMSetCoarseCorrectionType(PC pc, PCHPDDMCoarseCorrectionType type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscValidLogicalCollectiveEnum(pc, type, 2);
  PetscTryMethod(pc, "PCHPDDMSetCoarseCorrectionType_C", (PC, PCHPDDMCoarseCorrectionType), (pc, type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCHPDDMGetCoarseCorrectionType - Gets the coarse correction type.

  Input Parameter:
. pc - preconditioner context

  Output Parameter:
. type - `PC_HPDDM_COARSE_CORRECTION_DEFLATED`, `PC_HPDDM_COARSE_CORRECTION_ADDITIVE`, `PC_HPDDM_COARSE_CORRECTION_BALANCED`, or `PC_HPDDM_COARSE_CORRECTION_NONE`

  Level: intermediate

.seealso: [](ch_ksp), `PCHPDDMSetCoarseCorrectionType()`, `PCHPDDM`, `PCHPDDMCoarseCorrectionType`
@*/
PetscErrorCode PCHPDDMGetCoarseCorrectionType(PC pc, PCHPDDMCoarseCorrectionType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  if (type) {
    PetscAssertPointer(type, 2);
    PetscUseMethod(pc, "PCHPDDMGetCoarseCorrectionType_C", (PC, PCHPDDMCoarseCorrectionType *), (pc, type));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCHPDDMSetCoarseCorrectionType_HPDDM(PC pc, PCHPDDMCoarseCorrectionType type)
{
  PC_HPDDM *data = (PC_HPDDM *)pc->data;

  PetscFunctionBegin;
  data->correction = type;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCHPDDMGetCoarseCorrectionType_HPDDM(PC pc, PCHPDDMCoarseCorrectionType *type)
{
  PC_HPDDM *data = (PC_HPDDM *)pc->data;

  PetscFunctionBegin;
  *type = data->correction;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCHPDDMSetSTShareSubKSP - Sets whether the `KSP` in SLEPc `ST` and the fine-level subdomain solver should be shared.

  Input Parameters:
+ pc    - preconditioner context
- share - whether the `KSP` should be shared or not

  Note:
  This is not the same as `PCSetReusePreconditioner()`. Given certain conditions (visible using -info), a symbolic factorization can be skipped
  when using a subdomain `PCType` such as `PCLU` or `PCCHOLESKY`.

  Level: advanced

.seealso: [](ch_ksp), `PCHPDDM`, `PCHPDDMGetSTShareSubKSP()`
@*/
PetscErrorCode PCHPDDMSetSTShareSubKSP(PC pc, PetscBool share)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscTryMethod(pc, "PCHPDDMSetSTShareSubKSP_C", (PC, PetscBool), (pc, share));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCHPDDMGetSTShareSubKSP - Gets whether the `KSP` in SLEPc `ST` and the fine-level subdomain solver is shared.

  Input Parameter:
. pc - preconditioner context

  Output Parameter:
. share - whether the `KSP` is shared or not

  Note:
  This is not the same as `PCGetReusePreconditioner()`. The return value is unlikely to be true, but when it is, a symbolic factorization can be skipped
  when using a subdomain `PCType` such as `PCLU` or `PCCHOLESKY`.

  Level: advanced

.seealso: [](ch_ksp), `PCHPDDM`, `PCHPDDMSetSTShareSubKSP()`
@*/
PetscErrorCode PCHPDDMGetSTShareSubKSP(PC pc, PetscBool *share)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  if (share) {
    PetscAssertPointer(share, 2);
    PetscUseMethod(pc, "PCHPDDMGetSTShareSubKSP_C", (PC, PetscBool *), (pc, share));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCHPDDMSetSTShareSubKSP_HPDDM(PC pc, PetscBool share)
{
  PC_HPDDM *data = (PC_HPDDM *)pc->data;

  PetscFunctionBegin;
  data->share = share;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCHPDDMGetSTShareSubKSP_HPDDM(PC pc, PetscBool *share)
{
  PC_HPDDM *data = (PC_HPDDM *)pc->data;

  PetscFunctionBegin;
  *share = data->share;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCHPDDMSetDeflationMat - Sets the deflation space used to assemble a coarser operator.

  Input Parameters:
+ pc - preconditioner context
. is - index set of the local deflation matrix
- U  - deflation sequential matrix stored as a `MATSEQDENSE`

  Level: advanced

.seealso: [](ch_ksp), `PCHPDDM`, `PCDeflationSetSpace()`, `PCMGSetRestriction()`
@*/
PetscErrorCode PCHPDDMSetDeflationMat(PC pc, IS is, Mat U)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscValidHeaderSpecific(is, IS_CLASSID, 2);
  PetscValidHeaderSpecific(U, MAT_CLASSID, 3);
  PetscTryMethod(pc, "PCHPDDMSetDeflationMat_C", (PC, IS, Mat), (pc, is, U));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCHPDDMSetDeflationMat_HPDDM(PC pc, IS is, Mat U)
{
  PetscFunctionBegin;
  PetscCall(PCHPDDMSetAuxiliaryMat_Private(pc, is, U, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode HPDDMLoadDL_Private(PetscBool *found)
{
  PetscBool flg;
  char      lib[PETSC_MAX_PATH_LEN], dlib[PETSC_MAX_PATH_LEN], dir[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  PetscAssertPointer(found, 1);
  PetscCall(PetscStrncpy(dir, "${PETSC_LIB_DIR}", sizeof(dir)));
  PetscCall(PetscOptionsGetString(nullptr, nullptr, "-hpddm_dir", dir, sizeof(dir), nullptr));
  PetscCall(PetscSNPrintf(lib, sizeof(lib), "%s/libhpddm_petsc", dir));
  PetscCall(PetscDLLibraryRetrieve(PETSC_COMM_SELF, lib, dlib, 1024, found));
#if defined(SLEPC_LIB_DIR) /* this variable is passed during SLEPc ./configure when PETSc has not been configured   */
  if (!*found) {           /* with --download-hpddm since slepcconf.h is not yet built (and thus can't be included) */
    PetscCall(PetscStrncpy(dir, HPDDM_STR(SLEPC_LIB_DIR), sizeof(dir)));
    PetscCall(PetscSNPrintf(lib, sizeof(lib), "%s/libhpddm_petsc", dir));
    PetscCall(PetscDLLibraryRetrieve(PETSC_COMM_SELF, lib, dlib, 1024, found));
  }
#endif
  if (!*found) { /* probable options for this to evaluate to PETSC_TRUE: system inconsistency (libhpddm_petsc moved by user?) or PETSc configured without --download-slepc */
    PetscCall(PetscOptionsGetenv(PETSC_COMM_SELF, "SLEPC_DIR", dir, sizeof(dir), &flg));
    if (flg) { /* if both PETSc and SLEPc are configured with --download-hpddm but PETSc has been configured without --download-slepc, one must ensure that libslepc is loaded before libhpddm_petsc */
      PetscCall(PetscSNPrintf(lib, sizeof(lib), "%s/lib/libslepc", dir));
      PetscCall(PetscDLLibraryRetrieve(PETSC_COMM_SELF, lib, dlib, 1024, found));
      PetscCheck(*found, PETSC_COMM_SELF, PETSC_ERR_PLIB, "%s not found but SLEPC_DIR=%s", lib, dir);
      PetscCall(PetscDLLibraryAppend(PETSC_COMM_SELF, &PetscDLLibrariesLoaded, dlib));
      PetscCall(PetscSNPrintf(lib, sizeof(lib), "%s/lib/libhpddm_petsc", dir)); /* libhpddm_petsc is always in the same directory as libslepc */
      PetscCall(PetscDLLibraryRetrieve(PETSC_COMM_SELF, lib, dlib, 1024, found));
    }
  }
  PetscCheck(*found, PETSC_COMM_SELF, PETSC_ERR_PLIB, "%s not found", lib);
  PetscCall(PetscDLLibraryAppend(PETSC_COMM_SELF, &PetscDLLibrariesLoaded, dlib));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   PCHPDDM - Interface with the HPDDM library.

   This `PC` may be used to build multilevel spectral domain decomposition methods based on the GenEO framework {cite}`spillane2011robust` {cite}`al2021multilevel`.
   It may be viewed as an alternative to spectral
   AMGe or `PCBDDC` with adaptive selection of constraints. The interface is explained in details in {cite}`jolivetromanzampini2020`

   The matrix used for building the preconditioner (Pmat) may be unassembled (`MATIS`), assembled (`MATAIJ`, `MATBAIJ`, or `MATSBAIJ`), hierarchical (`MATHTOOL`), `MATNORMAL`, `MATNORMALHERMITIAN`, or `MATSCHURCOMPLEMENT` (when `PCHPDDM` is used as part of an outer `PCFIELDSPLIT`).

   For multilevel preconditioning, when using an assembled or hierarchical Pmat, one must provide an auxiliary local `Mat` (unassembled local operator for GenEO) using
   `PCHPDDMSetAuxiliaryMat()`. Calling this routine is not needed when using a `MATIS` Pmat, assembly is done internally using `MatConvert()`.

   Options Database Keys:
+   -pc_hpddm_define_subdomains <true, default=false>    - on the finest level, calls `PCASMSetLocalSubdomains()` with the `IS` supplied in `PCHPDDMSetAuxiliaryMat()`
                                                         (not relevant with an unassembled Pmat)
.   -pc_hpddm_has_neumann <true, default=false>          - on the finest level, informs the `PC` that the local Neumann matrix is supplied in `PCHPDDMSetAuxiliaryMat()`
-   -pc_hpddm_coarse_correction <type, default=deflated> - determines the `PCHPDDMCoarseCorrectionType` when calling `PCApply()`

   Options for subdomain solvers, subdomain eigensolvers (for computing deflation vectors), and the coarse solver can be set using the following options database prefixes.
.vb
      -pc_hpddm_levels_%d_pc_
      -pc_hpddm_levels_%d_ksp_
      -pc_hpddm_levels_%d_eps_
      -pc_hpddm_levels_%d_p
      -pc_hpddm_levels_%d_mat_type
      -pc_hpddm_coarse_
      -pc_hpddm_coarse_p
      -pc_hpddm_coarse_mat_type
      -pc_hpddm_coarse_mat_filter
.ve

   E.g., -pc_hpddm_levels_1_sub_pc_type lu -pc_hpddm_levels_1_eps_nev 10 -pc_hpddm_levels_2_p 4 -pc_hpddm_levels_2_sub_pc_type lu -pc_hpddm_levels_2_eps_nev 10
    -pc_hpddm_coarse_p 2 -pc_hpddm_coarse_mat_type baij will use 10 deflation vectors per subdomain on the fine "level 1",
    aggregate the fine subdomains into 4 "level 2" subdomains, then use 10 deflation vectors per subdomain on "level 2",
    and assemble the coarse matrix (of dimension 4 x 10 = 40) on two processes as a `MATBAIJ` (default is `MATSBAIJ`).

   In order to activate a "level N+1" coarse correction, it is mandatory to call -pc_hpddm_levels_N_eps_nev <nu> or -pc_hpddm_levels_N_eps_threshold_absolute <val>. The default -pc_hpddm_coarse_p value is 1, meaning that the coarse operator is aggregated on a single process.

   Level: intermediate

   Notes:
   This preconditioner requires that PETSc is built with SLEPc (``--download-slepc``).

   By default, the underlying concurrent eigenproblems
   are solved using SLEPc shift-and-invert spectral transformation. This is usually what gives the best performance for GenEO, cf.
   {cite}`spillane2011robust` {cite}`jolivet2013scalabledd`. As
   stated above, SLEPc options are available through -pc_hpddm_levels_%d_, e.g., -pc_hpddm_levels_1_eps_type arpack -pc_hpddm_levels_1_eps_nev 10
   -pc_hpddm_levels_1_st_type sinvert. There are furthermore three options related to the (subdomain-wise local) eigensolver that are not described in
   SLEPc documentation since they are specific to `PCHPDDM`.
.vb
      -pc_hpddm_levels_1_st_share_sub_ksp
      -pc_hpddm_levels_%d_eps_threshold_absolute
      -pc_hpddm_levels_1_eps_use_inertia
.ve

   The first option from the list only applies to the fine-level eigensolver, see `PCHPDDMSetSTShareSubKSP()`. The second option from the list is
   used to filter eigenmodes retrieved after convergence of `EPSSolve()` at "level N" such that eigenvectors used to define a "level N+1" coarse
   correction are associated to eigenvalues whose magnitude are lower or equal than -pc_hpddm_levels_N_eps_threshold_absolute. When using an `EPS` which cannot
   determine a priori the proper -pc_hpddm_levels_N_eps_nev such that all wanted eigenmodes are retrieved, it is possible to get an estimation of the
   correct value using the third option from the list, -pc_hpddm_levels_1_eps_use_inertia, see `MatGetInertia()`. In that case, there is no need
   to supply -pc_hpddm_levels_1_eps_nev. This last option also only applies to the fine-level (N = 1) eigensolver.

   See also {cite}`dolean2015introduction`, {cite}`al2022robust`, {cite}`al2022robustpd`, and {cite}`nataf2022recent`

.seealso: [](ch_ksp), `PCCreate()`, `PCSetType()`, `PCType`, `PC`, `PCHPDDMSetAuxiliaryMat()`, `MATIS`, `PCBDDC`, `PCDEFLATION`, `PCTELESCOPE`, `PCASM`,
          `PCHPDDMSetCoarseCorrectionType()`, `PCHPDDMHasNeumannMat()`, `PCHPDDMSetRHSMat()`, `PCHPDDMSetDeflationMat()`, `PCHPDDMSetSTShareSubKSP()`,
          `PCHPDDMGetSTShareSubKSP()`, `PCHPDDMGetCoarseCorrectionType()`, `PCHPDDMGetComplexities()`
M*/
PETSC_EXTERN PetscErrorCode PCCreate_HPDDM(PC pc)
{
  PC_HPDDM *data;
  PetscBool found;

  PetscFunctionBegin;
  if (!loadedSym) {
    PetscCall(HPDDMLoadDL_Private(&found));
    if (found) PetscCall(PetscDLLibrarySym(PETSC_COMM_SELF, &PetscDLLibrariesLoaded, nullptr, "PCHPDDM_Internal", (void **)&loadedSym));
  }
  PetscCheck(loadedSym, PETSC_COMM_SELF, PETSC_ERR_PLIB, "PCHPDDM_Internal symbol not found in loaded libhpddm_petsc");
  PetscCall(PetscNew(&data));
  pc->data                   = data;
  data->Neumann              = PETSC_BOOL3_UNKNOWN;
  pc->ops->reset             = PCReset_HPDDM;
  pc->ops->destroy           = PCDestroy_HPDDM;
  pc->ops->setfromoptions    = PCSetFromOptions_HPDDM;
  pc->ops->setup             = PCSetUp_HPDDM;
  pc->ops->apply             = PCApply_HPDDM<false>;
  pc->ops->matapply          = PCMatApply_HPDDM<false>;
  pc->ops->applytranspose    = PCApply_HPDDM<true>;
  pc->ops->matapplytranspose = PCMatApply_HPDDM<true>;
  pc->ops->view              = PCView_HPDDM;
  pc->ops->presolve          = PCPreSolve_HPDDM;

  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCHPDDMSetAuxiliaryMat_C", PCHPDDMSetAuxiliaryMat_HPDDM));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCHPDDMHasNeumannMat_C", PCHPDDMHasNeumannMat_HPDDM));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCHPDDMSetRHSMat_C", PCHPDDMSetRHSMat_HPDDM));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCHPDDMSetCoarseCorrectionType_C", PCHPDDMSetCoarseCorrectionType_HPDDM));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCHPDDMGetCoarseCorrectionType_C", PCHPDDMGetCoarseCorrectionType_HPDDM));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCHPDDMSetSTShareSubKSP_C", PCHPDDMSetSTShareSubKSP_HPDDM));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCHPDDMGetSTShareSubKSP_C", PCHPDDMGetSTShareSubKSP_HPDDM));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCHPDDMSetDeflationMat_C", PCHPDDMSetDeflationMat_HPDDM));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PCHPDDMInitializePackage - This function initializes everything in the `PCHPDDM` package. It is called from `PCInitializePackage()`.

  Level: developer

.seealso: [](ch_ksp), `PetscInitialize()`
@*/
PetscErrorCode PCHPDDMInitializePackage(void)
{
  char ename[32];

  PetscFunctionBegin;
  if (PCHPDDMPackageInitialized) PetscFunctionReturn(PETSC_SUCCESS);
  PCHPDDMPackageInitialized = PETSC_TRUE;
  PetscCall(PetscRegisterFinalize(PCHPDDMFinalizePackage));
  /* general events registered once during package initialization */
  /* some of these events are not triggered in libpetsc,          */
  /* but rather directly in libhpddm_petsc,                       */
  /* which is in charge of performing the following operations    */

  /* domain decomposition structure from Pmat sparsity pattern    */
  PetscCall(PetscLogEventRegister("PCHPDDMStrc", PC_CLASSID, &PC_HPDDM_Strc));
  /* Galerkin product, redistribution, and setup (not triggered in libpetsc)                */
  PetscCall(PetscLogEventRegister("PCHPDDMPtAP", PC_CLASSID, &PC_HPDDM_PtAP));
  /* Galerkin product with summation, redistribution, and setup (not triggered in libpetsc) */
  PetscCall(PetscLogEventRegister("PCHPDDMPtBP", PC_CLASSID, &PC_HPDDM_PtBP));
  /* next level construction using PtAP and PtBP (not triggered in libpetsc)                */
  PetscCall(PetscLogEventRegister("PCHPDDMNext", PC_CLASSID, &PC_HPDDM_Next));
  static_assert(PETSC_PCHPDDM_MAXLEVELS <= 9, "PETSC_PCHPDDM_MAXLEVELS value is too high");
  for (PetscInt i = 1; i < PETSC_PCHPDDM_MAXLEVELS; ++i) {
    PetscCall(PetscSNPrintf(ename, sizeof(ename), "PCHPDDMSetUp L%1" PetscInt_FMT, i));
    /* events during a PCSetUp() at level #i _except_ the assembly */
    /* of the Galerkin operator of the coarser level #(i + 1)      */
    PetscCall(PetscLogEventRegister(ename, PC_CLASSID, &PC_HPDDM_SetUp[i - 1]));
    PetscCall(PetscSNPrintf(ename, sizeof(ename), "PCHPDDMSolve L%1" PetscInt_FMT, i));
    /* events during a PCApply() at level #i _except_              */
    /* the KSPSolve() of the coarser level #(i + 1)                */
    PetscCall(PetscLogEventRegister(ename, PC_CLASSID, &PC_HPDDM_Solve[i - 1]));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PCHPDDMFinalizePackage - This function frees everything from the `PCHPDDM` package. It is called from `PetscFinalize()`.

  Level: developer

.seealso: [](ch_ksp), `PetscFinalize()`
@*/
PetscErrorCode PCHPDDMFinalizePackage(void)
{
  PetscFunctionBegin;
  PCHPDDMPackageInitialized = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMult_Harmonic(Mat A, Vec x, Vec y)
{
  Harmonic h; /* [ A_00  A_01       ], furthermore, A_00 = [ A_loc,loc  A_loc,ovl ], thus, A_01 = [         ] */
              /* [ A_10  A_11  A_12 ]                      [ A_ovl,loc  A_ovl,ovl ]               [ A_ovl,1 ] */
  Vec sub;    /*  y = A x = R_loc R_0 [ A_00  A_01 ]^-1                                   R_loc = [  I_loc  ] */
              /*                      [ A_10  A_11 ]    R_1^T A_12 x                              [         ] */
  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A, &h));
  PetscCall(VecSet(h->v, 0.0));
  PetscCall(VecGetSubVector(h->v, h->is[0], &sub));
  PetscCall(MatMult(h->A[0], x, sub));
  PetscCall(VecRestoreSubVector(h->v, h->is[0], &sub));
  PetscCall(KSPSolve(h->ksp, h->v, h->v));
  PetscCall(VecISCopy(h->v, h->is[1], SCATTER_REVERSE, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMultTranspose_Harmonic(Mat A, Vec y, Vec x)
{
  Harmonic h;   /* x = A^T y =            [ A_00  A_01 ]^-T R_0^T R_loc^T y */
  Vec      sub; /*             A_12^T R_1 [ A_10  A_11 ]                    */

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A, &h));
  PetscCall(VecSet(h->v, 0.0));
  PetscCall(VecISCopy(h->v, h->is[1], SCATTER_FORWARD, y));
  PetscCall(KSPSolveTranspose(h->ksp, h->v, h->v));
  PetscCall(VecGetSubVector(h->v, h->is[0], &sub));
  PetscCall(MatMultTranspose(h->A[0], sub, x));
  PetscCall(VecRestoreSubVector(h->v, h->is[0], &sub));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatProduct_AB_Harmonic(Mat S, Mat X, Mat Y, void *)
{
  Harmonic h;
  Mat      A, B;
  Vec      a, b;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(S, &h));
  PetscCall(MatMatMult(h->A[0], X, MAT_INITIAL_MATRIX, PETSC_CURRENT, &A));
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, h->ksp->pc->mat->rmap->n, A->cmap->n, nullptr, &B));
  for (PetscInt i = 0; i < A->cmap->n; ++i) {
    PetscCall(MatDenseGetColumnVecRead(A, i, &a));
    PetscCall(MatDenseGetColumnVecWrite(B, i, &b));
    PetscCall(VecISCopy(b, h->is[0], SCATTER_FORWARD, a));
    PetscCall(MatDenseRestoreColumnVecWrite(B, i, &b));
    PetscCall(MatDenseRestoreColumnVecRead(A, i, &a));
  }
  PetscCall(MatDestroy(&A));
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, h->ksp->pc->mat->rmap->n, B->cmap->n, nullptr, &A));
  PetscCall(KSPMatSolve(h->ksp, B, A));
  PetscCall(MatDestroy(&B));
  for (PetscInt i = 0; i < A->cmap->n; ++i) {
    PetscCall(MatDenseGetColumnVecRead(A, i, &a));
    PetscCall(MatDenseGetColumnVecWrite(Y, i, &b));
    PetscCall(VecISCopy(a, h->is[1], SCATTER_REVERSE, b));
    PetscCall(MatDenseRestoreColumnVecWrite(Y, i, &b));
    PetscCall(MatDenseRestoreColumnVecRead(A, i, &a));
  }
  PetscCall(MatDestroy(&A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDestroy_Harmonic(Mat A)
{
  Harmonic h;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A, &h));
  for (PetscInt i = 0; i < 5; ++i) PetscCall(ISDestroy(h->is + i));
  PetscCall(PetscFree(h->is));
  PetscCall(VecDestroy(&h->v));
  for (PetscInt i = 0; i < 2; ++i) PetscCall(MatDestroy(h->A + i));
  PetscCall(PetscFree(h->A));
  PetscCall(KSPDestroy(&h->ksp));
  PetscCall(PetscFree(h));
  PetscFunctionReturn(PETSC_SUCCESS);
}

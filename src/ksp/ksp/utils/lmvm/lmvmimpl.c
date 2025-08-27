#include <petscdevice.h>
#include <../src/ksp/ksp/utils/lmvm/lmvm.h> /*I "petscksp.h" I*/
#include <petsc/private/deviceimpl.h>
#include "blas_cyclic/blas_cyclic.h"
#include "rescale/symbrdnrescale.h"

PetscLogEvent MATLMVM_Update;

static PetscBool MatLMVMPackageInitialized = PETSC_FALSE;

static PetscErrorCode MatLMVMPackageInitialize(void)
{
  PetscFunctionBegin;
  if (MatLMVMPackageInitialized) PetscFunctionReturn(PETSC_SUCCESS);
  MatLMVMPackageInitialized = PETSC_TRUE;
  PetscCall(PetscLogEventRegister("AXPBYCyclic", MAT_CLASSID, &AXPBY_Cyc));
  PetscCall(PetscLogEventRegister("DMVCyclic", MAT_CLASSID, &DMV_Cyc));
  PetscCall(PetscLogEventRegister("DSVCyclic", MAT_CLASSID, &DSV_Cyc));
  PetscCall(PetscLogEventRegister("TRSVCyclic", MAT_CLASSID, &TRSV_Cyc));
  PetscCall(PetscLogEventRegister("GEMVCyclic", MAT_CLASSID, &GEMV_Cyc));
  PetscCall(PetscLogEventRegister("HEMVCyclic", MAT_CLASSID, &HEMV_Cyc));
  PetscCall(PetscLogEventRegister("LMBasisGEMM", MAT_CLASSID, &LMBASIS_GEMM));
  PetscCall(PetscLogEventRegister("LMBasisGEMV", MAT_CLASSID, &LMBASIS_GEMV));
  PetscCall(PetscLogEventRegister("LMBasisGEMVH", MAT_CLASSID, &LMBASIS_GEMVH));
  PetscCall(PetscLogEventRegister("LMProdsMult", MAT_CLASSID, &LMPROD_Mult));
  PetscCall(PetscLogEventRegister("LMProdsSolve", MAT_CLASSID, &LMPROD_Solve));
  PetscCall(PetscLogEventRegister("LMProdsUpdate", MAT_CLASSID, &LMPROD_Update));
  PetscCall(PetscLogEventRegister("MatLMVMUpdate", MAT_CLASSID, &MATLMVM_Update));
  PetscCall(PetscLogEventRegister("SymBrdnRescale", MAT_CLASSID, &SBRDN_Rescale));
  PetscFunctionReturn(PETSC_SUCCESS);
}

const char *const MatLMVMMultAlgorithms[] = {
  "recursive", "dense", "compact_dense", "MatLMVMMatvecTypes", "MATLMVM_MATVEC_", NULL,
};

PetscBool  ByrdNocedalSchnabelCite       = PETSC_FALSE;
const char ByrdNocedalSchnabelCitation[] = "@article{Byrd1994,"
                                           "  title = {Representations of quasi-Newton matrices and their use in limited memory methods},"
                                           "  volume = {63},"
                                           "  ISSN = {1436-4646},"
                                           "  url = {http://dx.doi.org/10.1007/BF01582063},"
                                           "  DOI = {10.1007/bf01582063},"
                                           "  number = {1-3},"
                                           "  journal = {Mathematical Programming},"
                                           "  publisher = {Springer Science and Business Media LLC},"
                                           "  author = {Byrd,  Richard H. and Nocedal,  Jorge and Schnabel,  Robert B.},"
                                           "  year = {1994},"
                                           "  month = jan,"
                                           "  pages = {129-156}"
                                           "}\n";

PETSC_INTERN PetscErrorCode MatReset_LMVM(Mat B, MatLMVMResetMode mode)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;

  PetscFunctionBegin;
  lmvm->k        = 0;
  lmvm->prev_set = PETSC_FALSE;
  lmvm->shift    = 0.0;
  if (MatLMVMResetClearsBases(mode)) {
    for (PetscInt i = 0; i < LMBASIS_END; i++) PetscCall(LMBasisDestroy(&lmvm->basis[i]));
    for (PetscInt k = 0; k < LMBLOCK_END; k++) {
      for (PetscInt i = 0; i < LMBASIS_END; i++) {
        for (PetscInt j = 0; j < LMBASIS_END; j++) PetscCall(LMProductsDestroy(&lmvm->products[k][i][j]));
      }
    }
    B->preallocated = PETSC_FALSE; // MatSetUp() needs to be run to create at least the S and Y bases
  } else {
    for (PetscInt i = 0; i < LMBASIS_END; i++) PetscCall(LMBasisReset(lmvm->basis[i]));
    for (PetscInt k = 0; k < LMBLOCK_END; k++) {
      for (PetscInt i = 0; i < LMBASIS_END; i++) {
        for (PetscInt j = 0; j < LMBASIS_END; j++) PetscCall(LMProductsReset(lmvm->products[k][i][j]));
      }
    }
  }
  if (MatLMVMResetClearsJ0(mode)) PetscCall(MatLMVMClearJ0(B));
  if (MatLMVMResetClearsVecs(mode)) {
    PetscCall(VecDestroy(&lmvm->Xprev));
    PetscCall(VecDestroy(&lmvm->Fprev));
    B->preallocated = PETSC_FALSE; // MatSetUp() needs to be run to create these vecs
  }
  if (MatLMVMResetClearsAll(mode)) {
    lmvm->nupdates = 0;
    lmvm->nrejects = 0;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatLMVMAllocateBases(Mat B)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;

  PetscFunctionBegin;
  PetscCheck(lmvm->Xprev != NULL && lmvm->Fprev != NULL, PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_WRONGSTATE, "Must allocate Xprev and Fprev before allocating bases");
  if (!lmvm->basis[LMBASIS_S]) PetscCall(LMBasisCreate(lmvm->Xprev, lmvm->m, &lmvm->basis[LMBASIS_S]));
  if (!lmvm->basis[LMBASIS_Y]) PetscCall(LMBasisCreate(lmvm->Fprev, lmvm->m, &lmvm->basis[LMBASIS_Y]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatLMVMAllocateVecs(Mat B)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;

  PetscFunctionBegin;
  if (!lmvm->Xprev) PetscCall(MatCreateVecs(B, &lmvm->Xprev, NULL));
  if (!lmvm->Fprev) PetscCall(MatCreateVecs(B, NULL, &lmvm->Fprev));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatAllocate_LMVM(Mat B, Vec X, Vec F)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  PetscBool same;
  VecType   vtype, Bvtype;

  PetscFunctionBegin;
  PetscCall(MatLMVMUseVecLayoutsIfCompatible(B, X, F));
  PetscCall(VecGetType(X, &vtype));
  PetscCall(MatGetVecType(B, &Bvtype));
  PetscCall(PetscStrcmp(vtype, Bvtype, &same));
  if (!same) {
    /* Given X vector has a different type than allocated X-type data structures.
       We need to destroy all of this and duplicate again out of the given vector. */
    PetscCall(MatLMVMReset_Internal(B, MAT_LMVM_RESET_BASES | MAT_LMVM_RESET_VECS));
    PetscCall(MatSetVecType(B, vtype));
    if (lmvm->created_J0) PetscCall(MatSetVecType(lmvm->J0, vtype));
  }
  PetscCall(MatLMVMAllocateVecs(B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatUpdateKernel_LMVM(Mat B, Vec S, Vec Y)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  Vec       s_k, y_k;

  PetscFunctionBegin;
  PetscCall(LMBasisGetNextVec(lmvm->basis[LMBASIS_S], &s_k));
  PetscCall(VecCopy(S, s_k));
  PetscCall(LMBasisRestoreNextVec(lmvm->basis[LMBASIS_S], &s_k));

  PetscCall(LMBasisGetNextVec(lmvm->basis[LMBASIS_Y], &y_k));
  PetscCall(VecCopy(Y, y_k));
  PetscCall(LMBasisRestoreNextVec(lmvm->basis[LMBASIS_Y], &y_k));
  lmvm->nupdates++;
  lmvm->k++;
  PetscAssert(lmvm->k == lmvm->basis[LMBASIS_S]->k, PetscObjectComm((PetscObject)B), PETSC_ERR_PLIB, "Basis S and Mat B out of sync");
  PetscAssert(lmvm->k == lmvm->basis[LMBASIS_Y]->k, PetscObjectComm((PetscObject)B), PETSC_ERR_PLIB, "Basis Y and Mat B out of sync");
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatUpdate_LMVM(Mat B, Vec X, Vec F)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;

  PetscFunctionBegin;
  if (!lmvm->m) PetscFunctionReturn(PETSC_SUCCESS);
  if (lmvm->prev_set) {
    /* Compute the new (S = X - Xprev) and (Y = F - Fprev) vectors */
    PetscCall(VecAXPBY(lmvm->Xprev, 1.0, -1.0, X));
    PetscCall(VecAXPBY(lmvm->Fprev, 1.0, -1.0, F));
    /* Update S and Y */
    PetscCall(MatUpdateKernel_LMVM(B, lmvm->Xprev, lmvm->Fprev));
  }

  /* Save the solution and function to be used in the next update */
  PetscCall(VecCopy(X, lmvm->Xprev));
  PetscCall(VecCopy(F, lmvm->Fprev));
  lmvm->prev_set = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMultAdd_LMVM(Mat B, Vec X, Vec Y, Vec Z)
{
  PetscFunctionBegin;
  PetscCall(MatMult(B, X, Z));
  PetscCall(VecAXPY(Z, 1.0, Y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMult_LMVM(Mat B, Vec X, Vec Y)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;

  PetscFunctionBegin;
  PetscCall((*lmvm->ops->mult)(B, X, Y));
  if (lmvm->shift != 0.0) PetscCall(VecAXPY(Y, lmvm->shift, X));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMultHermitianTranspose_LMVM(Mat B, Vec X, Vec Y)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;

  PetscFunctionBegin;
  PetscCall((*lmvm->ops->multht)(B, X, Y));
  if (lmvm->shift != 0.0) PetscCall(VecAXPY(Y, PetscConj(lmvm->shift), X));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSolve_LMVM(Mat B, Vec x, Vec y)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;

  PetscFunctionBegin;
  PetscCheck(lmvm->shift == 0.0, PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_WRONGSTATE, "Cannot solve a MatLMVM when it has a nonzero shift");
  PetscCall((*lmvm->ops->solve)(B, x, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSolveHermitianTranspose_LMVM(Mat B, Vec x, Vec y)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;

  PetscFunctionBegin;
  PetscCheck(lmvm->shift == 0.0, PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_WRONGSTATE, "Cannot solve a MatLMVM when it has a nonzero shift");
  PetscCall((*lmvm->ops->solveht)(B, x, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSolveTranspose_LMVM(Mat B, Vec x, Vec y)
{
  PetscFunctionBegin;
  if (!PetscDefined(USE_COMPLEX)) {
    PetscCall(MatSolveHermitianTranspose_LMVM(B, x, y));
  } else {
    Vec x_conj;
    PetscCall(VecDuplicate(x, &x_conj));
    PetscCall(VecCopy(x, x_conj));
    PetscCall(VecConjugate(x_conj));
    PetscCall(MatSolveHermitianTranspose_LMVM(B, x_conj, y));
    PetscCall(VecDestroy(&x_conj));
    PetscCall(VecConjugate(y));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// MatCopy() calls MatCheckPreallocated(), so B will have Xprev, Fprev, LMBASIS_S, and LMBASIS_Y
static PetscErrorCode MatCopy_LMVM(Mat B, Mat M, MatStructure str)
{
  Mat_LMVM *bctx = (Mat_LMVM *)B->data;
  Mat_LMVM *mctx;
  Mat       J0_copy;

  PetscFunctionBegin;
  if (str == DIFFERENT_NONZERO_PATTERN) {
    PetscCall(MatLMVMReset(M, PETSC_TRUE));
    PetscCall(MatLMVMAllocate(M, bctx->Xprev, bctx->Fprev));
  } else MatCheckSameSize(B, 1, M, 2);

  mctx = (Mat_LMVM *)M->data;
  PetscCall(MatDuplicate(bctx->J0, MAT_COPY_VALUES, &J0_copy));
  PetscCall(MatLMVMSetJ0(M, J0_copy));
  PetscCall(MatDestroy(&J0_copy));
  mctx->nupdates = bctx->nupdates;
  mctx->nrejects = bctx->nrejects;
  mctx->k        = bctx->k;
  PetscCall(MatLMVMAllocateVecs(M));
  PetscCall(VecCopy(bctx->Xprev, mctx->Xprev));
  PetscCall(VecCopy(bctx->Fprev, mctx->Fprev));
  PetscCall(MatLMVMAllocateBases(M));
  PetscCall(LMBasisCopy(bctx->basis[LMBASIS_S], mctx->basis[LMBASIS_S]));
  PetscCall(LMBasisCopy(bctx->basis[LMBASIS_Y], mctx->basis[LMBASIS_Y]));
  mctx->do_not_cache_J0_products = bctx->do_not_cache_J0_products;
  mctx->cache_gradient_products  = bctx->cache_gradient_products;
  mctx->mult_alg                 = bctx->mult_alg;
  if (mctx->ops->setmultalgorithm) PetscCall((*mctx->ops->setmultalgorithm)(M));
  if (bctx->ops->copy) PetscCall((*bctx->ops->copy)(B, M, str));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDuplicate_LMVM(Mat B, MatDuplicateOption op, Mat *mat)
{
  Mat_LMVM *bctx = (Mat_LMVM *)B->data;
  Mat_LMVM *mctx;
  MatType   lmvmType;
  Mat       A;

  PetscFunctionBegin;
  PetscCall(MatGetType(B, &lmvmType));
  PetscCall(MatCreate(PetscObjectComm((PetscObject)B), mat));
  PetscCall(MatSetType(*mat, lmvmType));

  A       = *mat;
  mctx    = (Mat_LMVM *)A->data;
  mctx->m = bctx->m;
  if (bctx->J0ksp) {
    PetscReal rtol, atol, dtol;
    PetscInt  max_it;

    PetscCall(KSPGetTolerances(bctx->J0ksp, &rtol, &atol, &dtol, &max_it));
    PetscCall(KSPSetTolerances(mctx->J0ksp, rtol, atol, dtol, max_it));
  }
  mctx->shift = bctx->shift;

  PetscCall(MatLMVMAllocate(*mat, bctx->Xprev, bctx->Fprev));
  if (op == MAT_COPY_VALUES) PetscCall(MatCopy(B, *mat, SAME_NONZERO_PATTERN));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatShift_LMVM(Mat B, PetscScalar a)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;

  PetscFunctionBegin;
  lmvm->shift += PetscRealPart(a);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatView_LMVM(Mat B, PetscViewer pv)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  PetscBool isascii;
  MatType   type;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)pv, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscBool         is_exact;
    PetscViewerFormat format;

    PetscCall(MatGetType(B, &type));
    PetscCall(PetscViewerASCIIPrintf(pv, "Max. storage: %" PetscInt_FMT "\n", lmvm->m));
    PetscCall(PetscViewerASCIIPrintf(pv, "Used storage: %" PetscInt_FMT "\n", PetscMin(lmvm->k, lmvm->m)));
    PetscCall(PetscViewerASCIIPrintf(pv, "Number of updates: %" PetscInt_FMT "\n", lmvm->nupdates));
    PetscCall(PetscViewerASCIIPrintf(pv, "Number of rejects: %" PetscInt_FMT "\n", lmvm->nrejects));
    PetscCall(PetscViewerASCIIPrintf(pv, "Number of resets: %" PetscInt_FMT "\n", lmvm->nresets));
    PetscCall(PetscViewerGetFormat(pv, &format));
    if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
      PetscCall(PetscViewerASCIIPrintf(pv, "Mult algorithm: %s\n", MatLMVMMultAlgorithms[lmvm->mult_alg]));
      PetscCall(PetscViewerASCIIPrintf(pv, "Cache J0 products: %s\n", lmvm->do_not_cache_J0_products ? "false" : "true"));
      PetscCall(PetscViewerASCIIPrintf(pv, "Cache gradient products: %s\n", lmvm->cache_gradient_products ? "true" : "false"));
    }
    PetscCall(MatLMVMJ0KSPIsExact(B, &is_exact));
    if (is_exact) {
      PetscBool is_scalar;

      PetscCall(PetscObjectTypeCompare((PetscObject)lmvm->J0, MATCONSTANTDIAGONAL, &is_scalar));
      PetscCall(PetscViewerASCIIPrintf(pv, "J0:\n"));
      PetscCall(PetscViewerASCIIPushTab(pv));
      PetscCall(PetscViewerPushFormat(pv, is_scalar ? PETSC_VIEWER_DEFAULT : PETSC_VIEWER_ASCII_INFO));
      PetscCall(MatView(lmvm->J0, pv));
      PetscCall(PetscViewerPopFormat(pv));
      PetscCall(PetscViewerASCIIPopTab(pv));
    } else {
      PetscCall(PetscViewerASCIIPrintf(pv, "J0 KSP:\n"));
      PetscCall(PetscViewerASCIIPushTab(pv));
      PetscCall(PetscViewerPushFormat(pv, PETSC_VIEWER_ASCII_INFO));
      PetscCall(KSPView(lmvm->J0ksp, pv));
      PetscCall(PetscViewerPopFormat(pv));
      PetscCall(PetscViewerASCIIPopTab(pv));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatSetFromOptions_LMVM(Mat B, PetscOptionItems PetscOptionsObject)
{
  Mat_LMVM            *lmvm     = (Mat_LMVM *)B->data;
  PetscBool            cache_J0 = lmvm->do_not_cache_J0_products ? PETSC_FALSE : PETSC_TRUE; // Default is false, but flipping double negative so that the command line option make sense
  PetscBool            set;
  PetscInt             hist_size = lmvm->m;
  MatLMVMMultAlgorithm mult_alg;

  PetscFunctionBegin;
  PetscCall(MatLMVMGetMultAlgorithm(B, &mult_alg));
  PetscOptionsHeadBegin(PetscOptionsObject, "Limited-memory Variable Metric matrix for approximating Jacobians");
  PetscCall(PetscOptionsInt("-mat_lmvm_hist_size", "number of past updates kept in memory for the approximation", "", hist_size, &hist_size, NULL));
  PetscCall(PetscOptionsEnum("-mat_lmvm_mult_algorithm", "Algorithm used to matrix-vector products", "", MatLMVMMultAlgorithms, (PetscEnum)mult_alg, (PetscEnum *)&mult_alg, &set));
  PetscCall(PetscOptionsReal("-mat_lmvm_eps", "(developer) machine zero definition", "", lmvm->eps, &lmvm->eps, NULL));
  PetscCall(PetscOptionsBool("-mat_lmvm_cache_J0_products", "Cache applications of the kernel J0 or its inverse", "", cache_J0, &cache_J0, NULL));
  PetscCall(PetscOptionsBool("-mat_lmvm_cache_gradient_products", "Cache data used to apply the inverse Hessian to a gradient vector to accelerate the quasi-Newton update", "", lmvm->cache_gradient_products, &lmvm->cache_gradient_products, NULL));
  PetscCall(PetscOptionsBool("-mat_lmvm_debug", "(developer) Perform internal debugging checks", "", lmvm->debug, &lmvm->debug, NULL));
  PetscOptionsHeadEnd();
  lmvm->do_not_cache_J0_products = cache_J0 ? PETSC_FALSE : PETSC_TRUE;
  if (hist_size != lmvm->m) PetscCall(MatLMVMSetHistorySize(B, hist_size));
  if (set) PetscCall(MatLMVMSetMultAlgorithm(B, mult_alg));
  if (lmvm->created_J0) PetscCall(MatSetFromOptions(lmvm->J0));
  if (lmvm->created_J0ksp) PetscCall(KSPSetFromOptions(lmvm->J0ksp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatSetUp_LMVM(Mat B)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;

  PetscFunctionBegin;
  PetscCall(PetscLayoutSetUp(B->rmap));
  PetscCall(PetscLayoutSetUp(B->cmap));
  if (lmvm->created_J0) {
    PetscCall(PetscLayoutReference(B->rmap, &lmvm->J0->rmap));
    PetscCall(PetscLayoutReference(B->cmap, &lmvm->J0->cmap));
    PetscCall(MatSetUp(lmvm->J0));
  }
  PetscCall(MatLMVMAllocateVecs(B));
  PetscCall(MatLMVMAllocateBases(B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatLMVMSetMultAlgorithm - Set the algorithm used by a `MatLMVM` for products

  Logically collective

  Input Parameters:
+ B   - a `MatLMVM` matrix
- alg - one of the algorithm classes (`MAT_LMVM_MULT_RECURSIVE`, `MAT_LMVM_MULT_DENSE`, `MAT_LMVM_MULT_COMPACT_DENSE`)

  Level: advanced

.seealso: [](ch_matrices), `MatLMVM`, `MatLMVMMultAlgorithm`, `MatLMVMGetMultAlgorithm()`
@*/
PetscErrorCode MatLMVMSetMultAlgorithm(Mat B, MatLMVMMultAlgorithm alg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  PetscTryMethod(B, "MatLMVMSetMultAlgorithm_C", (Mat, MatLMVMMultAlgorithm), (B, alg));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatLMVMSetMultAlgorithm_LMVM(Mat B, MatLMVMMultAlgorithm alg)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;

  PetscFunctionBegin;
  lmvm->mult_alg = alg;
  if (lmvm->ops->setmultalgorithm) PetscCall((*lmvm->ops->setmultalgorithm)(B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatLMVMGetMultAlgorithm - Get the algorithm used by a `MatLMVM` for products

  Not collective

  Input Parameter:
. B - a `MatLMVM` matrix

  Output Parameter:
. alg - one of the algorithm classes (`MAT_LMVM_MULT_RECURSIVE`, `MAT_LMVM_MULT_DENSE`, `MAT_LMVM_MULT_COMPACT_DENSE`)

  Level: advanced

.seealso: [](ch_matrices), `MatLMVM`, `MatLMVMMultAlgorithm`, `MatLMVMSetMultAlgorithm()`
@*/
PetscErrorCode MatLMVMGetMultAlgorithm(Mat B, MatLMVMMultAlgorithm *alg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  PetscAssertPointer(alg, 2);
  PetscUseMethod(B, "MatLMVMGetMultAlgorithm_C", (Mat, MatLMVMMultAlgorithm *), (B, alg));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatLMVMGetMultAlgorithm_LMVM(Mat B, MatLMVMMultAlgorithm *alg)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;

  PetscFunctionBegin;
  *alg = lmvm->mult_alg;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatDestroy_LMVM(Mat B)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;

  PetscFunctionBegin;
  PetscCall(MatReset_LMVM(B, MAT_LMVM_RESET_ALL));
  PetscCall(KSPDestroy(&lmvm->J0ksp));
  PetscCall(MatDestroy(&lmvm->J0));
  PetscCall(PetscFree(B->data));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatLMVMGetLastUpdate_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatLMVMSetMultAlgorithm_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatLMVMGetMultAlgorithm_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatSetOptionsPrefix_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatAppendOptionsPrefix_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatLMVMGetLastUpdate - Get the last vectors passed to `MatLMVMUpdate()`

  Not collective

  Input Parameter:
. B - a `MatLMVM` matrix

  Output Parameters:
+ x_prev - the last solution vector
- f_prev - the last function vector

  Level: intermediate

.seealso: [](ch_matrices), `MatLMVM`, `MatLMVMUpdate()`
@*/
PetscErrorCode MatLMVMGetLastUpdate(Mat B, Vec *x_prev, Vec *f_prev)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  PetscTryMethod(B, "MatLMVMGetLastUpdate_C", (Mat, Vec *, Vec *), (B, x_prev, f_prev));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatLMVMGetLastUpdate_LMVM(Mat B, Vec *x_prev, Vec *f_prev)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;

  PetscFunctionBegin;
  if (x_prev) *x_prev = (lmvm->prev_set) ? lmvm->Xprev : NULL;
  if (f_prev) *f_prev = (lmvm->prev_set) ? lmvm->Fprev : NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* in both MatSetOptionsPrefix() and MatAppendOptionsPrefix(), this is called after
   the prefix of B has been changed, so we just query the prefix of B rather than
   using the passed prefix */
static PetscErrorCode MatSetOptionsPrefix_LMVM(Mat B, const char unused[])
{
  Mat_LMVM   *lmvm = (Mat_LMVM *)B->data;
  const char *prefix;

  PetscFunctionBegin;
  PetscCall(MatGetOptionsPrefix(B, &prefix));
  if (lmvm->created_J0) {
    PetscCall(MatSetOptionsPrefix(lmvm->J0, prefix));
    PetscCall(MatAppendOptionsPrefix(lmvm->J0, "mat_lmvm_J0_"));
  }
  if (lmvm->created_J0ksp) {
    PetscCall(KSPSetOptionsPrefix(lmvm->J0ksp, prefix));
    PetscCall(KSPAppendOptionsPrefix(lmvm->J0ksp, "mat_lmvm_J0_"));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   MATLMVM - MATLMVM = "lmvm" - A matrix type used for Limited-Memory Variable Metric (LMVM) matrices.

   Level: intermediate

   Developer notes:
   Improve this manual page as well as many others in the MATLMVM family.

.seealso: [](sec_matlmvm), `Mat`
M*/
PetscErrorCode MatCreate_LMVM(Mat B)
{
  Mat_LMVM *lmvm;

  PetscFunctionBegin;
  PetscCall(MatLMVMPackageInitialize());
  PetscCall(PetscNew(&lmvm));
  B->data = (void *)lmvm;

  lmvm->m   = 5;
  lmvm->eps = PetscPowReal(PETSC_MACHINE_EPSILON, 2.0 / 3.0);

  B->ops->destroy                = MatDestroy_LMVM;
  B->ops->setfromoptions         = MatSetFromOptions_LMVM;
  B->ops->view                   = MatView_LMVM;
  B->ops->setup                  = MatSetUp_LMVM;
  B->ops->shift                  = MatShift_LMVM;
  B->ops->duplicate              = MatDuplicate_LMVM;
  B->ops->mult                   = MatMult_LMVM;
  B->ops->multhermitiantranspose = MatMultHermitianTranspose_LMVM;
  B->ops->multadd                = MatMultAdd_LMVM;
  B->ops->copy                   = MatCopy_LMVM;
  B->ops->solve                  = MatSolve_LMVM;
  B->ops->solvetranspose         = MatSolveTranspose_LMVM;
  if (!PetscDefined(USE_COMPLEX)) B->ops->multtranspose = MatMultHermitianTranspose_LMVM;

  /*
    There is no assembly phase, Mat_LMVM relies on B->preallocated to ensure that
    necessary setup happens in MatSetUp(), which is called in MatCheckPreallocated()
    in all major operations (MatLMVMUpdate(), MatMult(), MatSolve(), etc.)
   */
  B->assembled = PETSC_TRUE;

  lmvm->ops->update = MatUpdate_LMVM;

  PetscCall(PetscObjectChangeTypeName((PetscObject)B, MATLMVM));
  // J0 should be present at all times, calling ClearJ0() here initializes it to the identity
  PetscCall(MatLMVMClearJ0(B));

  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatLMVMGetLastUpdate_C", MatLMVMGetLastUpdate_LMVM));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatLMVMSetMultAlgorithm_C", MatLMVMSetMultAlgorithm_LMVM));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatLMVMGetMultAlgorithm_C", MatLMVMGetMultAlgorithm_LMVM));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatSetOptionsPrefix_C", MatSetOptionsPrefix_LMVM));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatAppendOptionsPrefix_C", MatSetOptionsPrefix_LMVM));
  PetscFunctionReturn(PETSC_SUCCESS);
}

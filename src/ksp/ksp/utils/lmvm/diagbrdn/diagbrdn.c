#include <petscdevice.h>
#include <../src/ksp/ksp/utils/lmvm/rescale/symbrdnrescale.h> /*I "petscksp.h" I*/

static PetscErrorCode MatSolve_DiagBrdn(Mat B, Vec F, Vec dX)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;

  PetscFunctionBegin;
  PetscCall(MatSolve(lmvm->J0, F, dX));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMult_DiagBrdn(Mat B, Vec X, Vec Z)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;

  PetscFunctionBegin;
  PetscCall(MatMult(lmvm->J0, X, Z));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatUpdate_DiagBrdn(Mat B, Vec X, Vec F)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;

  PetscFunctionBegin;
  if (!lmvm->m) PetscFunctionReturn(PETSC_SUCCESS);
  if (lmvm->prev_set) {
    SymBroydenRescale ldb = (SymBroydenRescale)lmvm->ctx;
    PetscScalar       curvature;
    PetscReal         curvtol, ststmp;
    PetscInt          oldest, next;

    PetscCall(MatLMVMGetRange(B, &oldest, &next));
    /* Compute the new (S = X - Xprev) and (Y = F - Fprev) vectors */
    PetscCall(VecAYPX(lmvm->Xprev, -1.0, X));
    PetscCall(VecAYPX(lmvm->Fprev, -1.0, F));

    /* Test if the updates can be accepted */
    PetscCall(VecDotNorm2(lmvm->Fprev, lmvm->Xprev, &curvature, &ststmp));
    if (ststmp < lmvm->eps) curvtol = 0.0;
    else curvtol = lmvm->eps * ststmp;

    /* Test the curvature for the update */
    if (PetscRealPart(curvature) > curvtol) {
      /* Update is good so we accept it */
      PetscCall(MatUpdateKernel_LMVM(B, lmvm->Xprev, lmvm->Fprev));
      PetscCall(MatLMVMProductsInsertDiagonalValue(B, LMBASIS_Y, LMBASIS_S, next, PetscRealPart(curvature)));
      PetscCall(MatLMVMProductsInsertDiagonalValue(B, LMBASIS_S, LMBASIS_S, next, ststmp));
      PetscCall(SymBroydenRescaleUpdate(B, ldb));
    } else {
      /* reset */
      PetscCall(SymBroydenRescaleReset(B, ldb, MAT_LMVM_RESET_HISTORY));
    }
    /* End DiagBrdn update */
  }
  /* Save the solution and function to be used in the next update */
  PetscCall(VecCopy(X, lmvm->Xprev));
  PetscCall(VecCopy(F, lmvm->Fprev));
  lmvm->prev_set = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatCopy_DiagBrdn(Mat B, Mat M, MatStructure str)
{
  Mat_LMVM         *bdata = (Mat_LMVM *)B->data;
  SymBroydenRescale bctx  = (SymBroydenRescale)bdata->ctx;
  Mat_LMVM         *mdata = (Mat_LMVM *)M->data;
  SymBroydenRescale mctx  = (SymBroydenRescale)mdata->ctx;

  PetscFunctionBegin;
  PetscCall(SymBroydenRescaleCopy(bctx, mctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatView_DiagBrdn(Mat B, PetscViewer pv)
{
  Mat_LMVM         *lmvm = (Mat_LMVM *)B->data;
  SymBroydenRescale ldb  = (SymBroydenRescale)lmvm->ctx;

  PetscFunctionBegin;
  PetscCall(MatView_LMVM(B, pv));
  PetscCall(SymBroydenRescaleView(ldb, pv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetFromOptions_DiagBrdn(Mat B, PetscOptionItems PetscOptionsObject)
{
  Mat_LMVM         *lmvm = (Mat_LMVM *)B->data;
  SymBroydenRescale ldb  = (SymBroydenRescale)lmvm->ctx;

  PetscFunctionBegin;
  PetscCall(MatSetFromOptions_LMVM(B, PetscOptionsObject));
  PetscCall(SymBroydenRescaleSetFromOptions(B, ldb, PetscOptionsObject));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatReset_DiagBrdn(Mat B, MatLMVMResetMode mode)
{
  Mat_LMVM         *lmvm = (Mat_LMVM *)B->data;
  SymBroydenRescale ldb  = (SymBroydenRescale)lmvm->ctx;

  PetscFunctionBegin;
  PetscCall(SymBroydenRescaleReset(B, ldb, mode));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDestroy_DiagBrdn(Mat B)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;

  PetscFunctionBegin;
  PetscCall(SymBroydenRescaleDestroy((SymBroydenRescale *)&lmvm->ctx));
  PetscCall(MatDestroy_LMVM(B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetUp_DiagBrdn(Mat B)
{
  Mat_LMVM         *lmvm = (Mat_LMVM *)B->data;
  SymBroydenRescale ldb  = (SymBroydenRescale)lmvm->ctx;

  PetscFunctionBegin;
  PetscCall(MatSetUp_LMVM(B));
  PetscCall(SymBroydenRescaleInitializeJ0(B, ldb));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatCreate_LMVMDiagBrdn(Mat B)
{
  Mat_LMVM         *lmvm;
  SymBroydenRescale ldb;

  PetscFunctionBegin;
  PetscCall(MatCreate_LMVM(B));
  PetscCall(PetscObjectChangeTypeName((PetscObject)B, MATLMVMDIAGBROYDEN));
  PetscCall(MatSetOption(B, MAT_HERMITIAN, PETSC_TRUE));
  PetscCall(MatSetOption(B, MAT_SPD, PETSC_TRUE));
  PetscCall(MatSetOption(B, MAT_SPD_ETERNAL, PETSC_TRUE));
  B->ops->setup          = MatSetUp_DiagBrdn;
  B->ops->setfromoptions = MatSetFromOptions_DiagBrdn;
  B->ops->destroy        = MatDestroy_DiagBrdn;
  B->ops->view           = MatView_DiagBrdn;

  lmvm              = (Mat_LMVM *)B->data;
  lmvm->ops->reset  = MatReset_DiagBrdn;
  lmvm->ops->mult   = MatMult_DiagBrdn;
  lmvm->ops->solve  = MatSolve_DiagBrdn;
  lmvm->ops->update = MatUpdate_DiagBrdn;
  lmvm->ops->copy   = MatCopy_DiagBrdn;

  PetscCall(SymBroydenRescaleCreate(&ldb));
  lmvm->ctx = (void *)ldb;

  PetscCall(MatLMVMSetHistorySize(B, 1));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatCreateLMVMDiagBroyden - DiagBrdn creates a symmetric Broyden-type diagonal matrix used
  for approximating Hessians.

  Collective

  Input Parameters:
+ comm - MPI communicator
. n    - number of local rows for storage vectors
- N    - global size of the storage vectors

  Output Parameter:
. B - the matrix

  Options Database Keys:
+ -mat_lmvm_theta      - (developer) convex ratio between BFGS and DFP components of the diagonal J0 scaling
. -mat_lmvm_rho        - (developer) update limiter for the J0 scaling
. -mat_lmvm_alpha      - (developer) coefficient factor for the quadratic subproblem in J0 scaling
. -mat_lmvm_beta       - (developer) exponential factor for the diagonal J0 scaling
. -mat_lmvm_sigma_hist - (developer) number of past updates to use in J0 scaling.
. -mat_lmvm_tol        - (developer) tolerance for bounding the denominator of the rescaling away from 0.
- -mat_lmvm_forward    - (developer) whether or not to use the forward or backward Broyden update to the diagonal

  Level: intermediate

  Notes:
  It is recommended that one use the `MatCreate()`, `MatSetType()` and/or `MatSetFromOptions()`
  paradigm instead of this routine directly.

  It consists of a convex combination of DFP and BFGS
  diagonal approximation schemes, such that DiagBrdn = (1-theta)*BFGS + theta*DFP.
  To preserve symmetric positive-definiteness, we restrict theta to be in [0, 1].
  We also ensure positive definiteness by taking the `VecAbs()` of the final vector.

  There are two ways of approximating the diagonal: using the forward (B) update
  schemes for BFGS and DFP and then taking the inverse, or directly working with
  the inverse (H) update schemes for the BFGS and DFP updates, derived using the
  Sherman-Morrison-Woodbury formula. We have implemented both, controlled by a
  parameter below.

  In order to use the DiagBrdn matrix with other vector types, i.e. doing matrix-vector products
  and matrix solves, the matrix must first be created using `MatCreate()` and `MatSetType()`,
  followed by `MatLMVMAllocate()`. Then it will be available for updating
  (via `MatLMVMUpdate()`) in one's favored solver implementation.

.seealso: [](ch_ksp), `MatCreate()`, `MATLMVM`, `MATLMVMDIAGBRDN`, `MatCreateLMVMDFP()`, `MatCreateLMVMSR1()`,
          `MatCreateLMVMBFGS()`, `MatCreateLMVMBroyden()`, `MatCreateLMVMSymBroyden()`
@*/
PetscErrorCode MatCreateLMVMDiagBroyden(MPI_Comm comm, PetscInt n, PetscInt N, Mat *B)
{
  PetscFunctionBegin;
  PetscCall(KSPInitializePackage());
  PetscCall(MatCreate(comm, B));
  PetscCall(MatSetSizes(*B, n, n, N, N));
  PetscCall(MatSetType(*B, MATLMVMDIAGBROYDEN));
  PetscCall(MatSetUp(*B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

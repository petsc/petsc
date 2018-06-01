#include <../src/ksp/ksp/utils/lmvm/lmvm.h> /*I "petscksp.h" I*/

/*
  Zero-memory Symmetric Broyden method for explicitly approximating 
  the diagonal of a Jacobian.
*/

typedef struct {
  Vec D, y2, Ds2, w2;
  PetscBool allocated;
  PetscReal phi;
} Mat_DiagBrdn;

/*------------------------------------------------------------*/

static PetscErrorCode MatSolve_LMVMDiagBrdn(Mat B, Vec F, Vec dX)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_DiagBrdn      *ldb = (Mat_DiagBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(F, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(dX, VEC_CLASSID, 3);
  VecCheckSameSize(F, 2, dX, 3);
  if (!lmvm->allocated) {
    ierr = MatLMVMAllocate(B, dX, F);CHKERRQ(ierr);
  } else {
    VecCheckMatCompatible(B, dX, 3, F, 2);
  }
  ierr = VecPointwiseDivide(dX, F, ldb->D);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatMult_LMVMDiagBrdn(Mat B, Vec X, Vec Z)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_DiagBrdn      *ldb = (Mat_DiagBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(X, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(Z, VEC_CLASSID, 3);
  VecCheckSameSize(X, 2, Z, 3);
  if (!lmvm->allocated) {
    ierr = MatLMVMAllocate(B, X, Z);CHKERRQ(ierr);
  } else {
    VecCheckMatCompatible(B, X, 2, Z, 3);
  }
  ierr = VecPointwiseMult(Z, X, ldb->D);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/*
  The diagonal Hessian update is derived from Equation 2 in 
  Erway and Marcia "On Solving Large-scale Limited-Memory 
  Quasi-Newton Equations" (https://arxiv.org/pdf/1510.06378.pdf).
  In this "zero"-memory implementation, the matrix-matrix products 
  are replaced by pointwise multiplications between their diagonal 
  vectors. Unlike limited-memory methods, the incoming updates 
  are directly applied to the diagonal instead of being stored 
  for later use.
*/
static PetscErrorCode MatUpdate_LMVMDiagBrdn(Mat B, Vec X, Vec F)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_DiagBrdn      *ldb = (Mat_DiagBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  PetscReal         yts, stDs;

  PetscFunctionBegin;
  if (lmvm->prev_set) {
    /* Compute the new (S = X - Xprev) and (Y = F - Fprev) vectors */
    ierr = VecAXPBY(lmvm->Xprev, 1.0, -1.0, X);CHKERRQ(ierr);
    ierr = VecAXPBY(lmvm->Fprev, 1.0, -1.0, F);CHKERRQ(ierr);
    /* Test if the updates can be accepted */
    ierr = VecDot(lmvm->Fprev, lmvm->Xprev, &yts);CHKERRQ(ierr);
    if (yts > -lmvm->eps) {
      /* Update is good, accept it */
      ++lmvm->nupdates;
      /* Compute D * s once here because we need it for dot products */
      ierr = VecPointwiseMult(ldb->Ds2, ldb->D, lmvm->Xprev);CHKERRQ(ierr);
      /* Compute dot product s^T Ds for later */
      ierr = VecDot(lmvm->Xprev, ldb->Ds2, &stDs);CHKERRQ(ierr);
      /* Compute W = y/(y^T s) - Ds/(s^T Ds) */
      ierr = VecAXPBYPCZ(ldb->w2, 1.0/yts, -1.0/stDs, 0.0, lmvm->Fprev, ldb->Ds2);CHKERRQ(ierr);
      /* Compute W*W^T diagonal (Hadamard product) */
      ierr = VecPow(ldb->w2, 2.0);CHKERRQ(ierr);
      /* Compute (Ds)*(Ds)^T diagonal (Hadamard product) */
      ierr = VecPow(ldb->Ds2, 2.0);CHKERRQ(ierr);
      /* Compute Y*Y^T diagonal (Hadamard product) */
      ierr = VecPointwiseMult(ldb->y2, lmvm->Fprev, lmvm->Fprev);CHKERRQ(ierr);
      /* Apply the BFGS part of the update */
      ierr = VecAXPBYPCZ(ldb->D, -1.0/stDs, 1.0/yts, 1.0, ldb->Ds2, ldb->y2);CHKERRQ(ierr);
      /* Now apply the convexly scaled DFP component */ 
      ierr = VecAXPY(ldb->D, ldb->phi*stDs, ldb->w2);CHKERRQ(ierr);
    } else {
      /* Update is bad, skip it */
      ++lmvm->nrejects;
    }
  } else {
    ierr = VecSet(ldb->D, 1.0);CHKERRQ(ierr);
  }
  /* Save the solution and function to be used in the next update */
  ierr = VecCopy(X, lmvm->Xprev);CHKERRQ(ierr);
  ierr = VecCopy(F, lmvm->Fprev);CHKERRQ(ierr);
  lmvm->prev_set = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatCopy_LMVMDiagBrdn(Mat B, Mat M, MatStructure str)
{
  Mat_LMVM          *bdata = (Mat_LMVM*)B->data;
  Mat_DiagBrdn      *bctx = (Mat_DiagBrdn*)bdata->ctx;
  Mat_LMVM          *mdata = (Mat_LMVM*)M->data;
  Mat_DiagBrdn      *mctx = (Mat_DiagBrdn*)mdata->ctx;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  mctx->phi = bctx->phi;
  ierr = VecCopy(bctx->D, mctx->D);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatReset_LMVMDiagBrdn(Mat B, PetscBool destructive)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_DiagBrdn      *ldb = (Mat_DiagBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  if (destructive && ldb->allocated) {
    ierr = VecDestroy(&ldb->D);CHKERRQ(ierr);
    ierr = VecDestroy(&ldb->y2);CHKERRQ(ierr);
    ierr = VecDestroy(&ldb->Ds2);CHKERRQ(ierr);
    ierr = VecDestroy(&ldb->w2);CHKERRQ(ierr);
    ldb->allocated = PETSC_FALSE;
  } else {
    ierr = VecSet(ldb->D, 1.0);CHKERRQ(ierr);
  }
  ierr = MatReset_LMVM(B, destructive);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatAllocate_LMVMDiagBrdn(Mat B, Vec X, Vec F)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_DiagBrdn      *ldb = (Mat_DiagBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  lmvm->m = 0;
  ierr = MatAllocate_LMVM(B, X, F);CHKERRQ(ierr);
  if (!ldb->allocated) {
    ierr = VecDuplicate(X, &ldb->D);CHKERRQ(ierr);
    ierr = VecDuplicate(X, &ldb->y2);CHKERRQ(ierr);
    ierr = VecDuplicate(X, &ldb->Ds2);CHKERRQ(ierr);
    ierr = VecDuplicate(X, &ldb->w2);CHKERRQ(ierr);
    ldb->allocated = PETSC_TRUE;
  }
  ierr = VecSet(ldb->D, 1.0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatDestroy_LMVMDiagBrdn(Mat B)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_DiagBrdn      *ldb = (Mat_DiagBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (ldb->allocated) {
    ierr = VecDestroy(&ldb->D);CHKERRQ(ierr);
    ierr = VecDestroy(&ldb->y2);CHKERRQ(ierr);
    ierr = VecDestroy(&ldb->Ds2);CHKERRQ(ierr);
    ierr = VecDestroy(&ldb->w2);CHKERRQ(ierr);
    ldb->allocated = PETSC_FALSE;
  }
  ierr = PetscFree(lmvm->ctx);CHKERRQ(ierr);
  ierr = MatDestroy_LMVM(B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatSetUp_LMVMDiagBrdn(Mat B)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_DiagBrdn      *ldb = (Mat_DiagBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  lmvm->m = 0;
  ierr = MatSetUp_LMVM(B);CHKERRQ(ierr);
  if (!ldb->allocated) {
    ierr = VecDuplicate(lmvm->Xprev, &ldb->D);CHKERRQ(ierr);
    ierr = VecDuplicate(lmvm->Xprev, &ldb->y2);CHKERRQ(ierr);
    ierr = VecDuplicate(lmvm->Xprev, &ldb->Ds2);CHKERRQ(ierr);
    ierr = VecDuplicate(lmvm->Xprev, &ldb->w2);CHKERRQ(ierr);
    ldb->allocated = PETSC_TRUE;
  }
  ierr = VecSet(ldb->D, 1.0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatSetFromOptions_LMVMDiagBrdn(PetscOptionItems *PetscOptionsObject, Mat B)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_DiagBrdn      *ldb = (Mat_DiagBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatSetFromOptions_LMVM(PetscOptionsObject, B);CHKERRQ(ierr);
  ierr = PetscOptionsHead(PetscOptionsObject,"Limited-memory Variable Metric matrix for approximating Jacobians");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_lmvm_phi","(developer) convex ratio between BFGS and DFP components in the Broyden update","",ldb->phi,&ldb->phi,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  if ((ldb->phi < 0.0) || (ldb->phi > 1.0)) SETERRQ(PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_OUTOFRANGE, "convex ratio cannot be outside the range of [0, 1]");
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode MatCreate_LMVMDiagBrdn(Mat B)
{
  Mat_DiagBrdn       *ldb;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatCreate_LMVM(B);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B, MATLMVMDIAGBRDN);CHKERRQ(ierr);
  ierr = MatSetOption(B, MAT_SPD, PETSC_TRUE);CHKERRQ(ierr);
  
  B->ops->solve = MatSolve_LMVMDiagBrdn;
  B->ops->setfromoptions = MatSetFromOptions_LMVMDiagBrdn;
  B->ops->setup = MatSetUp_LMVMDiagBrdn;
  B->ops->destroy = MatDestroy_LMVMDiagBrdn;
  
  Mat_LMVM *lmvm = (Mat_LMVM*)B->data;
  lmvm->m = -1;
  lmvm->square = PETSC_TRUE;
  lmvm->ops->update = MatUpdate_LMVMDiagBrdn;
  lmvm->ops->allocate = MatAllocate_LMVMDiagBrdn;
  lmvm->ops->reset = MatReset_LMVMDiagBrdn;
  lmvm->ops->mult = MatMult_LMVMDiagBrdn;
  lmvm->ops->copy = MatCopy_LMVMDiagBrdn;
  
  ierr = PetscNewLog(B, &ldb);CHKERRQ(ierr);
  lmvm->ctx = (void*)ldb;
  ldb->allocated = PETSC_FALSE;
  ldb->phi = 0.125;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

/*@
   MatCreateLMVMDiagBrdn - Creates a zero-memory symmetric Broyden approximation 
   for the diagonal of a Jacobian. This matrix does not store any LMVM update vectors, 
   and instead uses the full-memory symmetric Broyden formula to update a vector that 
   respresents the diagonal of a Jacobian.

   Collective on MPI_Comm

   Input Parameters:
+  comm - MPI communicator, set to PETSC_COMM_SELF
.  n - number of local rows for storage vectors
-  N - global size of the storage vectors

   Output Parameter:
.  B - the matrix

   It is recommended that one use the MatCreate(), MatSetType() and/or MatSetFromOptions()
   paradigm instead of this routine directly.

   Options Database Keys:
.   -mat_lmvm_phi - (developer) convex ratio between BFGS and DFP components of the inverse

   Level: intermediate

.seealso: MatCreate(), MATLMVM, MATLMVMSYMBRDN, MatCreateLMVMDFP(), MatCreateLMVMSR1(), 
          MatCreateLMVMBFGS(), MatCreateLMVMBrdn(), MatCreateLMVMBadBrdn()
@*/
PetscErrorCode MatCreateLMVMDiagBrdn(MPI_Comm comm, PetscInt n, PetscInt N, Mat *B)
{
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  ierr = MatCreate(comm, B);CHKERRQ(ierr);
  ierr = MatSetSizes(*B, n, n, N, N);CHKERRQ(ierr);
  ierr = MatSetType(*B, MATLMVMDIAGBRDN);CHKERRQ(ierr);
  ierr = MatSetUp(*B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
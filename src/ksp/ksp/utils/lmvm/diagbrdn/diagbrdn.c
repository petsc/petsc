#include <../src/ksp/ksp/utils/lmvm/lmvm.h> /*I "petscksp.h" I*/

/*
  Zero-memory Symmetric Broyden method for explicitly approximating 
  the diagonal of a Jacobian.
*/

typedef struct {
  Vec D, BssTB, yyT, wwT;
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
  VecCheckMatCompatible(B, dX, 3, F, 2);
  ierr = VecPointwiseDivide(dX, F, ldb->D);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode MatMult_LMVMDiagBrdn(Mat B, Vec F, Vec dX)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  Mat_DiagBrdn      *ldb = (Mat_DiagBrdn*)lmvm->ctx;
  PetscErrorCode    ierr;
  
  PetscFunctionBegin;
  ierr = VecPointwiseMult(dX, ldb->D, F);CHKERRQ(ierr);
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
  PetscReal         yts, stbs;

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
      /* Compute B_k * s_k once here because we need it for dot products */
      ierr = VecPointwiseMult(ldb->wwT, ldb->D, lmvm->Xprev);CHKERRQ(ierr);
      /* Compute W = y_k/yTs - B_k*s_k/stBs */
      ierr = VecDot(lmvm->Xprev, ldb->wwT, &stbs);CHKERRQ(ierr);
      ierr = VecAXPBY(ldb->wwT, 1.0/yts, -1.0/stbs, lmvm->Fprev);CHKERRQ(ierr);
      /* Now compute diagonal of W*W^T */
      ierr = VecPow(ldb->wwT, 2.0);CHKERRQ(ierr);
      /* Compute S*S^T diagonal */
      ierr = VecPointwiseMult(ldb->BssTB, lmvm->Xprev, lmvm->Xprev);CHKERRQ(ierr);
      /* Hit it with B_k on both sides */
      ierr = VecPointwiseMult(ldb->BssTB, ldb->BssTB, ldb->D);CHKERRQ(ierr);
      ierr = VecPointwiseMult(ldb->BssTB, ldb->D, ldb->BssTB);CHKERRQ(ierr);
      /* Compute Y*Y^T */
      ierr = VecPointwiseMult(ldb->yyT, lmvm->Fprev, lmvm->Fprev);CHKERRQ(ierr);
      /* Update the diagonal */
      ierr = VecAXPBYPCZ(ldb->D, -1.0/stbs, 1.0/yts, 1.0, ldb->BssTB, ldb->yyT);CHKERRQ(ierr);
      ierr = VecAXPY(ldb->D, ldb->phi*stbs, ldb->wwT);CHKERRQ(ierr);
    } else {
      /* Update is bad, skip it */
      ++lmvm->nrejects;
    }
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
  PetscInt          i;

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
    ierr = VecDestroy(&ldb->BssTB);CHKERRQ(ierr);
    ierr = VecDestroy(&ldb->yyT);CHKERRQ(ierr);
    ierr = VecDestroy(&ldb->wwT);CHKERRQ(ierr);
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
    ierr = VecDuplicate(X, &ldb->BssTB);CHKERRQ(ierr);
    ierr = VecDuplicate(X, &ldb->yyT);CHKERRQ(ierr);
    ierr = VecDuplicate(X, &ldb->wwT);CHKERRQ(ierr);
    ldb->allocated = PETSC_TRUE;
  }
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
    ierr = VecDestroy(&ldb->BssTB);CHKERRQ(ierr);
    ierr = VecDestroy(&ldb->yyT);CHKERRQ(ierr);
    ierr = VecDestroy(&ldb->wwT);CHKERRQ(ierr);
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
    ierr = VecDuplicate(lmvm->Xprev, &ldb->BssTB);CHKERRQ(ierr);
    ierr = VecDuplicate(lmvm->Xprev, &ldb->yyT);CHKERRQ(ierr);
    ierr = VecDuplicate(lmvm->Xprev, &ldb->wwT);CHKERRQ(ierr);
    ldb->allocated = PETSC_TRUE;
  }
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
  ierr = PetscObjectChangeTypeName((PetscObject)B, MATLMVMSYMBRDN);CHKERRQ(ierr);
  ierr = MatSetOption(B, MAT_SPD, PETSC_TRUE);CHKERRQ(ierr);
  
  B->ops->solve = MatSolve_LMVMDiagBrdn;
  B->ops->setfromoptions = MatSetFromOptions_LMVMDiagBrdn;
  B->ops->setup = MatSetUp_LMVMDiagBrdn;
  B->ops->destroy = MatDestroy_LMVMDiagBrdn;
  
  Mat_LMVM *lmvm = (Mat_LMVM*)B->data;
  lmvm->m = 0;
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
#include <../src/mat/impls/lmvm/lmvm.h>

PetscErrorCode MatUpdate_LMVM(Mat B, Vec X, Vec F)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  PetscErrorCode    ierr;
  PetscBool         same;
  PetscInt          i;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(X, VEC_CLASSID, 1);
  PetscValidHeaderSpecific(F, VEC_CLASSID, 2);
  if (lmvm->k == 0) {
    ierr = VecCopy(X, lmvm->Xprev);CHKERRQ(ierr);
    ierr = VecCopy(F, lmvm->Fprev);CHKERRQ(ierr);
    ++lmvm->k;
  } else {
    if (lmvm->k == lmvm->m) {
      ierr=PetscObjectDereference((PetscObject)lmvm->S[0]);CHKERRQ(ierr);
      ierr=PetscObjectDereference((PetscObject)lmvm->Y[0]);CHKERRQ(ierr);
      for (i = lmvm->k-1; i >= 0; --i) {
        lmvm->S[i+1] = ctx->S[i];
        lmvm->Y[i+1] = ctx->Y[i];
        ctx->rho[i+1] = ctx->rho[i];
      }
      lmvm->S[0] = lmvm->Xprev;
      lmvm->Y[0] = lmvm->Gprev;
      PetscObjectReference((PetscObject)ctx->S[0]);
      PetscObjectReference((PetscObject)ctx->Y[0]);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetH0_LMVM(Mat B, Mat H0)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetH0KSP_LMVM(Mat B, KSP ksp)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

PetscErrorCode MatView_LMVM(Mat B)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetUp_LMVM(Mat B)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_LMVM(Mat B)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

PetscErrorCode MatCreate_LMVM(Mat B)
{
  Mat_LMVM          *lmvm = (Mat_LMVM*)B->data;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
#ifndef _PETSC_COMPAT_MAT_H
#define _PETSC_COMPAT_MAT_H

#include "private/matimpl.h"

#define MAT_KEEP_NONZERO_PATTERN MAT_KEEP_ZEROED_ROWS

#undef __FUNCT__
#define __FUNCT__ "MatGetDiagonalBlock_300"
static PETSC_UNUSED
PetscErrorCode MatGetDiagonalBlock_300(Mat A,PetscTruth *iscopy,MatReuse reuse,Mat *a)
{
  PetscErrorCode ierr,(*f)(Mat,PetscTruth*,MatReuse,Mat*);
  PetscMPIInt    size;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_COOKIE,1);
  PetscValidPointer(iscopy,2);
  PetscValidPointer(a,3);
  PetscValidType(A,1);
  if (!A->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (A->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  ierr = MPI_Comm_size(((PetscObject)A)->comm,&size);CHKERRQ(ierr);
  ierr = PetscObjectQueryFunction((PetscObject)A,"MatGetDiagonalBlock_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(A,iscopy,reuse,a);CHKERRQ(ierr);
  } else if (size == 1) {
    *a = A;
    *iscopy = PETSC_FALSE;
  } else {
    SETERRQ(PETSC_ERR_SUP,"Cannot get diagonal part for this matrix");
  }
  PetscFunctionReturn(0);
}
#define MatGetDiagonalBlock MatGetDiagonalBlock_300

#undef __FUNCT__
#define __FUNCT__ "MatGetSubMatrix_300"
static PETSC_UNUSED
PetscErrorCode MatGetSubMatrix_300(Mat mat,IS isrow,IS iscol,MatReuse cll,Mat *newmat)
{
  MPI_Comm comm;
  PetscMPIInt size;
  IS iscolall = PETSC_NULL;
  PetscInt csize = PETSC_DECIDE;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidHeaderSpecific(isrow,IS_COOKIE,2);
  if (iscol) PetscValidHeaderSpecific(iscol,IS_COOKIE,3);
  PetscValidPointer(newmat,6);
  if (cll == MAT_REUSE_MATRIX) PetscValidHeaderSpecific(*newmat,MAT_COOKIE,6);
  PetscValidType(mat,1);

  ierr = PetscObjectGetComm((PetscObject)mat,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  if (iscol) {
    ierr = ISGetLocalSize(iscol,&csize);CHKERRQ(ierr);
    if (size == 1) {
      iscolall = iscol;
    } else if (cll == MAT_INITIAL_MATRIX) {
      ierr = ISAllGather(iscol, &iscolall);CHKERRQ(ierr);
    } else {
      ierr = PetscObjectQuery((PetscObject)*newmat,"ISAllGather",(PetscObject*)&iscolall);CHKERRQ(ierr);
      if (!iscolall) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Submatrix passed in was not used before, cannot reuse");
    }
  }
  ierr = MatGetSubMatrix(mat,isrow,iscolall,csize,cll,newmat); CHKERRQ(ierr);
  if (iscol && size > 1 && cll == MAT_INITIAL_MATRIX) {
    ierr = PetscObjectCompose((PetscObject)*newmat,"ISAllGather",(PetscObject)iscolall);CHKERRQ(ierr);
    ierr = ISDestroy(iscolall); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}
#define MatGetSubMatrix MatGetSubMatrix_300

#undef __FUNCT__
#define __FUNCT__ "MatMFFDSetOptionsPrefix"
static PETSC_UNUSED
PetscErrorCode MatMFFDSetOptionsPrefix(Mat mat, const char prefix[]) 
{
  MatMFFD        mfctx = mat ? (MatMFFD)mat->data : PETSC_NULL ;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidHeaderSpecific(mfctx,MATMFFD_COOKIE,1);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)mfctx,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#define MatMFFDSetOptionsPrefix MatMFFDSetOptionsPrefix

#endif /* _PETSC_COMPAT_MAT_H */

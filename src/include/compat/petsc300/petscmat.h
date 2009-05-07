#ifndef _PETSC_COMPAT_MAT_H
#define _PETSC_COMPAT_MAT_H

#include "private/matimpl.h"

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

#endif /* _PETSC_COMPAT_MAT_H */

#define PETSCMAT_DLL

#include "petscmat.h"

#undef __FUNCT__
#define __FUNCT__ "MatPtAP_Basic"
PetscErrorCode MatPtAP_Basic(Mat A,Mat P,MatReuse scall,PetscReal fill,Mat *C) 
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (scall == MAT_INITIAL_MATRIX){
    ierr = MatPtAPSymbolic(A,P,fill,C);CHKERRQ(ierr);
  }
  ierr = MatPtAPNumeric(A,P,*C);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

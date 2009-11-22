#define PETSCMAT_DLL

#include "petscmat.h"
#include "../src/mat/order/order.h"

EXTERN_C_BEGIN
/*
    MatOrdering_ND - Find the nested dissection ordering of a given matrix.
*/    
#undef __FUNCT__  
#define __FUNCT__ "MatOrdering_ND"
PetscErrorCode PETSCMAT_DLLEXPORT MatOrdering_ND(Mat mat,const MatOrderingType type,IS *row,IS *col)
{
  PetscErrorCode ierr;
  PetscInt       i, *mask,*xls,*ls,nrow,*ia,*ja,*perm;
  PetscTruth     done;

  PetscFunctionBegin;
  ierr = MatGetRowIJ(mat,1,PETSC_TRUE,PETSC_TRUE,&nrow,&ia,&ja,&done);CHKERRQ(ierr);
  if (!done) SETERRQ1(PETSC_ERR_SUP,"Cannot get rows for matrix type %s",((PetscObject)mat)->type_name);

  ierr = PetscMalloc4(nrow,PetscInt,&mask,nrow,PetscInt,&perm,nrow+1,PetscInt,&xls,nrow,PetscInt,&ls);CHKERRQ(ierr);
  SPARSEPACKgennd(&nrow,ia,ja,mask,perm,xls,ls);
  ierr = MatRestoreRowIJ(mat,1,PETSC_TRUE,PETSC_TRUE,&nrow,&ia,&ja,&done);CHKERRQ(ierr);

  /* shift because Sparsepack indices start at one */
  for (i=0; i<nrow; i++) perm[i]--;

  ierr = ISCreateGeneral(PETSC_COMM_SELF,nrow,perm,row);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,nrow,perm,col);CHKERRQ(ierr);
  ierr = PetscFree4(mask,perm,xls,ls);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END



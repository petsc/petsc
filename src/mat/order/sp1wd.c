#define PETSCMAT_DLL

#include "petscmat.h"
#include "../src/mat/order/order.h"

EXTERN_C_BEGIN
/*
    MatOrdering_1WD - Find the 1-way dissection ordering of a given matrix.
*/    
#undef __FUNCT__  
#define __FUNCT__ "MatOrdering_1WD"
PetscErrorCode PETSCMAT_DLLEXPORT MatOrdering_1WD(Mat mat,const MatOrderingType type,IS *row,IS *col)
{
  PetscErrorCode ierr;
  PetscInt       i,*mask,*xls,nblks,*xblk,*ls,nrow,*perm,*ia,*ja;
  PetscTruth     done;

  PetscFunctionBegin;
  ierr = MatGetRowIJ(mat,1,PETSC_TRUE,PETSC_TRUE,&nrow,&ia,&ja,&done);CHKERRQ(ierr);
  if (!done) SETERRQ(PETSC_ERR_SUP,"Cannot get rows for matrix");

  ierr = PetscMalloc5(nrow,PetscInt,&mask,nrow+1,PetscInt,&xls,nrow,PetscInt,&ls,nrow+1,PetscInt,&xblk,nrow,PetscInt,&perm);CHKERRQ(ierr);
  SPARSEPACKgen1wd(&nrow,ia,ja,mask,&nblks,xblk,perm,xls,ls);
  ierr = MatRestoreRowIJ(mat,1,PETSC_TRUE,PETSC_TRUE,&nrow,&ia,&ja,&done);CHKERRQ(ierr);

  for (i=0; i<nrow; i++) perm[i]--;

  ierr = ISCreateGeneral(PETSC_COMM_SELF,nrow,perm,row);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,nrow,perm,col);CHKERRQ(ierr);
  ierr = PetscFree5(mask,xls,ls,xblk,perm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END


#define PETSCMAT_DLL

#include "petscmat.h"
#include "src/mat/order/order.h"

EXTERN_C_BEGIN
/*
    MatOrdering_RCM - Find the Reverse Cuthill-McKee ordering of a given matrix.
*/    
#undef __FUNCT__  
#define __FUNCT__ "MatOrdering_RCM"
PetscErrorCode PETSCMAT_DLLEXPORT MatOrdering_RCM(Mat mat,const MatOrderingType type,IS *row,IS *col)
{
  PetscErrorCode ierr;
  PetscInt       i,*mask,*xls,nrow,*ia,*ja,*perm;
  PetscTruth     done;

  PetscFunctionBegin;
  ierr = MatGetRowIJ(mat,1,PETSC_TRUE,&nrow,&ia,&ja,&done);CHKERRQ(ierr);
  if (!done) SETERRQ(PETSC_ERR_SUP,"Cannot get rows for matrix");

  ierr = PetscMalloc(4*nrow * sizeof(PetscInt),&mask);CHKERRQ(ierr);
  perm = mask + nrow;
  xls  = perm + nrow;

  SPARSEPACKgenrcm(&nrow,ia,ja,perm,mask,xls);
  ierr = MatRestoreRowIJ(mat,1,PETSC_TRUE,&nrow,&ia,&ja,&done);CHKERRQ(ierr);

  /* shift because Sparsepack indices start at one */
  for (i=0; i<nrow; i++) perm[i]--;

  ierr = ISCreateGeneral(PETSC_COMM_SELF,nrow,perm,row);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,nrow,perm,col);CHKERRQ(ierr);
  ierr = PetscFree(mask);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END


#include <petscmat.h>
#include <../src/mat/order/order.h>

EXTERN_C_BEGIN
/*
    MatGetOrdering_RCM - Find the Reverse Cuthill-McKee ordering of a given matrix.
*/    
#undef __FUNCT__  
#define __FUNCT__ "MatGetOrdering_RCM"
PetscErrorCode  MatGetOrdering_RCM(Mat mat,const MatOrderingType type,IS *row,IS *col)
{
  PetscErrorCode ierr;
  PetscInt       i,*mask,*xls,nrow,*ia,*ja,*perm;
  PetscBool      done;

  PetscFunctionBegin;
  ierr = MatGetRowIJ(mat,1,PETSC_TRUE,PETSC_TRUE,&nrow,&ia,&ja,&done);CHKERRQ(ierr);
  if (!done) SETERRQ(((PetscObject)mat)->comm,PETSC_ERR_SUP,"Cannot get rows for matrix");

  ierr = PetscMalloc3(nrow,PetscInt,&mask,nrow,PetscInt,&perm,2*nrow,PetscInt,&xls);CHKERRQ(ierr);
  SPARSEPACKgenrcm(&nrow,ia,ja,perm,mask,xls);
  ierr = MatRestoreRowIJ(mat,1,PETSC_TRUE,PETSC_TRUE,&nrow,&ia,&ja,&done);CHKERRQ(ierr);

  /* shift because Sparsepack indices start at one */
  for (i=0; i<nrow; i++) perm[i]--;
  ierr = ISCreateGeneral(PETSC_COMM_SELF,nrow,perm,PETSC_COPY_VALUES,row);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,nrow,perm,PETSC_COPY_VALUES,col);CHKERRQ(ierr);
  ierr = PetscFree3(mask,perm,xls);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END


#include <petscmat.h>
#include <petsc/private/matorderimpl.h>

/*
    MatGetOrdering_1WD - Find the 1-way dissection ordering of a given matrix.
*/
PETSC_INTERN PetscErrorCode MatGetOrdering_1WD(Mat mat,MatOrderingType type,IS *row,IS *col)
{
  PetscErrorCode ierr;
  PetscInt       i,*mask,*xls,nblks,*xblk,*ls,nrow,*perm;
  const PetscInt *ia,*ja;
  PetscBool      done;

  PetscFunctionBegin;
  ierr = MatGetRowIJ(mat,1,PETSC_TRUE,PETSC_TRUE,&nrow,&ia,&ja,&done);CHKERRQ(ierr);
  PetscCheckFalse(!done,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Cannot get rows for matrix");

  ierr = PetscMalloc5(nrow,&mask,nrow+1,&xls,nrow,&ls,nrow+1,&xblk,nrow,&perm);CHKERRQ(ierr);
  SPARSEPACKgen1wd(&nrow,ia,ja,mask,&nblks,xblk,perm,xls,ls);
  ierr = MatRestoreRowIJ(mat,1,PETSC_TRUE,PETSC_TRUE,NULL,&ia,&ja,&done);CHKERRQ(ierr);

  for (i=0; i<nrow; i++) perm[i]--;

  ierr = ISCreateGeneral(PETSC_COMM_SELF,nrow,perm,PETSC_COPY_VALUES,row);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,nrow,perm,PETSC_COPY_VALUES,col);CHKERRQ(ierr);
  ierr = PetscFree5(mask,xls,ls,xblk,perm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



#include <petscmat.h>
#include <petsc/private/matorderimpl.h>

/*
    MatGetOrdering_ND - Find the nested dissection ordering of a given matrix.
*/
PETSC_INTERN PetscErrorCode MatGetOrdering_ND(Mat mat,MatOrderingType type,IS *row,IS *col)
{
  PetscErrorCode ierr;
  PetscInt       i, *mask,*xls,*ls,nrow,*perm;
  const PetscInt *ia,*ja;
  PetscBool      done;

  PetscFunctionBegin;
  ierr = MatGetRowIJ(mat,1,PETSC_TRUE,PETSC_TRUE,&nrow,&ia,&ja,&done);CHKERRQ(ierr);
  if (!done) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot get rows for matrix type %s",((PetscObject)mat)->type_name);

  ierr = PetscMalloc4(nrow,&mask,nrow,&perm,nrow+1,&xls,nrow,&ls);CHKERRQ(ierr);
  SPARSEPACKgennd(&nrow,ia,ja,mask,perm,xls,ls);
  ierr = MatRestoreRowIJ(mat,1,PETSC_TRUE,PETSC_TRUE,NULL,&ia,&ja,&done);CHKERRQ(ierr);

  /* shift because Sparsepack indices start at one */
  for (i=0; i<nrow; i++) perm[i]--;

  ierr = ISCreateGeneral(PETSC_COMM_SELF,nrow,perm,PETSC_COPY_VALUES,row);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,nrow,perm,PETSC_COPY_VALUES,col);CHKERRQ(ierr);
  ierr = PetscFree4(mask,perm,xls,ls);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



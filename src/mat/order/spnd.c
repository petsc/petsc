
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
  Mat            B = NULL;

  PetscFunctionBegin;
  ierr = MatGetRowIJ(mat,1,PETSC_TRUE,PETSC_TRUE,&nrow,&ia,&ja,&done);CHKERRQ(ierr);
  if (!done) {
    ierr = MatConvert(mat,MATSEQAIJ,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);
    ierr = MatGetRowIJ(B,1,PETSC_TRUE,PETSC_TRUE,&nrow,&ia,&ja,&done);CHKERRQ(ierr);
  }

  ierr = PetscMalloc4(nrow,&mask,nrow,&perm,nrow+1,&xls,nrow,&ls);CHKERRQ(ierr);
  SPARSEPACKgennd(&nrow,ia,ja,mask,perm,xls,ls);
  if (B) {
    ierr = MatRestoreRowIJ(B,1,PETSC_TRUE,PETSC_TRUE,NULL,&ia,&ja,&done);CHKERRQ(ierr);
    ierr = MatDestroy(&B);CHKERRQ(ierr);
  } else {
    ierr = MatRestoreRowIJ(mat,1,PETSC_TRUE,PETSC_TRUE,NULL,&ia,&ja,&done);CHKERRQ(ierr);
  }

  /* shift because Sparsepack indices start at one */
  for (i=0; i<nrow; i++) perm[i]--;

  ierr = ISCreateGeneral(PETSC_COMM_SELF,nrow,perm,PETSC_COPY_VALUES,row);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,nrow,perm,PETSC_COPY_VALUES,col);CHKERRQ(ierr);
  ierr = PetscFree4(mask,perm,xls,ls);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


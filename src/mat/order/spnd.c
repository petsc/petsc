
#include <petscmat.h>
#include <petsc/private/matorderimpl.h>

/*
    MatGetOrdering_ND - Find the nested dissection ordering of a given matrix.
*/
PETSC_INTERN PetscErrorCode MatGetOrdering_ND(Mat mat,MatOrderingType type,IS *row,IS *col)
{
  PetscInt       i, *mask,*xls,*ls,nrow,*perm;
  const PetscInt *ia,*ja;
  PetscBool      done;
  Mat            B = NULL;

  PetscFunctionBegin;
  CHKERRQ(MatGetRowIJ(mat,1,PETSC_TRUE,PETSC_TRUE,&nrow,&ia,&ja,&done));
  if (!done) {
    CHKERRQ(MatConvert(mat,MATSEQAIJ,MAT_INITIAL_MATRIX,&B));
    CHKERRQ(MatGetRowIJ(B,1,PETSC_TRUE,PETSC_TRUE,&nrow,&ia,&ja,&done));
  }

  CHKERRQ(PetscMalloc4(nrow,&mask,nrow,&perm,nrow+1,&xls,nrow,&ls));
  SPARSEPACKgennd(&nrow,ia,ja,mask,perm,xls,ls);
  if (B) {
    CHKERRQ(MatRestoreRowIJ(B,1,PETSC_TRUE,PETSC_TRUE,NULL,&ia,&ja,&done));
    CHKERRQ(MatDestroy(&B));
  } else {
    CHKERRQ(MatRestoreRowIJ(mat,1,PETSC_TRUE,PETSC_TRUE,NULL,&ia,&ja,&done));
  }

  /* shift because Sparsepack indices start at one */
  for (i=0; i<nrow; i++) perm[i]--;

  CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,nrow,perm,PETSC_COPY_VALUES,row));
  CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,nrow,perm,PETSC_COPY_VALUES,col));
  CHKERRQ(PetscFree4(mask,perm,xls,ls));
  PetscFunctionReturn(0);
}

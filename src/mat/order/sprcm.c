
#include <petscmat.h>
#include <petsc/private/matorderimpl.h>

/*
    MatGetOrdering_RCM - Find the Reverse Cuthill-McKee ordering of a given matrix.
*/
PETSC_INTERN PetscErrorCode MatGetOrdering_RCM(Mat mat,MatOrderingType type,IS *row,IS *col)
{
  PetscInt       i,*mask,*xls,nrow,*perm;
  const PetscInt *ia,*ja;
  PetscBool      done;

  PetscFunctionBegin;
  PetscCall(MatGetRowIJ(mat,1,PETSC_TRUE,PETSC_TRUE,&nrow,&ia,&ja,&done));
  PetscCheck(done,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Cannot get rows for matrix");

  PetscCall(PetscMalloc3(nrow,&mask,nrow,&perm,2*nrow,&xls));
  SPARSEPACKgenrcm(&nrow,ia,ja,perm,mask,xls);
  PetscCall(MatRestoreRowIJ(mat,1,PETSC_TRUE,PETSC_TRUE,NULL,&ia,&ja,&done));

  /* shift because Sparsepack indices start at one */
  for (i=0; i<nrow; i++) perm[i]--;
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF,nrow,perm,PETSC_COPY_VALUES,row));
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF,nrow,perm,PETSC_COPY_VALUES,col));
  PetscCall(PetscFree3(mask,perm,xls));
  PetscFunctionReturn(0);
}

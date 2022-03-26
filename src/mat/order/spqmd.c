
#include <petscmat.h>
#include <petsc/private/matorderimpl.h>

/*
    MatGetOrdering_QMD - Find the Quotient Minimum Degree ordering of a given matrix.
*/
PETSC_INTERN PetscErrorCode MatGetOrdering_QMD(Mat mat,MatOrderingType type,IS *row,IS *col)
{
  PetscInt       i,  *deg,*marker,*rchset,*nbrhd,*qsize,*qlink,nofsub,*iperm,nrow,*perm;
  const PetscInt *ia,*ja;
  PetscBool      done;

  PetscFunctionBegin;
  PetscCall(MatGetRowIJ(mat,1,PETSC_TRUE,PETSC_TRUE,&nrow,&ia,&ja,&done));
  PetscCheck(done,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Cannot get rows for matrix");

  PetscCall(PetscMalloc1(nrow,&perm));
  PetscCall(PetscMalloc5(nrow,&iperm,nrow,&deg,nrow,&marker,nrow,&rchset,nrow,&nbrhd));
  PetscCall(PetscMalloc2(nrow,&qsize,nrow,&qlink));
  /* WARNING - genqmd trashes ja */
  SPARSEPACKgenqmd(&nrow,ia,ja,perm,iperm,deg,marker,rchset,nbrhd,qsize,qlink,&nofsub);
  PetscCall(MatRestoreRowIJ(mat,1,PETSC_TRUE,PETSC_TRUE,NULL,&ia,&ja,&done));

  PetscCall(PetscFree2(qsize,qlink));
  PetscCall(PetscFree5(iperm,deg,marker,rchset,nbrhd));
  for (i=0; i<nrow; i++) perm[i]--;
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF,nrow,perm,PETSC_COPY_VALUES,row));
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF,nrow,perm,PETSC_OWN_POINTER,col));
  PetscFunctionReturn(0);
}

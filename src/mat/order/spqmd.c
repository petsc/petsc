
#include <petscmat.h>
#include <petsc/private/matorderimpl.h>

/*
    MatGetOrdering_QMD - Find the Quotient Minimum Degree ordering of a given matrix.
*/
PETSC_INTERN PetscErrorCode MatGetOrdering_QMD(Mat mat,MatOrderingType type,IS *row,IS *col)
{
  PetscInt       i,  *deg,*marker,*rchset,*nbrhd,*qsize,*qlink,nofsub,*iperm,nrow,*perm;
  PetscErrorCode ierr;
  const PetscInt *ia,*ja;
  PetscBool      done;

  PetscFunctionBegin;
  ierr = MatGetRowIJ(mat,1,PETSC_TRUE,PETSC_TRUE,&nrow,&ia,&ja,&done);CHKERRQ(ierr);
  PetscAssertFalse(!done,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Cannot get rows for matrix");

  ierr = PetscMalloc1(nrow,&perm);CHKERRQ(ierr);
  ierr = PetscMalloc5(nrow,&iperm,nrow,&deg,nrow,&marker,nrow,&rchset,nrow,&nbrhd);CHKERRQ(ierr);
  ierr = PetscMalloc2(nrow,&qsize,nrow,&qlink);CHKERRQ(ierr);
  /* WARNING - genqmd trashes ja */
  SPARSEPACKgenqmd(&nrow,ia,ja,perm,iperm,deg,marker,rchset,nbrhd,qsize,qlink,&nofsub);
  ierr = MatRestoreRowIJ(mat,1,PETSC_TRUE,PETSC_TRUE,NULL,&ia,&ja,&done);CHKERRQ(ierr);

  ierr = PetscFree2(qsize,qlink);CHKERRQ(ierr);
  ierr = PetscFree5(iperm,deg,marker,rchset,nbrhd);CHKERRQ(ierr);
  for (i=0; i<nrow; i++) perm[i]--;
  ierr = ISCreateGeneral(PETSC_COMM_SELF,nrow,perm,PETSC_COPY_VALUES,row);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,nrow,perm,PETSC_OWN_POINTER,col);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#include <petscmat.h>
#include <../src/mat/order/order.h>

EXTERN_C_BEGIN
/*
    MatGetOrdering_QMD - Find the Quotient Minimum Degree ordering of a given matrix.
*/
#undef __FUNCT__
#define __FUNCT__ "MatGetOrdering_QMD"
PetscErrorCode  MatGetOrdering_QMD(Mat mat,MatOrderingType type,IS *row,IS *col)
{
  PetscInt       i,  *deg,*marker,*rchset,*nbrhd,*qsize,*qlink,nofsub,*iperm,nrow,*perm;
  PetscErrorCode ierr;
  const PetscInt *ia,*ja;
  PetscBool       done;

  PetscFunctionBegin;
  ierr = MatGetRowIJ(mat,1,PETSC_TRUE,PETSC_TRUE,&nrow,&ia,&ja,&done);CHKERRQ(ierr);
  if (!done) SETERRQ(((PetscObject)mat)->comm,PETSC_ERR_SUP,"Cannot get rows for matrix");

  ierr = PetscMalloc(nrow * sizeof(PetscInt),&perm);CHKERRQ(ierr);
  ierr = PetscMalloc5(nrow,PetscInt,&iperm,nrow,PetscInt,&deg,nrow,PetscInt,&marker,nrow,PetscInt,&rchset,nrow,PetscInt,&nbrhd);CHKERRQ(ierr);
  ierr = PetscMalloc2(nrow,PetscInt,&qsize,nrow,PetscInt,&qlink);CHKERRQ(ierr);
  /* WARNING - genqmd trashes ja */
  SPARSEPACKgenqmd(&nrow,ia,ja,perm,iperm,deg,marker,rchset,nbrhd,qsize,qlink,&nofsub);
  ierr = MatRestoreRowIJ(mat,1,PETSC_TRUE,PETSC_TRUE,&nrow,&ia,&ja,&done);CHKERRQ(ierr);

  ierr = PetscFree2(qsize,qlink);CHKERRQ(ierr);
  ierr = PetscFree5(iperm,deg,marker,rchset,nbrhd);CHKERRQ(ierr);
  for (i=0; i<nrow; i++) perm[i]--;
  ierr = ISCreateGeneral(PETSC_COMM_SELF,nrow,perm,PETSC_COPY_VALUES,row);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,nrow,perm,PETSC_OWN_POINTER,col);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

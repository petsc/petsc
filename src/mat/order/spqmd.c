#define PETSCMAT_DLL

#include "petscmat.h"
#include "src/mat/order/order.h"

EXTERN_C_BEGIN
/*
    MatOrdering_QMD - Find the Quotient Minimum Degree ordering of a given matrix.
*/    
#undef __FUNCT__  
#define __FUNCT__ "MatOrdering_QMD"
PetscErrorCode PETSCMAT_DLLEXPORT MatOrdering_QMD(Mat mat,const MatOrderingType type,IS *row,IS *col) 
{
  PetscInt       i,  *deg,*marker,*rchset,*nbrhd,*qsize,*qlink,nofsub,*iperm,nrow;
  PetscErrorCode ierr;
  PetscInt       *ia,*ja,*perm;
  PetscTruth      done; 

  PetscFunctionBegin;
  ierr = MatGetRowIJ(mat,1,PETSC_TRUE,&nrow,&ia,&ja,&done);CHKERRQ(ierr);
  if (!done) SETERRQ(PETSC_ERR_SUP,"Cannot get rows for matrix");

  ierr = PetscMalloc(nrow * sizeof(PetscInt),&perm);CHKERRQ(ierr);
  ierr = PetscMalloc(nrow * sizeof(PetscInt),&iperm);CHKERRQ(ierr);
  ierr = PetscMalloc(nrow * sizeof(PetscInt),&deg);CHKERRQ(ierr);
  ierr = PetscMalloc(nrow * sizeof(PetscInt),&marker);CHKERRQ(ierr);
  ierr = PetscMalloc(nrow * sizeof(PetscInt),&rchset);CHKERRQ(ierr);
  ierr = PetscMalloc(nrow * sizeof(PetscInt),&nbrhd);CHKERRQ(ierr);
  ierr = PetscMalloc(nrow * sizeof(PetscInt),&qsize);CHKERRQ(ierr);
  ierr = PetscMalloc(nrow * sizeof(PetscInt),&qlink);CHKERRQ(ierr);
  /* WARNING - genqmd trashes ja */    
  SPARSEPACKgenqmd(&nrow,ia,ja,perm,iperm,deg,marker,rchset,nbrhd,qsize,qlink,&nofsub);
  ierr = MatRestoreRowIJ(mat,1,PETSC_TRUE,&nrow,&ia,&ja,&done);CHKERRQ(ierr);

  ierr = PetscFree(deg);CHKERRQ(ierr);
  ierr = PetscFree(marker);CHKERRQ(ierr);
  ierr = PetscFree(rchset);CHKERRQ(ierr);
  ierr = PetscFree(nbrhd);CHKERRQ(ierr);
  ierr = PetscFree(qsize);CHKERRQ(ierr);
  ierr = PetscFree(qlink);CHKERRQ(ierr);
  ierr = PetscFree(iperm);CHKERRQ(ierr);
  for (i=0; i<nrow; i++) perm[i]--;
  ierr = ISCreateGeneral(PETSC_COMM_SELF,nrow,perm,row);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,nrow,perm,col);CHKERRQ(ierr);
  ierr = PetscFree(perm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

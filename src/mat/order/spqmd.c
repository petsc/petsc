/*$Id: spqmd.c,v 1.37 2000/09/28 21:12:22 bsmith Exp bsmith $*/

#include "petscmat.h"
#include "src/mat/order/order.h"

EXTERN_C_BEGIN
/*
    MatOrdering_QMD - Find the Quotient Minimum Degree ordering of a given matrix.
*/    
#undef __FUNC__  
#define __FUNC__ "MatOrdering_QMD"
int MatOrdering_QMD(Mat mat,MatOrderingType type,IS *row,IS *col) 
{
  int        i,  *deg,*marker,*rchset,*nbrhd,*qsize,*qlink,nofsub,*iperm,nrow;
  int        ierr,*ia,*ja,*perm;
  PetscTruth done; 

  PetscFunctionBegin;
  ierr = MatGetRowIJ(mat,1,PETSC_TRUE,&nrow,&ia,&ja,&done);CHKERRQ(ierr);
  if (!done) SETERRQ(PETSC_ERR_SUP,"Cannot get rows for matrix");

  perm   = (int *)PetscMalloc(nrow * sizeof(int));CHKERRQ(ierr);
  iperm  = (int *)PetscMalloc(nrow * sizeof(int));CHKERRQ(ierr);
  deg    = (int *)PetscMalloc(nrow * sizeof(int));CHKERRQ(ierr);
  marker = (int *)PetscMalloc(nrow * sizeof(int));CHKERRQ(ierr);
  rchset = (int *)PetscMalloc(nrow * sizeof(int));CHKERRQ(ierr);
  nbrhd  = (int *)PetscMalloc(nrow * sizeof(int));CHKERRQ(ierr);
  qsize  = (int *)PetscMalloc(nrow * sizeof(int));CHKERRQ(ierr);
  qlink  = (int *)PetscMalloc(nrow * sizeof(int));CHKERRQ(ierr);
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

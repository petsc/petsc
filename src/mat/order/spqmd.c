#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: spqmd.c,v 1.28 1999/05/04 18:25:58 balay Exp balay $";
#endif

#include "mat.h"
#include "src/mat/impls/order/order.h"

EXTERN_C_BEGIN
/*
    MatOrdering_QMD - Find the Quotient Minimum Degree ordering of a given matrix.
*/    
#undef __FUNC__  
#define __FUNC__ "MatOrdering_QMD"
int MatOrdering_QMD(Mat mat, MatOrderingType type, IS *row, IS *col) 
{
  int        i,   *deg, *marker, *rchset, *nbrhd, *qsize, *qlink, nofsub, *iperm, nrow;
  int        ierr, *ia,*ja,*perm;
  PetscTruth done; 

  PetscFunctionBegin;
  ierr = MatGetRowIJ(mat,1,PETSC_TRUE,&nrow,&ia,&ja,&done);CHKERRQ(ierr);
  if (!done) SETERRQ(PETSC_ERR_SUP,0,"Cannot get rows for matrix");

  perm   = (int *)PetscMalloc( nrow * sizeof(int) );CHKPTRQ(perm);
  iperm  = (int *)PetscMalloc( nrow * sizeof(int) );CHKPTRQ(iperm);
  deg    = (int *)PetscMalloc( nrow * sizeof(int) );CHKPTRQ(deg);
  marker = (int *)PetscMalloc( nrow * sizeof(int) );CHKPTRQ(marker);
  rchset = (int *)PetscMalloc( nrow * sizeof(int) );CHKPTRQ(rchset);
  nbrhd  = (int *)PetscMalloc( nrow * sizeof(int) );CHKPTRQ(nbrhd);
  qsize  = (int *)PetscMalloc( nrow * sizeof(int) );CHKPTRQ(qsize);
  qlink  = (int *)PetscMalloc( nrow * sizeof(int) );CHKPTRQ(qlink);
  /* WARNING - genqmd trashes ja */    
  SPARSEPACKgenqmd( &nrow, ia, ja, perm, iperm, deg, marker, rchset, nbrhd, qsize,qlink, &nofsub );
  ierr = MatRestoreRowIJ(mat,1,PETSC_TRUE,&nrow,&ia,&ja,&done);CHKERRQ(ierr);

  PetscFree( deg ); PetscFree( marker ); PetscFree( rchset ); 
  PetscFree( nbrhd ); PetscFree( qsize );
  PetscFree( qlink ); PetscFree(iperm);
  for (i=0; i<nrow; i++) perm[i]--;
  ierr = ISCreateGeneral(PETSC_COMM_SELF,nrow,perm,row);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,nrow,perm,col);CHKERRQ(ierr);
  PetscFree(perm);
  PetscFunctionReturn(0);
}
EXTERN_C_END

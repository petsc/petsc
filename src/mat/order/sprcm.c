#ifndef lint
static char vcid[] = "$Id: sprcm.c,v 1.19 1997/02/04 19:47:54 balay Exp bsmith $";
#endif

#include "petsc.h"
#include "mat.h"
#include "src/mat/impls/order/order.h"

/*
    MatOrder_RCM - Find the Reverse Cuthill-McKee ordering of a given matrix.
*/    
#undef __FUNC__  
#define __FUNC__ "MatOrder_RCM" /* ADIC Ignore */
int MatOrder_RCM( Mat mat, MatReordering type, IS *row, IS *col)
{
  int        ierr,i,   *mask, *xls, nrow,*ia,*ja,*perm;
  PetscTruth done;

  ierr = MatGetRowIJ(mat,1,PETSC_TRUE,&nrow,&ia,&ja,&done); CHKERRQ(ierr);
  if (!done) SETERRQ(1,0,"Cannot get rows for matrix");

  mask = (int *)PetscMalloc( 4*nrow * sizeof(int) ); CHKPTRQ(mask);
  perm = mask + nrow;
  xls  = perm + nrow;

  genrcm( &nrow, ia, ja, perm, mask, xls );
  ierr = MatRestoreRowIJ(mat,1,PETSC_TRUE,&nrow,&ia,&ja,&done); CHKERRQ(ierr);

  /* shift because Sparsepack indices start at one */
  for (i=0; i<nrow; i++) perm[i]--;

  ierr = ISCreateGeneral(MPI_COMM_SELF,nrow,perm,row); CHKERRQ(ierr);
  ierr = ISCreateGeneral(MPI_COMM_SELF,nrow,perm,col); CHKERRQ(ierr);
  PetscFree(mask);

  return 0;
}

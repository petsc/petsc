#ifndef lint
static char vcid[] = "$Id: sprcm.c,v 1.13 1996/08/08 14:43:21 bsmith Exp bsmith $";
#endif

#include "petsc.h"
#include "mat.h"
#include "src/mat/impls/order/order.h"

/*
    MatOrder_RCM - Find the Reverse Cuthill-McGee ordering of a given matrix.
*/    
int MatOrder_RCM( Mat mat, MatReordering type, IS *row, IS *col)
{
  int        ierr,i,   *mask, *xls, nrow,*ia,*ja,*perm;
  PetscTruth done;

  ierr = MatGetRowIJ(mat,1,PETSC_TRUE,&nrow,&ia,&ja,&done); CHKERRQ(ierr);
  if (!done) SETERRQ(1,"MatOrder_RCM:Cannot get rows for matrix");

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

#ifndef lint
static char vcid[] = "$Id: spnd.c,v 1.19 1997/02/22 02:25:40 bsmith Exp bsmith $";
#endif

#include "petsc.h"
#include "mat.h"
#include "src/mat/impls/order/order.h"

/*
    MatOrder_ND - Find the nested dissection ordering of a given matrix.
*/    
#undef __FUNC__  
#define __FUNC__ "MatOrder_ND" /* ADIC Ignore */
int MatOrder_ND( Mat mat, MatReordering type, IS *row, IS *col)
{
  int        ierr, i,  *mask, *xls, *ls, nrow,*ia,*ja,*perm;
  PetscTruth done;

  ierr = MatGetRowIJ(mat,1,PETSC_TRUE,&nrow,&ia,&ja,&done); CHKERRQ(ierr);
  if (!done) SETERRQ(1,0,"Cannot get rows for matrix");

  mask = (int *)PetscMalloc( (4*nrow +1) * sizeof(int) ); CHKPTRQ(mask);
  perm = mask + nrow;
  xls  = perm + nrow;
  ls   = xls  + nrow + 1;

  gennd( &nrow, ia, ja, mask, perm, xls, ls );
  ierr = MatRestoreRowIJ(mat,1,PETSC_TRUE,&nrow,&ia,&ja,&done); CHKERRQ(ierr);

  /* shift because Sparsepack indices start at one */
  for (i=0; i<nrow; i++) perm[i]--;

  ierr = ISCreateGeneral(PETSC_COMM_SELF,nrow,perm,row); CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,nrow,perm,col); CHKERRQ(ierr);
  PetscFree(mask);

  return 0;
}



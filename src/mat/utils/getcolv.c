#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: getcolv.c,v 1.2 1998/03/06 00:16:14 bsmith Exp bsmith $";
#endif

#include "src/mat/matimpl.h"  /*I   "mat.h"  I*/

#undef __FUNC__  
#define __FUNC__ "MatGetColumnVector"
/*@
   MatGetColumnVector - Gets the values from a given column of a matrix.

   Input Parameters:
.  X - the matrix
.  v - the vector
.  c - the column requested

   Collective on Mat and Vec

   Contributed by: Denis Vanderstraeten

.keywords: matrix, column, get 

@*/
int MatGetColumnVector(Mat A, Vec yy, int col)
{
  Scalar *y,*v,zero = 0.0;
  int    ierr,i,j,nz,*idx,M,N,Mv,Rs,Re,rs,re;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_COOKIE); 
  PetscValidHeaderSpecific(yy,VEC_COOKIE); 

  if (col < 0)  SETERRQ(1,1,"Requested negative column");
  ierr = MatGetSize(A,&M,&N); CHKERRQ(ierr);
  if (col >= N)  SETERRQ(1,1,"Requested column larger than number columns in matrix");

  ierr = VecGetSize(yy,&Mv); CHKERRQ(ierr);
  if (M != Mv) SETERRQ(1,1,"Matrix does not have same number of columns as vector");

  ierr = MatGetOwnershipRange(A,&Rs,&Re);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(yy,&rs,&re);CHKERRQ(ierr);
  if (Rs != rs || Re != re) SETERRQ(1,1,"Matrix does not have same ownership range as vector");

  ierr = VecSet(&zero,yy);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);

  for ( i=Rs; i<Re; i++ ) {
    ierr = MatGetRow(A,i,&nz,&idx,&v);CHKERRQ(ierr);
    if (nz && idx[0] <= col) {
      /*
          Should use faster search here 
      */
      for ( j=0; j<nz; j++ ) {
        if (idx[j] >= col) {
          if (idx[j] == col) y[i-rs] = v[j];
          break;
        }
      }
    }
    ierr = MatRestoreRow(A,i,&nz,&idx,&v);CHKERRQ(ierr);
  }

  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);  
  PetscFunctionReturn(0);
}

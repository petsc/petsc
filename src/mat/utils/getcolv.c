/*$Id: getcolv.c,v 1.10 2000/04/09 04:36:57 bsmith Exp bsmith $*/

#include "src/mat/matimpl.h"  /*I   "mat.h"  I*/

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatGetColumnVector"
/*@
   MatGetColumnVector - Gets the values from a given column of a matrix.

   Collective on Mat and Vec

   Input Parameters:
+  X - the matrix
.  v - the vector
-  c - the column requested

   Level: advanced

   Contributed by: Denis Vanderstraeten

.keywords: matrix, column, get 

.seealso: MatGetRow(), MatGetDiagonal()

@*/
int MatGetColumnVector(Mat A,Vec yy,int col)
{
  Scalar *y,*v,zero = 0.0;
  int    ierr,i,j,nz,*idx,M,N,Mv,Rs,Re,rs,re;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_COOKIE); 
  PetscValidHeaderSpecific(yy,VEC_COOKIE); 

  if (col < 0)  SETERRQ1(1,1,"Requested negative column: %d",col);
  ierr = MatGetSize(A,&M,&N);CHKERRQ(ierr);
  if (col >= N)  SETERRQ2(1,1,"Requested column %d larger than number columns in matrix %d",col,N);

  ierr = VecGetSize(yy,&Mv);CHKERRQ(ierr);
  if (M != Mv) SETERRQ2(1,1,"Matrix does not have same number of columns %d as vector %d",M,Mv);

  ierr = MatGetOwnershipRange(A,&Rs,&Re);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(yy,&rs,&re);CHKERRQ(ierr);
  if (Rs != rs || Re != re) SETERRQ4(1,1,"Matrix %d %d does not have same ownership range as vector %d %d",Rs,Re,rs,re);

  ierr = VecSet(&zero,yy);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);

  for (i=Rs; i<Re; i++) {
    ierr = MatGetRow(A,i,&nz,&idx,&v);CHKERRQ(ierr);
    if (nz && idx[0] <= col) {
      /*
          Should use faster search here 
      */
      for (j=0; j<nz; j++) {
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

/*$Id: getcolv.c,v 1.23 2001/08/08 15:47:19 bsmith Exp $*/

#include "src/mat/matimpl.h"  /*I   "petscmat.h"  I*/

#undef __FUNCT__  
#define __FUNCT__ "MatGetColumnVector"
/*@
   MatGetColumnVector - Gets the values from a given column of a matrix.

   Not Collective

   Input Parameters:
+  A - the matrix
.  yy - the vector
-  c - the column requested (in global numbering)

   Level: advanced

   Notes:
   Each processor for which this is called gets the values for its rows.

   Since PETSc matrices are usually stored in compressed row format, this routine
   will generally be slow.

   The vector must have the same parallel row layout as the matrix.

   Contributed by: Denis Vanderstraeten

.keywords: matrix, column, get 

.seealso: MatGetRow(), MatGetDiagonal()

@*/
int MatGetColumnVector(Mat A,Vec yy,int col)
{
  PetscScalar   *y,*v,zero = 0.0;
  int      ierr,i,j,nz,*idx,N,Rs,Re,rs,re;
  MPI_Comm comm;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_COOKIE,1); 
  PetscValidHeaderSpecific(yy,VEC_COOKIE,2); 

  if (col < 0)  SETERRQ1(1,"Requested negative column: %d",col);
  ierr = MatGetSize(A,PETSC_NULL,&N);CHKERRQ(ierr);
  if (col >= N)  SETERRQ2(1,"Requested column %d larger than number columns in matrix %d",col,N);

  ierr = MatGetOwnershipRange(A,&Rs,&Re);CHKERRQ(ierr);

  ierr = PetscObjectGetComm((PetscObject)yy,&comm);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(yy,&rs,&re);CHKERRQ(ierr);
  if (Rs != rs || Re != re) SETERRQ4(1,"Matrix %d %d does not have same ownership range (size) as vector %d %d",Rs,Re,rs,re);

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

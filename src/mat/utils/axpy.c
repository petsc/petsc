#ifndef lint
static char vcid[] = "$Id: axpy.c,v 1.12 1995/11/01 23:19:34 bsmith Exp bsmith $";
#endif

#include "matimpl.h"  /*I   "mat.h"  I*/

/*@
   MatAXPY - Computes Y = a*X + Y

   Input Parameters:
.  X,Y - the matrices
.  a - the scalar multiplier

.keywords: matrix, add
 @*/
int MatAXPY(Scalar *a,Mat X,Mat Y)
{
  int    m1,m2,n1,n2,i,*row,start,end,j,ncols,ierr;
  Scalar *val,*vals;

  PETSCVALIDHEADERSPECIFIC(X,MAT_COOKIE);  PETSCVALIDHEADERSPECIFIC(Y,MAT_COOKIE);
  MatGetSize(X,&m1,&n1);  MatGetSize(X,&m2,&n2);
  if (m1 != m2 || n1 != n2) SETERRQ(1,"MatAXPY:Non conforming matrix add");

  if (X->ops.axpy) {
    ierr = (*X->ops.axpy)(a,X,Y); CHKERRQ(ierr);
  }
  else {
    vals = (Scalar *) PetscMalloc( n1*sizeof(Scalar) ); CHKPTRQ(vals);
    MatGetOwnershipRange(X,&start,&end);
    for ( i=start; i<end; i++ ) {
      MatGetRow(X,i,&ncols,&row,&val);
      for ( j=0; j<ncols; j++ ) {
        vals[j] = (*a)*val[j];
      }
      ierr = MatSetValues(Y,1,&i,ncols,row,vals,ADD_VALUES); CHKERRQ(ierr);
      MatRestoreRow(X,i,&ncols,&row,&val);
    }
    PetscFree(vals);
    ierr = MatAssemblyBegin(Y,FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Y,FINAL_ASSEMBLY); CHKERRQ(ierr);
  }
  return 0;
}

/*@
   MatShift - Computes Y =  Y + a I

   Input Parameters:
.  Y - the matrices
.  a - the scalar 

.keywords: matrix, add, shift
 @*/
int MatShift(Scalar *a,Mat Y)
{
  int    i,start,end,ierr;

  PETSCVALIDHEADERSPECIFIC(Y,MAT_COOKIE);
  if (Y->ops.shift) {
    ierr = (*Y->ops.shift)(a,Y); CHKERRQ(ierr);
  }
  else {
    MatGetOwnershipRange(Y,&start,&end);
    for ( i=start; i<end; i++ ) {
      ierr = MatSetValues(Y,1,&i,1,&i,a,ADD_VALUES); CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(Y,FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Y,FINAL_ASSEMBLY); CHKERRQ(ierr);
  }
  return 0;
}

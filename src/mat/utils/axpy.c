#ifndef lint
static char vcid[] = "$Id: axpy.c,v 1.9 1995/09/21 20:11:31 bsmith Exp curfman $";
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
  vals = (Scalar *) PETSCMALLOC( n1*sizeof(Scalar) ); CHKPTRQ(vals);
  MatGetOwnershipRange(X,&start,&end);
  for ( i=start; i<end; i++ ) {
    MatGetRow(X,i,&ncols,&row,&val);
    for ( j=0; j<ncols; j++ ) {
      vals[j] = (*a)*val[j];
    }
    ierr = MatSetValues(Y,1,&i,ncols,row,vals,ADD_VALUES); CHKERRQ(ierr);
    MatRestoreRow(X,i,&ncols,&row,&val);
  }
  PETSCFREE(vals);
  ierr = MatAssemblyBegin(Y,FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Y,FINAL_ASSEMBLY); CHKERRQ(ierr);
  return 0;
}

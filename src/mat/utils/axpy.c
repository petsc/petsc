#ifndef lint
static char vcid[] = "$Id: axpy.c,v 1.3 1995/05/16 00:42:16 curfman Exp bsmith $";
#endif

#include "matimpl.h"

/*@
   MatAXPY - Y = a*X + Y

   Input Parameters:
.  X,Y - the matrices
.  a - the scalar multiple

.keywods: Mat, add
 @*/

int MatAXPY(Scalar *a,Mat X,Mat Y)
{
  int    m1,m2,n1,n2,i,*row,start,end,j,ncols,ierr;
  Scalar *val,*vals;

  VALIDHEADER(X,MAT_COOKIE);  VALIDHEADER(Y,MAT_COOKIE);
  MatGetSize(X,&m1,&n1);  MatGetSize(X,&m2,&n2);
  if (m1 != m2 || n1 != n2) SETERRQ(1,"Non conforming matrix add");
  vals = (Scalar *) PETSCMALLOC( n1*sizeof(Scalar) ); CHKPTRQ(vals);
  MatGetOwnershipRange(X,&start,&end);
  for ( i=start; i<end; i++ ) {
    MatGetRow(X,i,&ncols,&row,&val);
    for ( j=0; j<ncols; j++ ) {
      vals[j] = (*a)*val[j];
    }
    ierr = MatSetValues(Y,1,&i,ncols,row,vals,ADDVALUES); CHKERRQ(ierr);
    MatRestoreRow(X,i,&ncols,&row,&val);
  }
  PETSCFREE(vals);
  ierr = MatAssemblyBegin(Y,FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Y,FINAL_ASSEMBLY); CHKERRQ(ierr);
  return 0;
}

#ifndef lint
static char vcid[] = "$Id: axpy.c,v 1.2 1995/05/12 04:17:09 bsmith Exp curfman $";
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
  if (m1 != m2 || n1 != n2) SETERR(1,"Non conforming matrix add");
  vals = (Scalar *) MALLOC( n1*sizeof(Scalar) ); CHKPTR(vals);
  MatGetOwnershipRange(X,&start,&end);
  for ( i=start; i<end; i++ ) {
    MatGetRow(X,i,&ncols,&row,&val);
    for ( j=0; j<ncols; j++ ) {
      vals[j] = (*a)*val[j];
    }
    ierr = MatSetValues(Y,1,&i,ncols,row,vals,ADDVALUES); CHKERR(ierr);
    MatRestoreRow(X,i,&ncols,&row,&val);
  }
  FREE(vals);
  ierr = MatAssemblyBegin(Y,FINAL_ASSEMBLY); CHKERR(ierr);
  ierr = MatAssemblyEnd(Y,FINAL_ASSEMBLY); CHKERR(ierr);
  return 0;
}

/*$Id: ex14.c,v 1.5 1999/10/24 14:01:38 bsmith Exp bsmith $*/

/* 
   Tests OptionsGetScalar() for complex numbers
 */

#include "petsc.h"


#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int    ierr;
  Scalar a;

  PetscInitialize(&argc,&argv,(char *)0,0);
  ierr = OptionsGetScalar(PETSC_NULL,"-a",&a,PETSC_NULL);CHKERRA(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"Scalar a = %g + %gi\n",PetscRealPart(a),PetscImaginaryPart(a));CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
 

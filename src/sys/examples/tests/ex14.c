#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex14.c,v 1.3 1999/03/19 21:17:16 bsmith Exp bsmith $";
#endif

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
  ierr = PetscPrintf(PETSC_COMM_SELF,"Scalar a = %g + %gi\n",PetscReal(a),PetscImaginary(a));CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
 

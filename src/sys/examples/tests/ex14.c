#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex14.c,v 1.1 1998/11/25 16:16:32 bsmith Exp bsmith $";
#endif

/* 
   Tests OptionsGetScalar() for complex numbers
 */

#include "petsc.h"


int main(int argc,char **argv)
{
  int    ierr;
  Scalar a;

  PetscInitialize(&argc,&argv,(char *)0,0);
  ierr = OptionsGetScalar(PETSC_NULL,"-a",&a,PETSC_NULL);CHKERRA(ierr);
  printf("Scalar a = %g + %gi\n",PetscReal(a),PetscImaginary(a));
  PetscFinalize();
  return 0;
}
 

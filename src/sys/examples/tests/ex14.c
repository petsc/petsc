
static char help[] = "Tests PetscOptionsGetScalar() for complex numbers\n";

#include <petscsys.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  int         ierr;
  PetscScalar a;

  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = PetscOptionsGetScalar(NULL,"-a",&a,NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"Scalar a = %g + %gi\n",(double)PetscRealPart(a),(double)PetscImaginaryPart(a));CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}


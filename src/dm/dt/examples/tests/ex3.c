static char help[] = "Tests quadrature.\n\n";

#include <petscdt.h>

#undef __FUNCT__
#define __FUNCT__ "func1"
static void func1(PetscReal x, PetscReal *val)
{
  *val = x*log(1+x);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  const PetscInt  digits      = 12;
  const PetscReal analytic[1] = {0.250000000000000};
  const PetscReal epsilon     = 1.0e-12;
  PetscReal       integral;
  PetscErrorCode  ierr;

  ierr = PetscInitialize(&argc, &argv, PETSC_NULL, help);CHKERRQ(ierr);
  ierr = PetscDTTanhSinhIntegrate(func1, 0.0, 1.0, digits, &integral);CHKERRQ(ierr);
  if (PetscAbsReal(integral - analytic[0]) < epsilon) {ierr = PetscPrintf(PETSC_COMM_SELF, "The integral of func1 is correct\n");CHKERRQ(ierr);}
  else                                                {ierr = PetscPrintf(PETSC_COMM_SELF, "The integral of func1 is wrong: %15.15f (%15.15f)\n", integral, PetscAbsReal(integral - analytic[0]));CHKERRQ(ierr);}
  PetscFinalize();
  return 0;
}

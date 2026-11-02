static char help[] = "Ensure matching C and Fortran constants (C variant)";

#include <petscsys.h>

PETSC_INTERN void get_constants(PetscReal *pi, PetscReal *phi, PetscReal *sqrt2, PetscReal *e, PetscReal *max_real, PetscReal *min_real, PetscReal *machine_epsilon, PetscReal *sqrt_machine_epsilon, PetscReal *small, PetscReal *infinity, PetscReal *ninfinity);

int main(int argc, char **args)
{
  PetscReal pi, phi, sqrt2, e, max_real, min_real, machine_epsilon, sqrt_machine_epsilon, small, infinity, ninfinity;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PetscCall(PetscInitializeFortran());

  get_constants(&pi, &phi, &sqrt2, &e, &max_real, &min_real, &machine_epsilon, &sqrt_machine_epsilon, &small, &infinity, &ninfinity);

  PetscCheck(pi == PETSC_PI, MPI_COMM_WORLD, PETSC_ERR_PLIB, "PETSC_PI: value differs between C and Fortran");
  PetscCheck(phi == PETSC_PHI, MPI_COMM_WORLD, PETSC_ERR_PLIB, "PETSC_PHI: value differs between C and Fortran");
  PetscCheck(sqrt2 == PETSC_SQRT2, MPI_COMM_WORLD, PETSC_ERR_PLIB, "PETSC_SQRT2: value differs between C and Fortran");
  PetscCheck(e == PETSC_E, MPI_COMM_WORLD, PETSC_ERR_PLIB, "PETSC_E: value differs between C and Fortran");
  PetscCheck(max_real == PETSC_MAX_REAL, MPI_COMM_WORLD, PETSC_ERR_PLIB, "PETSC_MAX_REAL: value differs between C and Fortran");
  PetscCheck(min_real == PETSC_MIN_REAL, MPI_COMM_WORLD, PETSC_ERR_PLIB, "PETSC_MIN_REAL: value differs between C and Fortran");
  PetscCheck(machine_epsilon == PETSC_MACHINE_EPSILON, MPI_COMM_WORLD, PETSC_ERR_PLIB, "PETSC_MACHINE_EPSILON: value differs between C and Fortran");
  PetscCheck(sqrt_machine_epsilon == PETSC_SQRT_MACHINE_EPSILON, MPI_COMM_WORLD, PETSC_ERR_PLIB, "PETSC_SQRT_MACHINE_EPSILON: value differs between C and Fortran");
  PetscCheck(small == PETSC_SMALL, MPI_COMM_WORLD, PETSC_ERR_PLIB, "PETSC_SMALL: value differs between C and Fortran");
  PetscCheck(infinity == PETSC_INFINITY, MPI_COMM_WORLD, PETSC_ERR_PLIB, "PETSC_INFINITY: value differs between C and Fortran");
  PetscCheck(ninfinity == PETSC_NINFINITY, MPI_COMM_WORLD, PETSC_ERR_PLIB, "PETSC_NINFINITY: value differs between C and Fortran");
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
     depends: ex83f.F90
     requires: fortran

   test:
     output_file: output/empty.out

TEST*/

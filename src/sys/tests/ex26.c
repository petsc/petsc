static char help[] = "Tests repeated PetscInitialize/PetscFinalize calls.\n\n";

#include <petscsys.h>

int main(int argc, char **argv)
{
  int i, imax;
#if defined(PETSC_HAVE_ELEMENTAL)
  PetscBool initialized;
#endif

#if defined(PETSC_HAVE_MPIUNI)
  imax = 32;
#else
  imax = 1024;
#endif

  PetscCallMPI(MPI_Init(&argc, &argv));
#if defined(PETSC_HAVE_ELEMENTAL)
  PetscCall(PetscElementalInitializePackage());
  PetscCall(PetscElementalInitialized(&initialized));
  PetscCheck(initialized, MPI_COMM_WORLD, PETSC_ERR_PLIB, "Error in Elemental package processing");
#endif
  for (i = 0; i < imax; ++i) {
    PetscFunctionBeginUser;
    PetscCall(PetscInitialize(&argc, &argv, NULL, help));
    PetscCall(PetscFinalize());
#if defined(PETSC_HAVE_ELEMENTAL)
    // if Elemental is initialized outside of PETSc it should remain initialized
    PetscCall(PetscElementalInitialized(&initialized));
    PetscCheck(initialized, MPI_COMM_WORLD, PETSC_ERR_PLIB, "Error in Elemental package processing");
#endif
  }
#if defined(PETSC_HAVE_ELEMENTAL)
  PetscCall(PetscElementalFinalizePackage());
  PetscCall(PetscElementalInitialized(&initialized));
  PetscCheck(!initialized, MPI_COMM_WORLD, PETSC_ERR_PLIB, "Error in Elemental package processing");
  for (i = 0; i < 32; ++i) { /* increasing the upper bound will generate an error in Elemental */
    PetscFunctionBeginUser;
    PetscCall(PetscInitialize(&argc, &argv, NULL, help));
    PetscCall(PetscElementalInitialized(&initialized));
    PetscCheck(initialized, MPI_COMM_WORLD, PETSC_ERR_PLIB, "Error in Elemental package processing");
    PetscCheck(initialized, PETSC_COMM_WORLD, PETSC_ERR_LIB, "Uninitialized Elemental");
    PetscCall(PetscFinalize());
    PetscCall(PetscElementalInitialized(&initialized));
    // if Elemental is initialized inside of PETSc it should be uninitialized in PetscFinalize()
    PetscCheck(!initialized, MPI_COMM_WORLD, PETSC_ERR_PLIB, "Error in Elemental package processing");
  }
#endif
  return MPI_Finalize();
}

/*TEST

   test:
      requires: !saws
      output_file: output/empty.out

   test:
      requires: !saws
      suffix: 2
      nsize: 2
      output_file: output/empty.out

TEST*/

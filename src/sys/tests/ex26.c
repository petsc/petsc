static char help[] = "Tests repeated PetscInitialize/PetscFinalize calls.\n\n";

#include <petscsys.h>

int main(int argc, char **argv)
{
  int            i,imax;
#if defined(PETSC_HAVE_ELEMENTAL)
  PetscBool      initialized;
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
  if (!initialized) return 1;
#endif
  for (i = 0; i < imax; ++i) {
    PetscCall(PetscInitialize(&argc, &argv, (char*) 0, help));
    PetscCall(PetscFinalize());
#if defined(PETSC_HAVE_ELEMENTAL)
    PetscCall(PetscElementalInitialized(&initialized));
    if (!initialized) return PETSC_ERR_LIB;
#endif
  }
#if defined(PETSC_HAVE_ELEMENTAL)
  PetscCall(PetscElementalFinalizePackage());
  PetscCall(PetscElementalInitialized(&initialized));
  if (initialized) return 1;
  for (i = 0; i < 32; ++i) { /* increasing the upper bound will generate an error in Elemental */
    PetscCall(PetscInitialize(&argc, &argv, (char*) 0, help));
    PetscCall(PetscElementalInitialized(&initialized));
    PetscCheck(initialized,PETSC_COMM_WORLD, PETSC_ERR_LIB, "Uninitialized Elemental");
    PetscCall(PetscFinalize());
    PetscCall(PetscElementalInitialized(&initialized));
    if (initialized) return PETSC_ERR_LIB;
  }
#endif
  return MPI_Finalize();
}

/*TEST

   test:
      requires: !saws

   test:
      requires: !saws
      suffix: 2
      nsize: 2
      output_file: output/ex26_1.out

TEST*/

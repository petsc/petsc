static char help[] = "Tests repeated PetscInitialize/PetscFinalize calls.\n\n";

#include <petscsys.h>

int main(int argc, char **argv)
{
  int            i,imax;
#if defined(PETSC_HAVE_ELEMENTAL)
  PetscBool      initialized;
#endif
  PetscErrorCode ierr;

#if defined(PETSC_HAVE_MPIUNI)
  imax = 32;
#else
  imax = 1024;
#endif

  MPI_Init(&argc, &argv);
#if defined(PETSC_HAVE_ELEMENTAL)
  ierr = PetscElementalInitializePackage(); if (ierr) return ierr;
  ierr = PetscElementalInitialized(&initialized); if (ierr) return ierr;
  if (!initialized) return 1;
#endif
  for (i = 0; i < imax; ++i) {
    ierr = PetscInitialize(&argc, &argv, (char*) 0, help); if (ierr) return ierr;
    ierr = PetscFinalize(); if (ierr) return ierr;
#if defined(PETSC_HAVE_ELEMENTAL)
    ierr = PetscElementalInitialized(&initialized); if (ierr) return ierr;
    if (!initialized) return PETSC_ERR_LIB;
#endif
  }
#if defined(PETSC_HAVE_ELEMENTAL)
  ierr = PetscElementalFinalizePackage(); if (ierr) return ierr;
  ierr = PetscElementalInitialized(&initialized); if (ierr) return ierr;
  if (initialized) return 1;
  for (i = 0; i < 32; ++i) { /* increasing the upper bound will generate an error in Elemental */
    ierr = PetscInitialize(&argc, &argv, (char*) 0, help); if (ierr) return ierr;
    CHKERRQ(PetscElementalInitialized(&initialized));
    PetscCheckFalse(!initialized,PETSC_COMM_WORLD, PETSC_ERR_LIB, "Uninitialized Elemental");
    ierr = PetscFinalize(); if (ierr) return ierr;
    ierr = PetscElementalInitialized(&initialized); if (ierr) return ierr;
    if (initialized) return PETSC_ERR_LIB;
  }
#endif
  MPI_Finalize();
  return ierr;
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

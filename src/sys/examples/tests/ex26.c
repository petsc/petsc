static char help[] = "Tests repeated PetscInitialize/PetscFinalize calls.\n\n";

#include <petscsys.h>

int main(int argc, char **argv)
{
  int i,imax;
  PetscErrorCode ierr;

#if defined(PETSC_HAVE_MPIUNI)
  imax = 32;
#else
  imax = 1024;
#endif

  MPI_Init(&argc, &argv);
  for (i = 0; i < imax; ++i) {
    ierr = PetscInitialize(&argc, &argv, (char*) 0, help); if (ierr) return ierr;
    ierr = PetscFinalize(); if (ierr) return ierr;
  }
  MPI_Finalize();
  return ierr;
}

/*TEST

   test:

   test:
      suffix: 2
      nsize: 2
      output_file: output/ex26_1.out

TEST*/

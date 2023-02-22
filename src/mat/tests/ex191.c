static char help[] = "Tests MatLoad() for dense matrix with uneven dimensions set in program\n\n";

#include <petscmat.h>

int main(int argc, char **args)
{
  Mat          A;
  PetscViewer  fd;
  PetscMPIInt  rank;
  PetscScalar *Av;
  PetscInt     i;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  PetscCall(MatCreateDense(PETSC_COMM_WORLD, 6, 6, 12, 12, NULL, &A));
  PetscCall(MatDenseGetArrayAndMemType(A, &Av, NULL));
  for (i = 0; i < 6 * 12; i++) Av[i] = (PetscScalar)i;
  PetscCall(MatDenseRestoreArrayAndMemType(A, &Av));

  /* Load matrices */
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, "ex191matrix", FILE_MODE_WRITE, &fd));
  PetscCall(PetscViewerPushFormat(fd, PETSC_VIEWER_NATIVE));
  PetscCall(MatView(A, fd));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscViewerPopFormat(fd));
  PetscCall(PetscViewerDestroy(&fd));

  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetType(A, MATDENSE));
  if (rank == 0) {
    PetscCall(MatSetSizes(A, 4, PETSC_DETERMINE, PETSC_DETERMINE, PETSC_DETERMINE));
  } else {
    PetscCall(MatSetSizes(A, 8, PETSC_DETERMINE, PETSC_DETERMINE, PETSC_DETERMINE));
  }
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, "ex191matrix", FILE_MODE_READ, &fd));
  PetscCall(MatLoad(A, fd));
  PetscCall(PetscViewerDestroy(&fd));
  PetscCall(MatView(A, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_ASCII_INFO_DETAIL));
  PetscCall(MatView(A, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 2
      filter: grep -v alloced

TEST*/

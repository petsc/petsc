static char help[] = "Tests MatLoad() for dense matrix with uneven dimensions set in program\n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            A;
  PetscViewer    fd;
  PetscMPIInt    rank;
  PetscScalar    *Av;
  PetscInt       i;

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  CHKERRQ(MatCreateDense(PETSC_COMM_WORLD,6,6,12,12,NULL,&A));
  CHKERRQ(MatDenseGetArray(A,&Av));
  for (i=0; i<6*12; i++) Av[i] = (PetscScalar) i;
  CHKERRQ(MatDenseRestoreArray(A,&Av));

  /* Load matrices */
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"ex191matrix",FILE_MODE_WRITE,&fd));
  CHKERRQ(PetscViewerPushFormat(fd,PETSC_VIEWER_NATIVE));
  CHKERRQ(MatView(A,fd));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(PetscViewerPopFormat(fd));
  CHKERRQ(PetscViewerDestroy(&fd));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetType(A,MATDENSE));
  if (rank == 0) {
    CHKERRQ(MatSetSizes(A, 4, PETSC_DETERMINE, PETSC_DETERMINE,PETSC_DETERMINE));
  } else {
    CHKERRQ(MatSetSizes(A, 8, PETSC_DETERMINE, PETSC_DETERMINE,PETSC_DETERMINE));
  }
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"ex191matrix",FILE_MODE_READ,&fd));
  CHKERRQ(MatLoad(A,fd));
  CHKERRQ(PetscViewerDestroy(&fd));
  CHKERRQ(MatView(A,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO_DETAIL));
  CHKERRQ(MatView(A,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 2
      filter: grep -v alloced

TEST*/

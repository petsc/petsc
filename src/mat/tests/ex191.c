static char help[] = "Tests MatLoad() for dense matrix with uneven dimensions set in program\n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            A;
  PetscViewer    fd;
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  PetscScalar    *Av;
  PetscInt       i;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);

  ierr = MatCreateDense(PETSC_COMM_WORLD,6,6,12,12,NULL,&A);CHKERRQ(ierr);
  ierr = MatDenseGetArray(A,&Av);CHKERRQ(ierr);
  for (i=0; i<6*12; i++) Av[i] = (PetscScalar) i;
  ierr = MatDenseRestoreArray(A,&Av);CHKERRQ(ierr);

  /* Load matrices */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"ex191matrix",FILE_MODE_WRITE,&fd);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(fd,PETSC_VIEWER_NATIVE);CHKERRQ(ierr);
  ierr = MatView(A,fd);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetType(A,MATDENSE);CHKERRQ(ierr);
  if (rank == 0) {
    ierr = MatSetSizes(A, 4, PETSC_DETERMINE, PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  } else {
    ierr = MatSetSizes(A, 8, PETSC_DETERMINE, PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  }
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"ex191matrix",FILE_MODE_READ,&fd);CHKERRQ(ierr);
  ierr = MatLoad(A,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);
  ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO_DETAIL);CHKERRQ(ierr);
  ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      nsize: 2
      filter: grep -v alloced

TEST*/

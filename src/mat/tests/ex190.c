static char help[] = "Tests MatLoad() with uneven dimensions set in program\n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            A;
  PetscViewer    fd;
  char           file[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;
  PetscBool      flg;
  PetscMPIInt    rank;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);

  /* Determine files from which we read the matrix */
  ierr = PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),&flg);CHKERRQ(ierr);
  PetscCheckFalse(!flg,PETSC_COMM_WORLD,PETSC_ERR_USER,"Must indicate binary file with the -f");

  /* Load matrices */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetBlockSize(A,2);CHKERRQ(ierr);
  if (rank == 0) {
    ierr = MatSetSizes(A, 4, PETSC_DETERMINE, PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  } else {
    ierr = MatSetSizes(A, 8, PETSC_DETERMINE, PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  }
  ierr = MatLoad(A,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

      test:
         nsize: 2
         args: -mat_type aij -mat_view -f ${wPETSC_DIR}/share/petsc/datafiles/matrices/ns-real-int32-float64
         requires: double !complex !defined(PETSC_USE_64BIT_INDICES)

      test:
         suffix: 2
         nsize: 2
         args: -mat_type baij -mat_view -f ${wPETSC_DIR}/share/petsc/datafiles/matrices/ns-real-int32-float64
         requires: double !complex !defined(PETSC_USE_64BIT_INDICES)

      test:
         suffix: 3
         nsize: 2
         args: -mat_type sbaij -mat_view -f ${wPETSC_DIR}/share/petsc/datafiles/matrices/ns-real-int32-float64
         requires: double !complex !defined(PETSC_USE_64BIT_INDICES)

TEST*/

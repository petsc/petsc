
static char help[] = "Tests MatLoad() MatView() for MPIBAIJ.\n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            A,B;
  PetscErrorCode ierr;
  char           file[PETSC_MAX_PATH_LEN];
  PetscBool      flg;
  PetscViewer    fd;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_USER,"Must indicate binary file with the -f option");

  /*
     Open binary file.  Note that we use FILE_MODE_READ to indicate
     reading from this file.
  */
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd));

  /*
     Load the matrix; then destroy the viewer.
  */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatLoad(A,fd));
  CHKERRQ(PetscViewerDestroy(&fd));

  /*
     Open another binary file.  Note that we use FILE_MODE_WRITE to indicate writing to the file
  */
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"fileoutput",FILE_MODE_WRITE,&fd));
  CHKERRQ(PetscViewerBinarySetFlowControl(fd,3));
  /*
     Save the matrix and vector; then destroy the viewer.
  */
  CHKERRQ(MatView(A,fd));
  CHKERRQ(PetscViewerDestroy(&fd));

  /* load the new matrix */
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"fileoutput",FILE_MODE_READ,&fd));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&B));
  CHKERRQ(MatSetFromOptions(B));
  CHKERRQ(MatLoad(B,fd));
  CHKERRQ(PetscViewerDestroy(&fd));

  CHKERRQ(MatEqual(A,B,&flg));
  if (flg) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Matrices are equal\n"));
  } else {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Matrices are not equal\n"));
  }

  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&B));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      nsize: 3
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/cfd.2.100 -mat_view ascii::ascii_info

   test:
      suffix: 2
      nsize: 5
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/cfd.2.100 -mat_view ascii::ascii_info

   test:
      suffix: 3
      nsize: 7
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/cfd.2.100 -mat_view ascii::ascii_info

   test:
      suffix: 4
      nsize: 3
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/cfd.2.100 -mat_view ascii::ascii_info -mat_type baij

   test:
      suffix: 5
      nsize: 5
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/cfd.2.100 -mat_view ascii::ascii_info -mat_type baij

   test:
      suffix: 6
      nsize: 7
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/cfd.2.100 -mat_view ascii::ascii_info -mat_type baij

TEST*/

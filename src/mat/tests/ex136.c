
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
  ierr = PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Must indicate binary file with the -f option");

  /*
     Open binary file.  Note that we use FILE_MODE_READ to indicate
     reading from this file.
  */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd);CHKERRQ(ierr);

  /*
     Load the matrix; then destroy the viewer.
  */
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatLoad(A,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);

  /*
     Open another binary file.  Note that we use FILE_MODE_WRITE to indicate writing to the file
  */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"fileoutput",FILE_MODE_WRITE,&fd);CHKERRQ(ierr);
  ierr = PetscViewerBinarySetFlowControl(fd,3);CHKERRQ(ierr);
  /*
     Save the matrix and vector; then destroy the viewer.
  */
  ierr = MatView(A,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);

  /* load the new matrix */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"fileoutput",FILE_MODE_READ,&fd);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&B);CHKERRQ(ierr);
  ierr = MatSetFromOptions(B);CHKERRQ(ierr);
  ierr = MatLoad(B,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);

  ierr = MatEqual(A,B,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Matrices are equal\n");CHKERRQ(ierr);
  } else {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Matrices are not equal\n");CHKERRQ(ierr);
  }

  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      nsize: 3
      requires: datafilespath !complex double !define(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/cfd.2.100 -mat_view ascii::ascii_info

   test:
      suffix: 2
      nsize: 5
      requires: datafilespath !complex double !define(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/cfd.2.100 -mat_view ascii::ascii_info

   test:
      suffix: 3
      nsize: 7
      requires: datafilespath !complex double !define(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/cfd.2.100 -mat_view ascii::ascii_info

   test:
      suffix: 4
      nsize: 3
      requires: datafilespath !complex double !define(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/cfd.2.100 -mat_view ascii::ascii_info -mat_type baij

   test:
      suffix: 5
      nsize: 5
      requires: datafilespath !complex double !define(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/cfd.2.100 -mat_view ascii::ascii_info -mat_type baij

   test:
      suffix: 6
      nsize: 7
      requires: datafilespath !complex double !define(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/cfd.2.100 -mat_view ascii::ascii_info -mat_type baij

TEST*/

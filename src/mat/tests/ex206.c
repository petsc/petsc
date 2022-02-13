static char help[] = "Reads binary matrix - twice\n";

#include <petscmat.h>
int main(int argc,char **args)
{
  Mat               A;
  PetscViewer       fd;                        /* viewer */
  char              file[PETSC_MAX_PATH_LEN];  /* input file name */
  PetscErrorCode    ierr;
  PetscBool         flg;

  PetscInitialize(&argc,&args,(char*)0,help);

  ierr = PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),&flg);CHKERRQ(ierr);
  PetscCheckFalse(!flg,PETSC_COMM_WORLD,PETSC_ERR_USER,"Must indicate binary file with the -f option");

  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "First MatLoad! \n");CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd);CHKERRQ(ierr);
  ierr = MatLoad(A,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);
  ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = PetscOptionsGetString(NULL,NULL,"-f2",file,sizeof(file),&flg);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Second MatLoad! \n");CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd);CHKERRQ(ierr);
  ierr = MatLoad(A,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);
  ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      suffix: aij_1
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/small -mat_type aij -mat_block_size 1
      filter: grep -v Mat_

   test:
      suffix: aij_2
      nsize: 2
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/small -mat_type aij -mat_block_size 1
      filter: grep -v Mat_

   test:
      suffix: aij_2_d
      nsize: 2
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/small -f2 ${DATAFILESPATH}/matrices/smallbs2 -mat_type aij -mat_block_size 1
      filter: grep -v Mat_

   test:
      suffix: baij_1_2
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/small -mat_type baij -mat_block_size 2
      filter: grep -v Mat_

   test:
      suffix: baij_2_1_d
      nsize: 2
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/small -f2 ${DATAFILESPATH}/matrices/smallbs2 -mat_type baij -mat_block_size 1
      filter: grep -v Mat_

   test:
      suffix: baij_2_2
      nsize: 2
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/small -mat_type baij -mat_block_size 2
      filter: grep -v Mat_

   test:
      suffix: baij_2_2_d
      nsize: 2
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/small -f2 ${DATAFILESPATH}/matrices/smallbs2 -mat_type baij -mat_block_size 2
      filter: grep -v Mat_

   test:
      suffix: sbaij_1_1
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/small -mat_type sbaij -mat_block_size 1
      filter: grep -v Mat_

   test:
      suffix: sbaij_1_2
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/small -mat_type sbaij -mat_block_size 2
      filter: grep -v Mat_

   test:
      suffix: sbaij_2_1_d
      nsize: 2
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/small -f2 ${DATAFILESPATH}/matrices/smallbs2 -mat_type sbaij -mat_block_size 1
      filter: grep -v Mat_

   test:
      suffix: sbaij_2_2
      nsize: 2
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/small -mat_type sbaij -mat_block_size 2
      filter: grep -v Mat_

TEST*/

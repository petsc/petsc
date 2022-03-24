static char help[] = "Reads binary matrix - twice\n";

#include <petscmat.h>
int main(int argc,char **args)
{
  Mat               A;
  PetscViewer       fd;                        /* viewer */
  char              file[PETSC_MAX_PATH_LEN];  /* input file name */
  PetscBool         flg;

  PetscInitialize(&argc,&args,(char*)0,help);

  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_USER,"Must indicate binary file with the -f option");

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "First MatLoad! \n"));
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd));
  CHKERRQ(MatLoad(A,fd));
  CHKERRQ(PetscViewerDestroy(&fd));
  CHKERRQ(MatView(A,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-f2",file,sizeof(file),&flg));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "Second MatLoad! \n"));
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd));
  CHKERRQ(MatLoad(A,fd));
  CHKERRQ(PetscViewerDestroy(&fd));
  CHKERRQ(MatView(A,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(MatDestroy(&A));
  CHKERRQ(PetscFinalize());
  return 0;
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

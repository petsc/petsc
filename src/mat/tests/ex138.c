
static char help[] = "Tests MatGetColumnNorms() for matrix read from file.";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            A;
  PetscErrorCode ierr;
  PetscReal      *reductions;
  char           file[PETSC_MAX_PATH_LEN];
  PetscBool      flg;
  PetscViewer    fd;
  PetscInt       n;
  PetscMPIInt    rank;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Must indicate binary file with the -f option");
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatLoad(A,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);

  ierr = MatGetSize(A,NULL,&n);CHKERRQ(ierr);
  ierr = PetscMalloc1(n,&reductions);CHKERRQ(ierr);

  ierr = MatGetColumnNorms(A,NORM_2,reductions);CHKERRQ(ierr);
  if (!rank) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"NORM_2:\n");CHKERRQ(ierr);
    ierr = PetscRealView(n,reductions,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  }

  ierr = MatGetColumnNorms(A,NORM_1,reductions);CHKERRQ(ierr);
  if (!rank) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"NORM_1:\n");CHKERRQ(ierr);
    ierr = PetscRealView(n,reductions,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  }

  ierr = MatGetColumnNorms(A,NORM_INFINITY,reductions);CHKERRQ(ierr);
  if (!rank) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"NORM_INFINITY:\n");CHKERRQ(ierr);
    ierr = PetscRealView(n,reductions,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  }

  ierr = MatGetColumnReductions(A,REDUCTION_SUM,reductions);CHKERRQ(ierr);
  if (!rank) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"REDUCTION_SUM:\n");CHKERRQ(ierr);
    ierr = PetscRealView(n,reductions,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  }

  ierr = MatGetColumnReductions(A,REDUCTION_MEAN,reductions);CHKERRQ(ierr);
  if (!rank) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"REDUCTION_MEAN:\n");CHKERRQ(ierr);
    ierr = PetscRealView(n,reductions,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  }

  ierr = PetscFree(reductions);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      suffix: 1
      nsize: 2
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/small -mat_type aij
      output_file: output/ex138.out

   test:
      suffix: 2
      nsize: {{1 2}}
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/small -mat_type baij -matload_block_size {{2 3}}
      output_file: output/ex138.out

TEST*/

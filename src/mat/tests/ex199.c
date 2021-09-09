
static char help[] = "Tests the different MatColoring implementatons.\n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            C;
  PetscErrorCode ierr;
  PetscViewer    viewer;
  char           file[128];
  PetscBool      flg;
  MatColoring    ctx;
  ISColoring     coloring;
  PetscMPIInt    size;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);

  ierr = PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Must use -f filename to load sparse matrix");
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&C);CHKERRQ(ierr);
  ierr = MatLoad(C,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  ierr = MatColoringCreate(C,&ctx);CHKERRQ(ierr);
  ierr = MatColoringSetFromOptions(ctx);CHKERRQ(ierr);
  ierr = MatColoringApply(ctx,&coloring);CHKERRQ(ierr);
  ierr = MatColoringTest(ctx,coloring);CHKERRQ(ierr);
  if (size == 1) {
    /* jp, power and greedy have bug -- need to be fixed */
    ierr = MatISColoringTest(C,coloring);CHKERRQ(ierr);
  }

  /* Free data structures */
  ierr = ISColoringDestroy(&coloring);CHKERRQ(ierr);
  ierr = MatColoringDestroy(&ctx);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      nsize: {{3}}
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/arco1 -mat_coloring_type {{ jp power natural greedy}} -mat_coloring_distance {{ 1 2}}

   test:
      suffix: 2
      nsize: {{1 2}}
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/arco1 -mat_coloring_type {{  sl lf id }} -mat_coloring_distance 2
      output_file: output/ex199_1.out

TEST*/

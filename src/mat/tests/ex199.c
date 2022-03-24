
static char help[] = "Tests the different MatColoring implementatons.\n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            C;
  PetscViewer    viewer;
  char           file[128];
  PetscBool      flg;
  MatColoring    ctx;
  ISColoring     coloring;
  PetscMPIInt    size;

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_USER,"Must use -f filename to load sparse matrix");
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&viewer));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&C));
  CHKERRQ(MatLoad(C,viewer));
  CHKERRQ(PetscViewerDestroy(&viewer));

  CHKERRQ(MatColoringCreate(C,&ctx));
  CHKERRQ(MatColoringSetFromOptions(ctx));
  CHKERRQ(MatColoringApply(ctx,&coloring));
  CHKERRQ(MatColoringTest(ctx,coloring));
  if (size == 1) {
    /* jp, power and greedy have bug -- need to be fixed */
    CHKERRQ(MatISColoringTest(C,coloring));
  }

  /* Free data structures */
  CHKERRQ(ISColoringDestroy(&coloring));
  CHKERRQ(MatColoringDestroy(&ctx));
  CHKERRQ(MatDestroy(&C));
  CHKERRQ(PetscFinalize());
  return 0;
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

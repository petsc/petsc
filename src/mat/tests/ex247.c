
static char help[] = "Tests MATCENTERING matrix type.\n\n";

#include <petscmat.h>

int main(int argc,char **argv)
{
  PetscInt       n;
  Mat            C;
  Vec            x,y;
  PetscReal      norm;
  PetscMPIInt    size;

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  /* Create a parallel vector with 10*size total entries, and fill it with 1s. */
  n = 10*size;
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&x));
  CHKERRQ(VecSetSizes(x,PETSC_DECIDE,n));
  CHKERRQ(VecSetFromOptions(x));
  CHKERRQ(VecSet(x,1.0));

  /* Create a corresponding n x n centering matrix and use it to create a mean-centered y = C * x. */
  CHKERRQ(VecDuplicate(x,&y));
  CHKERRQ(MatCreateCentering(PETSC_COMM_WORLD,PETSC_DECIDE,n,&C));
  CHKERRQ(MatMult(C,x,y));

  /* Verify that the centered vector y has norm 0. */
  CHKERRQ(VecNorm(y,NORM_2,&norm));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Vector norm after MatMult() with centering matrix applied to vector of ones is %f.\n",(double)norm));

  /* Now repeat, but using MatMultTranspose(). */
  CHKERRQ(MatMultTranspose(C,x,y));
  CHKERRQ(VecNorm(y,NORM_2,&norm));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Vector norm after MatMultTranspose() with centering matrix applied to vector of ones is %f.\n",(double)norm));

  /* Clean up. */
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&y));
  CHKERRQ(MatDestroy(&C));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

    test:
      suffix: 1
      nsize: 1
      output_file: output/ex247.out

    test:
      suffix: 2
      nsize: 2
      output_file: output/ex247.out

TEST*/


static char help[] = "Tests MATCENTERING matrix type.\n\n";

#include <petscmat.h>

int main(int argc,char **argv)
{
  PetscInt       n;
  Mat            C;
  Vec            x,y;
  PetscReal      norm;
  PetscMPIInt    size;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  /* Create a parallel vector with 10*size total entries, and fill it with 1s. */
  n = 10*size;
  PetscCall(VecCreate(PETSC_COMM_WORLD,&x));
  PetscCall(VecSetSizes(x,PETSC_DECIDE,n));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecSet(x,1.0));

  /* Create a corresponding n x n centering matrix and use it to create a mean-centered y = C * x. */
  PetscCall(VecDuplicate(x,&y));
  PetscCall(MatCreateCentering(PETSC_COMM_WORLD,PETSC_DECIDE,n,&C));
  PetscCall(MatMult(C,x,y));

  /* Verify that the centered vector y has norm 0. */
  PetscCall(VecNorm(y,NORM_2,&norm));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Vector norm after MatMult() with centering matrix applied to vector of ones is %f.\n",(double)norm));

  /* Now repeat, but using MatMultTranspose(). */
  PetscCall(MatMultTranspose(C,x,y));
  PetscCall(VecNorm(y,NORM_2,&norm));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Vector norm after MatMultTranspose() with centering matrix applied to vector of ones is %f.\n",(double)norm));

  /* Clean up. */
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(MatDestroy(&C));
  PetscCall(PetscFinalize());
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

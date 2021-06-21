
static char help[] = "Tests MATCENTERING matrix type.\n\n";

#include <petscmat.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       n;
  Mat            C;
  Vec            x,y;
  PetscReal      norm;
  PetscMPIInt    size;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);

  /* Create a parallel vector with 10*size total entries, and fill it with 1s. */
  n = 10*size;
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecSet(x,1.0);CHKERRQ(ierr);

  /* Create a corresponding n x n centering matrix and use it to create a mean-centered y = C * x. */
  ierr = VecDuplicate(x,&y);CHKERRQ(ierr);
  ierr = MatCreateCentering(PETSC_COMM_WORLD,PETSC_DECIDE,n,&C);CHKERRQ(ierr);
  ierr = MatMult(C,x,y);CHKERRQ(ierr);

  /* Verify that the centered vector y has norm 0. */
  ierr = VecNorm(y,NORM_2,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Vector norm after MatMult() with centering matrix applied to vector of ones is %f.\n",(double)norm);CHKERRQ(ierr);

  /* Now repeat, but using MatMultTranspose(). */
  ierr = MatMultTranspose(C,x,y);CHKERRQ(ierr);
  ierr = VecNorm(y,NORM_2,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Vector norm after MatMultTranspose() with centering matrix applied to vector of ones is %f.\n",(double)norm);CHKERRQ(ierr);

  /* Clean up. */
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
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

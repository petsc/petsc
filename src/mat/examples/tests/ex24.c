
static char help[] = "Tests the different MatColoring implementatons and ISColoringTestValid() \n\
                      Modifed from the code contributed by Ali Berk Kahraman. \n\n";
#include <petscmat.h>

PetscErrorCode FormJacobian(Mat A,void *ctx)
{
  PetscErrorCode ierr;
  PetscInt       M,ownbegin,ownend,i,j;
  PetscScalar    dummy=0.0;

  PetscFunctionBeginUser;
  ierr = MatGetSize(A,&M,NULL);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&ownbegin,&ownend);CHKERRQ(ierr);

  for (i=ownbegin; i<ownend; i++) {
    for(j=i-3; j<i+3; j++) {
      if (j >= 0 && j < M) {
        ierr = MatSetValues(A,1,&i,1,&j,&dummy,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char *argv[])
{
  PetscErrorCode ierr;
  Mat            J;
  Vec            solution,residual;
  PetscMPIInt    size;
  PetscInt       M=8;
  ISColoring     iscoloring;
  MatColoring    coloring;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);

  ierr= MatCreate(PETSC_COMM_WORLD,&J);CHKERRQ(ierr);
  ierr= MatSetSizes(J, PETSC_DECIDE, PETSC_DECIDE, M, M);CHKERRQ(ierr);
  ierr= MatSetFromOptions(J);CHKERRQ(ierr);
  ierr= MatSetUp(J);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&solution);CHKERRQ(ierr);
  ierr = VecSetSizes(solution,PETSC_DECIDE,M);CHKERRQ(ierr);
  ierr = VecSetFromOptions(solution);CHKERRQ(ierr);
  ierr = VecDuplicate(solution,&residual);CHKERRQ(ierr);

  ierr = FormJacobian(J,NULL);CHKERRQ(ierr);
  ierr = MatView(J,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /*
    Color the matrix, i.e. determine groups of columns that share no common
    rows. These columns in the Jacobian can all be computed simultaneously.
   */
  ierr = MatColoringCreate(J, &coloring);CHKERRQ(ierr);
  ierr = MatColoringSetType(coloring,MATCOLORINGGREEDY);CHKERRQ(ierr);
  ierr = MatColoringSetDistance(coloring,2);CHKERRQ(ierr);
  ierr = MatColoringSetFromOptions(coloring);CHKERRQ(ierr);
  ierr = MatColoringApply(coloring, &iscoloring);CHKERRQ(ierr);

  if (size == 1) {
    ierr = MatISColoringTest(J,iscoloring);CHKERRQ(ierr);
  }

  ierr = ISColoringDestroy(&iscoloring);CHKERRQ(ierr);
  ierr = MatColoringDestroy(&coloring);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = VecDestroy(&solution);CHKERRQ(ierr);
  ierr = VecDestroy(&residual);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

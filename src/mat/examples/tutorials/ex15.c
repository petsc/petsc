static char help[] = "Example of using graph partitioning to segment an image\n\n";

/*T
   Concepts: Mat^mat partitioning
   Concepts: Mat^image segmentation
   Processors: n
T*/

#include <petscmat.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **args)
{
  Mat             A;
  MatPartitioning part;
  IS              is;
  PetscInt        r,N = 10, start, end;
  PetscErrorCode  ierr;
  
  ierr = PetscInitialize(&argc, &args, (char *) 0, help);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL, "-N", &N, PETSC_NULL);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD, &A);CHKERRQ(ierr);
  ierr = MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, N, N);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(A, 3, PETSC_NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(A, 3, PETSC_NULL, 2, PETSC_NULL);CHKERRQ(ierr);

  // Create a linear mesh
  ierr = MatGetOwnershipRange(A, &start, &end);CHKERRQ(ierr);
  for(r = start; r < end; ++r) {
    if (r == 0) {
      PetscInt    cols[2];
      PetscScalar vals[2];

      cols[0] = r;   cols[1] = r+1;
      vals[0] = 1.0; vals[1] = 1.0;
      ierr = MatSetValues(A, 1, &r, 2, cols, vals, INSERT_VALUES);CHKERRQ(ierr);
    } else if (r == N-1) {
      PetscInt    cols[2];
      PetscScalar vals[2];

      cols[0] = r-1; cols[1] = r;
      vals[0] = 1.0; vals[1] = 1.0;
      ierr = MatSetValues(A, 1, &r, 2, cols, vals, INSERT_VALUES);CHKERRQ(ierr);
    } else {
      PetscInt    cols[3];
      PetscScalar vals[3];

      cols[0] = r-1; cols[1] = r;   cols[2] = r+1;
      vals[0] = 1.0; vals[1] = 1.0; vals[2] = 1.0;
      ierr = MatSetValues(A, 1, &r, 3, cols, vals, INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatPartitioningCreate(PETSC_COMM_WORLD, &part);CHKERRQ(ierr);
  ierr = MatPartitioningSetAdjacency(part, A);CHKERRQ(ierr);
  ierr = MatPartitioningSetFromOptions(part);CHKERRQ(ierr);
  //ierr = MatPartitioningSetVertexWeights(part, const PetscInt weights[]);CHKERRQ(ierr);
  //ierr = MatPartitioningSetPartitionWeights(part,const PetscReal weights[]);CHKERRQ(ierr);
  ierr = MatPartitioningApply(part, &is);CHKERRQ(ierr);
  ierr = ISView(is, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);
  ierr = MatPartitioningDestroy(&part);CHKERRQ(ierr);

  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}

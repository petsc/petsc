static char help[] = "Test CPU/GPU memory leaks, MatMult and MatMultTransposeAdd during successive matrix assemblies\n\n";

#include <petscmat.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank,size;
  Mat            A;
  PetscInt       i,j,k,n=3,vstart,rstart,rend,margin;
  Vec            x,y;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,n,n,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);

  ierr = MatMPIAIJSetPreallocation(A,n,NULL,0,NULL);CHKERRQ(ierr);
  ierr = MatSetOption(A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);
  ierr = MatCreateVecs(A,&x,&y);CHKERRQ(ierr);
  ierr = VecSet(x,1.0);CHKERRQ(ierr);

  /*
    Matrix A only has nonzeros in the diagonal block, which is of size 3x3.
    We do three successive assemblies on A. The first two have the same non-zero
    pattern but different values, and the third breaks the non-zero pattern. The
    first two assemblies have enough zero-rows that triggers compressed-row storage
    in MATAIJ and MATAIJCUSPARSE.

    These settings are used to test memory management and correctness in MatMult
    and MatMultTransposeAdd.
  */

  for (k=0; k<3; k++) { /* Three assemblies */
    vstart = (size*k + rank)*n*n+1;
    margin = (k == 2)? 0 : 2; /* Create two zero-rows in the first two assemblies */
    for (i=rstart; i<rend-margin; i++) {
      for (j=rstart; j<rend; j++) {
        ierr = MatSetValue(A,i,j,(PetscScalar)vstart,INSERT_VALUES);CHKERRQ(ierr);
        vstart++;
      }
    }
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatMult(A,x,y);CHKERRQ(ierr);
    ierr = MatMultTransposeAdd(A,x,y,y);CHKERRQ(ierr); /* y[i] = sum of row i and column i of A */
    ierr = VecView(y,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = PetscFinalize();

  /* Uncomment this line if you want to use "cuda-memcheck --leaf-check full" to check this program */
  /*cudaDeviceReset();*/
  return ierr;
}

/*TEST

   testset:
     nsize: 2
     output_file: output/ex236_1.out
     filter: grep -v type

     test:
       args: -mat_type aij

     test:
       requires: cuda
       suffix: cuda
       args: -mat_type aijcusparse
TEST*/

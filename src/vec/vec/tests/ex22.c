
static char help[] = "Scatters from a parallel vector to a parallel vector.\n\n";

#include <petscvec.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       n = 5,N,i;
  PetscMPIInt    size,rank;
  PetscScalar    value,zero = 0.0;
  Vec            x,y;
  IS             is1,is2;
  VecScatter     ctx = 0;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);

  /* create two vectors */
  N    = size*n;
  ierr = VecCreate(PETSC_COMM_WORLD,&y);CHKERRQ(ierr);
  ierr = VecSetSizes(y,n,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(y);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,n,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);

  /* create two index sets */
  ierr = ISCreateStride(PETSC_COMM_WORLD,n,n*rank,1,&is1);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_WORLD,n,(n*(rank+1))%N,1,&is2);CHKERRQ(ierr);

  /* fill local part of parallel vector x */
  value = (PetscScalar)(rank+1);
  for (i=n*rank; i<n*(rank+1); i++) {
    ierr = VecSetValues(x,1,&i,&value,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(x);CHKERRQ(ierr);

  ierr = VecSet(y,zero);CHKERRQ(ierr);

  ierr = VecScatterCreate(x,is1,y,is2,&ctx);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx,x,y,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx,x,y,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&ctx);CHKERRQ(ierr);

  ierr = VecView(y,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = ISDestroy(&is1);CHKERRQ(ierr);
  ierr = ISDestroy(&is2);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}



/*TEST

   testset:
      nsize: 4
      output_file: output/ex22_1.out
      filter: grep -v "  type:"
      test:
        suffix: standard
        args: -vec_type standard
      test:
        requires: cuda
        suffix: cuda
        args: -vec_type cuda
      test:
        requires: viennacl
        suffix:  viennacl
        args: -vec_type viennacl
      test:
        requires: kokkos_kernels
        suffix: kokkos
        args: -vec_type kokkos

TEST*/

static char help[]= "Test VecScatterCreateToZero, VecScatterCreateToAll\n\n";

#include <petscvec.h>
int main(int argc,char **argv)
{
  PetscErrorCode     ierr;
  PetscInt           i,N=10,low,high;
  PetscMPIInt        size,rank;
  Vec                x,y;
  VecScatter         vscat;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&x));
  CHKERRQ(VecSetFromOptions(x));
  CHKERRQ(VecSetSizes(x,PETSC_DECIDE,N));
  CHKERRQ(VecGetOwnershipRange(x,&low,&high));
  CHKERRQ(PetscObjectSetName((PetscObject)x,"x"));

  /*-------------------------------------*/
  /*       VecScatterCreateToZero        */
  /*-------------------------------------*/

  /* MPI vec x = [0, 1, 2, .., N-1] */
  for (i=low; i<high; i++) CHKERRQ(VecSetValue(x,i,(PetscScalar)i,INSERT_VALUES));
  CHKERRQ(VecAssemblyBegin(x));
  CHKERRQ(VecAssemblyEnd(x));

  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nTesting VecScatterCreateToZero\n"));
  CHKERRQ(VecScatterCreateToZero(x,&vscat,&y));
  CHKERRQ(PetscObjectSetName((PetscObject)y,"y"));

  /* Test PetscSFBcastAndOp with op = MPI_REPLACE, which does y = x on rank 0 */
  CHKERRQ(VecScatterBegin(vscat,x,y,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(vscat,x,y,INSERT_VALUES,SCATTER_FORWARD));
  if (rank == 0) CHKERRQ(VecView(y,PETSC_VIEWER_STDOUT_SELF));

  /* Test PetscSFBcastAndOp with op = MPI_SUM, which does y += x */
  CHKERRQ(VecScatterBegin(vscat,x,y,ADD_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(vscat,x,y,ADD_VALUES,SCATTER_FORWARD));
  if (rank == 0) CHKERRQ(VecView(y,PETSC_VIEWER_STDOUT_SELF));

  /* Test PetscSFReduce with op = MPI_REPLACE, which does x = y */
  CHKERRQ(VecScatterBegin(vscat,y,x,INSERT_VALUES,SCATTER_REVERSE));
  CHKERRQ(VecScatterEnd(vscat,y,x,INSERT_VALUES,SCATTER_REVERSE));
  CHKERRQ(VecView(x,PETSC_VIEWER_STDOUT_WORLD));

  /* Test PetscSFReduce with op = MPI_SUM, which does x += y on x's local part on rank 0*/
  CHKERRQ(VecScatterBegin(vscat,y,x,ADD_VALUES,SCATTER_REVERSE_LOCAL));
  CHKERRQ(VecScatterEnd(vscat,y,x,ADD_VALUES,SCATTER_REVERSE_LOCAL));
  CHKERRQ(VecView(x,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(VecDestroy(&y));
  CHKERRQ(VecScatterDestroy(&vscat));

  /*-------------------------------------*/
  /*       VecScatterCreateToAll         */
  /*-------------------------------------*/
  for (i=low; i<high; i++) CHKERRQ(VecSetValue(x,i,(PetscScalar)i,INSERT_VALUES));
  CHKERRQ(VecAssemblyBegin(x));
  CHKERRQ(VecAssemblyEnd(x));

  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nTesting VecScatterCreateToAll\n"));

  CHKERRQ(VecScatterCreateToAll(x,&vscat,&y));
  CHKERRQ(PetscObjectSetName((PetscObject)y,"y"));

  /* Test PetscSFBcastAndOp with op = MPI_REPLACE, which does y = x on all ranks */
  CHKERRQ(VecScatterBegin(vscat,x,y,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(vscat,x,y,INSERT_VALUES,SCATTER_FORWARD));
  if (rank == 0) CHKERRQ(VecView(y,PETSC_VIEWER_STDOUT_SELF));

  /* Test PetscSFBcastAndOp with op = MPI_SUM, which does y += x */
  CHKERRQ(VecScatterBegin(vscat,x,y,ADD_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(vscat,x,y,ADD_VALUES,SCATTER_FORWARD));
  if (rank == 0) CHKERRQ(VecView(y,PETSC_VIEWER_STDOUT_SELF));

  /* Test PetscSFReduce with op = MPI_REPLACE, which does x = y */
  CHKERRQ(VecScatterBegin(vscat,y,x,INSERT_VALUES,SCATTER_REVERSE));
  CHKERRQ(VecScatterEnd(vscat,y,x,INSERT_VALUES,SCATTER_REVERSE));
  CHKERRQ(VecView(x,PETSC_VIEWER_STDOUT_WORLD));

  /* Test PetscSFReduce with op = MPI_SUM, which does x += size*y */
  CHKERRQ(VecScatterBegin(vscat,y,x,ADD_VALUES,SCATTER_REVERSE));
  CHKERRQ(VecScatterEnd(vscat,y,x,ADD_VALUES,SCATTER_REVERSE));
  CHKERRQ(VecView(x,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&y));
  CHKERRQ(VecScatterDestroy(&vscat));

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   testset:
      # N=10 is divisible by nsize, to trigger Allgather/Gather in SF
      nsize: 2
      # Exact numbers really matter here
      diff_args: -j
      filter: grep -v "type"
      output_file: output/ex8_1.out

      test:
        suffix: 1_standard

      test:
        suffix: 1_cuda
        # sf_backend cuda is not needed if compiling only with cuda
        args: -vec_type cuda -sf_backend cuda
        requires: cuda

      test:
        suffix: 1_hip
        args: -vec_type hip -sf_backend hip
        requires: hip

      test:
        suffix: 1_cuda_aware_mpi
        # sf_backend cuda is not needed if compiling only with cuda
        args: -vec_type cuda -sf_backend cuda
        requires: cuda defined(PETSC_HAVE_MPI_GPU_AWARE)

   testset:
      # N=10 is not divisible by nsize, to trigger Allgatherv/Gatherv in SF
      nsize: 3
      # Exact numbers really matter here
      diff_args: -j
      filter: grep -v "type"
      output_file: output/ex8_2.out

      test:
        suffix: 2_standard

      test:
        suffix: 2_cuda
        # sf_backend cuda is not needed if compiling only with cuda
        args: -vec_type cuda -sf_backend cuda
        requires: cuda

      test:
        suffix: 2_hip
        # sf_backend hip is not needed if compiling only with hip
        args: -vec_type hip -sf_backend hip
        requires: hip

      test:
        suffix: 2_cuda_aware_mpi
        args: -vec_type cuda
        requires: cuda defined(PETSC_HAVE_MPI_GPU_AWARE)

TEST*/

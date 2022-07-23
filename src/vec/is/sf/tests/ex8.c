static char help[]= "Test VecScatterCreateToZero, VecScatterCreateToAll\n\n";

#include <petscvec.h>
int main(int argc,char **argv)
{
  PetscInt           i,N=10,low,high;
  PetscMPIInt        size,rank;
  Vec                x,y;
  VecScatter         vscat;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  PetscCall(VecCreate(PETSC_COMM_WORLD,&x));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecSetSizes(x,PETSC_DECIDE,N));
  PetscCall(VecGetOwnershipRange(x,&low,&high));
  PetscCall(PetscObjectSetName((PetscObject)x,"x"));

  /*-------------------------------------*/
  /*       VecScatterCreateToZero        */
  /*-------------------------------------*/

  /* MPI vec x = [0, 1, 2, .., N-1] */
  for (i=low; i<high; i++) PetscCall(VecSetValue(x,i,(PetscScalar)i,INSERT_VALUES));
  PetscCall(VecAssemblyBegin(x));
  PetscCall(VecAssemblyEnd(x));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\nTesting VecScatterCreateToZero\n"));
  PetscCall(VecScatterCreateToZero(x,&vscat,&y));
  PetscCall(PetscObjectSetName((PetscObject)y,"y"));

  /* Test PetscSFBcastAndOp with op = MPI_REPLACE, which does y = x on rank 0 */
  PetscCall(VecScatterBegin(vscat,x,y,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(vscat,x,y,INSERT_VALUES,SCATTER_FORWARD));
  if (rank == 0) PetscCall(VecView(y,PETSC_VIEWER_STDOUT_SELF));

  /* Test PetscSFBcastAndOp with op = MPI_SUM, which does y += x */
  PetscCall(VecScatterBegin(vscat,x,y,ADD_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(vscat,x,y,ADD_VALUES,SCATTER_FORWARD));
  if (rank == 0) PetscCall(VecView(y,PETSC_VIEWER_STDOUT_SELF));

  /* Test PetscSFReduce with op = MPI_REPLACE, which does x = y */
  PetscCall(VecScatterBegin(vscat,y,x,INSERT_VALUES,SCATTER_REVERSE));
  PetscCall(VecScatterEnd(vscat,y,x,INSERT_VALUES,SCATTER_REVERSE));
  PetscCall(VecView(x,PETSC_VIEWER_STDOUT_WORLD));

  /* Test PetscSFReduce with op = MPI_SUM, which does x += y on x's local part on rank 0*/
  PetscCall(VecScatterBegin(vscat,y,x,ADD_VALUES,SCATTER_REVERSE_LOCAL));
  PetscCall(VecScatterEnd(vscat,y,x,ADD_VALUES,SCATTER_REVERSE_LOCAL));
  PetscCall(VecView(x,PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(VecDestroy(&y));
  PetscCall(VecScatterDestroy(&vscat));

  /*-------------------------------------*/
  /*       VecScatterCreateToAll         */
  /*-------------------------------------*/
  for (i=low; i<high; i++) PetscCall(VecSetValue(x,i,(PetscScalar)i,INSERT_VALUES));
  PetscCall(VecAssemblyBegin(x));
  PetscCall(VecAssemblyEnd(x));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\nTesting VecScatterCreateToAll\n"));

  PetscCall(VecScatterCreateToAll(x,&vscat,&y));
  PetscCall(PetscObjectSetName((PetscObject)y,"y"));

  /* Test PetscSFBcastAndOp with op = MPI_REPLACE, which does y = x on all ranks */
  PetscCall(VecScatterBegin(vscat,x,y,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(vscat,x,y,INSERT_VALUES,SCATTER_FORWARD));
  if (rank == 0) PetscCall(VecView(y,PETSC_VIEWER_STDOUT_SELF));

  /* Test PetscSFBcastAndOp with op = MPI_SUM, which does y += x */
  PetscCall(VecScatterBegin(vscat,x,y,ADD_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(vscat,x,y,ADD_VALUES,SCATTER_FORWARD));
  if (rank == 0) PetscCall(VecView(y,PETSC_VIEWER_STDOUT_SELF));

  /* Test PetscSFReduce with op = MPI_REPLACE, which does x = y */
  PetscCall(VecScatterBegin(vscat,y,x,INSERT_VALUES,SCATTER_REVERSE));
  PetscCall(VecScatterEnd(vscat,y,x,INSERT_VALUES,SCATTER_REVERSE));
  PetscCall(VecView(x,PETSC_VIEWER_STDOUT_WORLD));

  /* Test PetscSFReduce with op = MPI_SUM, which does x += size*y */
  PetscCall(VecScatterBegin(vscat,y,x,ADD_VALUES,SCATTER_REVERSE));
  PetscCall(VecScatterEnd(vscat,y,x,ADD_VALUES,SCATTER_REVERSE));
  PetscCall(VecView(x,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(VecScatterDestroy(&vscat));

  PetscCall(PetscFinalize());
  return 0;
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

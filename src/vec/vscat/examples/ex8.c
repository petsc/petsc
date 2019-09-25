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
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,N);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(x,&low,&high);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)x,"x");CHKERRQ(ierr);

  /*-------------------------------------*/
  /*       VecScatterCreateToZero        */
  /*-------------------------------------*/

  /* MPI vec x = [0, 1, 2, .., N-1] */
  for (i=low; i<high; i++) {ierr = VecSetValue(x,i,(PetscScalar)i,INSERT_VALUES);CHKERRQ(ierr);}
  ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(x);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nTesting VecScatterCreateToZero\n");CHKERRQ(ierr);
  ierr = VecScatterCreateToZero(x,&vscat,&y);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)y,"y");CHKERRQ(ierr);

  /* Test PetscSFBcastAndOp with op = MPI_REPLACE, which does y = x on rank 0 */
  ierr = VecScatterBegin(vscat,x,y,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(vscat,x,y,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  if (!rank) {ierr = VecView(y,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);}

  /* Test PetscSFBcastAndOp with op = MPI_SUM, which does y += x */
  ierr = VecScatterBegin(vscat,x,y,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(vscat,x,y,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  if (!rank) {ierr = VecView(y,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);}

  /* Test PetscSFReduce with op = MPI_REPLACE, which does x = y */
  ierr = VecScatterBegin(vscat,y,x,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(vscat,y,x,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* Test PetscSFReduce with op = MPI_SUM, which does x += y on x's local part on rank 0*/
  ierr = VecScatterBegin(vscat,y,x,ADD_VALUES,SCATTER_REVERSE_LOCAL);CHKERRQ(ierr);
  ierr = VecScatterEnd(vscat,y,x,ADD_VALUES,SCATTER_REVERSE_LOCAL);CHKERRQ(ierr);
  ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&vscat);CHKERRQ(ierr);

  /*-------------------------------------*/
  /*       VecScatterCreateToAll         */
  /*-------------------------------------*/
  for (i=low; i<high; i++) {ierr = VecSetValue(x,i,(PetscScalar)i,INSERT_VALUES);CHKERRQ(ierr);}
  ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(x);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nTesting VecScatterCreateToAll\n");CHKERRQ(ierr);

  ierr = VecScatterCreateToAll(x,&vscat,&y);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)y,"y");CHKERRQ(ierr);

  /* Test PetscSFBcastAndOp with op = MPI_REPLACE, which does y = x on all ranks */
  ierr = VecScatterBegin(vscat,x,y,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(vscat,x,y,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  if (!rank) {ierr = VecView(y,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);}

  /* Test PetscSFBcastAndOp with op = MPI_SUM, which does y += x */
  ierr = VecScatterBegin(vscat,x,y,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(vscat,x,y,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  if (!rank) {ierr = VecView(y,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);}

  /* Test PetscSFReduce with op = MPI_REPLACE, which does x = y */
  ierr = VecScatterBegin(vscat,y,x,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(vscat,y,x,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* Test PetscSFReduce with op = MPI_SUM, which does x += size*y */
  ierr = VecScatterBegin(vscat,y,x,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(vscat,y,x,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&vscat);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      # N=10 is divisible by nsize, to trigger Allgather/Gather in SF
      nsize: 2
      args:
      requires:

   test:
      # N=10 is not divisible by nsize, to trigger Allgatherv/Gatherv in SF
      suffix: 2
      nsize: 3
      args:
      requires:

TEST*/



static char help[] = "Tests the routines VecScatterCreateToAll(), VecScatterCreateToZero()\n\n";

#include <petscvec.h>

int main(int argc,char **argv)
{
  PetscInt       n = 3,i,len,start,end;
  PetscErrorCode ierr;
  PetscMPIInt    size,rank;
  PetscScalar    value,*yy;
  Vec            x,y,z,y_t;
  VecScatter     toall,tozero;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  /* create two vectors */
  CHKERRQ(VecCreateMPI(PETSC_COMM_WORLD,PETSC_DECIDE,size*n,&x));

  /* each processor inserts its values */

  CHKERRQ(VecGetOwnershipRange(x,&start,&end));
  for (i=start; i<end; i++) {
    value = (PetscScalar) i;
    CHKERRQ(VecSetValues(x,1,&i,&value,INSERT_VALUES));
  }
  CHKERRQ(VecAssemblyBegin(x));
  CHKERRQ(VecAssemblyEnd(x));
  CHKERRQ(VecView(x,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(VecScatterCreateToAll(x,&toall,&y));
  CHKERRQ(VecScatterBegin(toall,x,y,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(toall,x,y,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterDestroy(&toall));

  /* Cannot view the above vector with VecView(), so place it in an MPI Vec
     and do a VecView() */
  CHKERRQ(VecGetArray(y,&yy));
  CHKERRQ(VecGetLocalSize(y,&len));
  CHKERRQ(VecCreateMPIWithArray(PETSC_COMM_WORLD,1,len,PETSC_DECIDE,yy,&y_t));
  CHKERRQ(VecView(y_t,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(VecDestroy(&y_t));
  CHKERRQ(VecRestoreArray(y,&yy));

  CHKERRQ(VecScatterCreateToAll(x,&tozero,&z));
  CHKERRQ(VecScatterBegin(tozero,x,z,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(tozero,x,z,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterDestroy(&tozero));
  if (rank == 0) {
    CHKERRQ(VecView(z,PETSC_VIEWER_STDOUT_SELF));
  }
  CHKERRQ(VecDestroy(&z));

  CHKERRQ(VecScatterCreateToZero(x,&tozero,&z));
  CHKERRQ(VecScatterBegin(tozero,x,z,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(tozero,x,z,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterDestroy(&tozero));
  CHKERRQ(VecDestroy(&z));

  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&y));

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      nsize: 4

TEST*/

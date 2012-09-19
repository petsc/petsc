
static char help[] = "Tests the routines VecScatterCreateToAll(), VecScatterCreateToZero()\n\n";

#include <petscvec.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscInt       n = 3,i,len,start,end;
  PetscErrorCode ierr;
  PetscMPIInt    size,rank;
  PetscScalar    value,*yy;
  Vec            x,y,z,y_t;
  VecScatter     toall,tozero;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  /* create two vectors */
  ierr = VecCreateMPI(PETSC_COMM_WORLD,PETSC_DECIDE,size*n,&x);CHKERRQ(ierr);

  /* each processor inserts its values */

  ierr = VecGetOwnershipRange(x,&start,&end);CHKERRQ(ierr);
  for (i=start; i<end; i++) {
    value = (PetscScalar) i;
    ierr = VecSetValues(x,1,&i,&value,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(x);CHKERRQ(ierr);
  ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = VecScatterCreateToAll(x,&toall,&y);CHKERRQ(ierr);
  ierr = VecScatterBegin(toall,x,y,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(toall,x,y,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&toall);CHKERRQ(ierr);

  /* Cannot view the above vector with VecView(), so place it in an MPI Vec
     and do a VecView() */
  ierr = VecGetArray(y,&yy);CHKERRQ(ierr);
  ierr = VecGetLocalSize(y,&len);CHKERRQ(ierr);
  ierr = VecCreateMPIWithArray(PETSC_COMM_WORLD,1,len,PETSC_DECIDE,yy,&y_t);CHKERRQ(ierr);
  ierr = VecView(y_t,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecDestroy(&y_t);
  ierr = VecRestoreArray(y,&yy);CHKERRQ(ierr);

  ierr = VecScatterCreateToAll(x,&tozero,&z);CHKERRQ(ierr);
  ierr = VecScatterBegin(tozero,x,z,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(tozero,x,z,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&tozero);CHKERRQ(ierr);
  if (!rank) {
    ierr = VecView(z,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&z);CHKERRQ(ierr);

  ierr = VecScatterCreateToZero(x,&tozero,&z);CHKERRQ(ierr);
  ierr = VecScatterBegin(tozero,x,z,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(tozero,x,z,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&tozero);CHKERRQ(ierr);
  ierr = VecDestroy(&z);CHKERRQ(ierr);

  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return 0;
}


static const char help[] = "Test VecGetSubVector()\n\n";

#include <petscvec.h>

int main(int argc, char *argv[])
{
  MPI_Comm       comm;
  Vec            X,Y,Z;
  PetscMPIInt    rank,size;
  PetscInt       i,rstart,rend;
  PetscScalar    *x;
  PetscViewer    viewer;
  IS             is0,is1;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,0,help);
  comm = PETSC_COMM_WORLD;
  viewer = PETSC_VIEWER_STDOUT_WORLD;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  ierr = VecCreate(comm,&X);CHKERRQ(ierr);
  ierr = VecSetSizes(X,10,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(X);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(X,&rstart,&rend);CHKERRQ(ierr);

  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  for (i=0; i<rend-rstart; i++) x[i] = rstart+i;
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);

  ierr = ISCreateStride(comm,(rend-rstart)/3+3*(rank>size/2),rstart,1,&is0);CHKERRQ(ierr);
  ierr = ISComplement(is0,rstart,rend,&is1);CHKERRQ(ierr);

  ierr = ISView(is0,viewer);CHKERRQ(ierr);
  ierr = ISView(is1,viewer);CHKERRQ(ierr);

  ierr = VecGetSubVector(X,is0,&Y);CHKERRQ(ierr);
  ierr = VecGetSubVector(X,is1,&Z);CHKERRQ(ierr);
  ierr = VecView(Y,viewer);CHKERRQ(ierr);
  ierr = VecView(Z,viewer);CHKERRQ(ierr);
  ierr = VecRestoreSubVector(X,is0,&Y);CHKERRQ(ierr);
  ierr = VecRestoreSubVector(X,is1,&Z);CHKERRQ(ierr);

  ierr = ISDestroy(&is0);CHKERRQ(ierr);
  ierr = ISDestroy(&is1);CHKERRQ(ierr);
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  PetscFinalize();
  return 0;
}

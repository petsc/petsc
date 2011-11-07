
static char help[] = "Demonstrates scattering with the indices specified by a process that is not sender or receiver.\n\n";

#include <petscvec.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank,size;
  Vec            x,y;
  IS             is1,is2;
  PetscInt       n,N,ix[2],iy[2];
  VecScatter     ctx;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size < 3) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"This example needs at least 3 processes");

  /* create two vectors */
  n = 2;
  N = 2*size;
  ierr = VecCreateMPI(PETSC_COMM_WORLD,n,N,&x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&y);CHKERRQ(ierr);

  /* Specify indices to send from the next process in the ring */
  ix[0] = ((rank+1)*n+0) % N;
  ix[1] = ((rank+1)*n+1) % N;
  /* And put them on the process after that in the ring */
  iy[0] = ((rank+2)*n+0) % N;
  iy[1] = ((rank+2)*n+1) % N;

  /* create two index sets */
  ierr = ISCreateGeneral(PETSC_COMM_WORLD,n,ix,PETSC_USE_POINTER,&is1);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_WORLD,n,iy,PETSC_USE_POINTER,&is2);CHKERRQ(ierr);

  ierr = VecSetValue(x,rank*n,rank*n,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecSetValue(x,rank*n+1,rank*n+1,INSERT_VALUES);CHKERRQ(ierr);

  ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"----\n");CHKERRQ(ierr);

  ierr = VecScatterCreate(x,is1,y,is2,&ctx);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx,x,y,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx,x,y,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&ctx);CHKERRQ(ierr);

  ierr = VecView(y,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = ISDestroy(&is1);CHKERRQ(ierr);
  ierr = ISDestroy(&is2);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}

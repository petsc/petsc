
static char help[] = "Scatters from a parallel vector to a parallel vector.\n\n";

#include "petscvec.h"
#include "petscsys.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  int           n = 5,ierr;
  int           size,rank,N;
  PetscScalar   value,zero = 0.0;
  Vec           x,y;
  IS            is1,is2;
  VecScatter    ctx = 0;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr); 
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  /* create two vectors */
  N = size*n;
  ierr = VecCreate(PETSC_COMM_WORLD,&y);CHKERRQ(ierr);
  ierr = VecSetSizes(y,PETSC_DECIDE,N);CHKERRQ(ierr);
  ierr = VecSetFromOptions(y);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,N);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  /* create two index sets */
  ierr = ISCreateStride(PETSC_COMM_WORLD,n,n*rank,1,&is1);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_WORLD,n,(n*(rank+1))%N,1,&is2);CHKERRQ(ierr);

  value = (PetscScalar)(rank+1); 
  ierr = VecSet(&value,x);CHKERRQ(ierr);
  ierr = VecSet(&zero,y);CHKERRQ(ierr);

  ierr = VecScatterCreate(x,is1,y,is2,&ctx);CHKERRQ(ierr);
  ierr = VecScatterBegin(x,y,ADD_VALUES,SCATTER_FORWARD,ctx);CHKERRQ(ierr);
  ierr = VecScatterEnd(x,y,ADD_VALUES,SCATTER_FORWARD,ctx);CHKERRQ(ierr);
  ierr = VecScatterDestroy(ctx);CHKERRQ(ierr);
  
  ierr = VecView(y,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = VecDestroy(x);CHKERRQ(ierr);
  ierr = VecDestroy(y);CHKERRQ(ierr);
  ierr = ISDestroy(is1);CHKERRQ(ierr);
  ierr = ISDestroy(is2);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
 

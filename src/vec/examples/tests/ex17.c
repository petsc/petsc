/*$Id: ex17.c,v 1.33 2001/01/22 23:03:19 bsmith Exp balay $*/

static char help[] = "Scatters from a parallel vector to a sequential vector.  In\n\
this case each local vector is as long as the entire parallel vector.\n\n";

#include "petscvec.h"
#include "petscsys.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int           n = 5,ierr;
  int           size,rank,N,low,high,iglobal,i;
  Scalar        value,zero = 0.0;
  Vec           x,y;
  IS            is1,is2;
  VecScatter    ctx;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr); 
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  /* create two vectors */
  N = size*n;
  ierr = VecCreateMPI(PETSC_COMM_WORLD,PETSC_DECIDE,N,&y);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,N,&x);CHKERRQ(ierr);

  /* create two index sets */
  ierr = ISCreateStride(PETSC_COMM_SELF,N,0,1,&is1);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_SELF,N,0,1,&is2);CHKERRQ(ierr);

  ierr = VecSet(&zero,x);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(y,&low,&high);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    iglobal = i + low; value = (Scalar) (i + 10*rank);
    ierr = VecSetValues(y,1,&iglobal,&value,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(y);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(y);CHKERRQ(ierr);
  ierr = VecView(y,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = VecScatterCreate(y,is2,x,is1,&ctx);CHKERRQ(ierr);
  ierr = VecScatterBegin(y,x,ADD_VALUES,SCATTER_FORWARD,ctx);CHKERRQ(ierr);
  ierr = VecScatterEnd(y,x,ADD_VALUES,SCATTER_FORWARD,ctx);CHKERRQ(ierr);
  ierr = VecScatterDestroy(ctx);CHKERRQ(ierr);
  
  if (!rank) 
    {printf("----\n"); ierr = VecView(x,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);}

  ierr = VecDestroy(x);CHKERRQ(ierr);
  ierr = VecDestroy(y);CHKERRQ(ierr);
  ierr = ISDestroy(is1);CHKERRQ(ierr);
  ierr = ISDestroy(is2);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
 

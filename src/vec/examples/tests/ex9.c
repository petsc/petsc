/*$Id: ex9.c,v 1.45 2000/05/05 22:15:11 balay Exp bsmith $*/

static char help[]= "Scatters from a parallel vector to a sequential vector.\n\n";

#include "petscvec.h"
#include "petscsys.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int           n = 5,ierr,idx2[3] = {0,2,3},idx1[3] = {0,1,2};
  int           size,rank,i;
  Scalar        mone = -1.0,value;
  Vec           x,y;
  IS            is1,is2;
  VecScatter    ctx = 0;

  PetscInitialize(&argc,&argv,(char*)0,help); 
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRA(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRA(ierr);

  /* create two vectors */
  ierr = VecCreateMPI(PETSC_COMM_WORLD,PETSC_DECIDE,size*n,&x);CHKERRA(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,n,&y);CHKERRA(ierr);

  /* create two index sets */
  ierr = ISCreateGeneral(PETSC_COMM_SELF,3,idx1,&is1);CHKERRA(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,3,idx2,&is2);CHKERRA(ierr);

  /* fill local part of parallel vector */
  for (i=n*rank; i<n*(rank+1); i++) {
    value = (Scalar) i;
    ierr = VecSetValues(x,1,&i,&value,INSERT_VALUES);CHKERRA(ierr);
  }
  ierr = VecAssemblyBegin(x);CHKERRA(ierr);
  ierr = VecAssemblyEnd(x);CHKERRA(ierr);

  ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRA(ierr);

  ierr = VecSet(&mone,y);CHKERRA(ierr);

  ierr = VecScatterCreate(x,is1,y,is2,&ctx);CHKERRA(ierr);
  ierr = VecScatterBegin(x,y,INSERT_VALUES,SCATTER_FORWARD,ctx);CHKERRA(ierr);
  ierr = VecScatterEnd(x,y,INSERT_VALUES,SCATTER_FORWARD,ctx);CHKERRA(ierr);
  ierr = VecScatterDestroy(ctx);CHKERRA(ierr);

  if (!rank) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"scattered vector\n");CHKERRA(ierr);
    ierr = VecView(y,PETSC_VIEWER_STDOUT_SELF);CHKERRA(ierr);
  }
  ierr = ISDestroy(is1);CHKERRA(ierr);
  ierr = ISDestroy(is2);CHKERRA(ierr);
  ierr = VecDestroy(x);CHKERRA(ierr);
  ierr = VecDestroy(y);CHKERRA(ierr);

  PetscFinalize(); 
  return 0;
}
 

/*$Id: ex16.c,v 1.5 1999/05/04 20:30:57 balay Exp bsmith $*/

static char help[] = "Tests VecSetValuesBlocked() on MPI vectors\n\n";

#include "vec.h"
#include "sys.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int          i,n = 8, ierr, size,rank,bs = 2,indices[2];
  Scalar       values[4];
  Vec          x;

  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRA(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRA(ierr);

  if (size != 2) SETERRA(1,0,"Must be run with two processors");

  /* create vector */
  ierr = VecCreateMPI(PETSC_COMM_WORLD,PETSC_DECIDE,n,&x);CHKERRA(ierr);
  ierr = VecSetBlockSize(x,bs);CHKERRA(ierr);

  if (!rank) {
    for ( i=0; i<4; i++ ) values[i] = i+1;
    indices[0] = 0; indices[1] = 2;
    ierr = VecSetValuesBlocked(x,2,indices,values,INSERT_VALUES);CHKERRA(ierr);
  }
  ierr = VecAssemblyBegin(x);CHKERRA(ierr);
  ierr = VecAssemblyEnd(x);CHKERRA(ierr);

  /* 
      Resulting vector should be 1 2 0 0 3 4 0 0
  */
  ierr = VecView(x,VIEWER_STDOUT_WORLD);CHKERRA(ierr);

  ierr = VecDestroy(x);CHKERRA(ierr);

  PetscFinalize();
  return 0;
}
 

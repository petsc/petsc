#ifndef lint
static char vcid[] = "$Id: ex2.c,v 1.1 1996/08/15 19:21:36 bsmith Exp bsmith $";
#endif

static char help[] = "Tests IS stride routines\n\n";

#include "is.h"
#include <math.h>

int main(int argc,char **argv)
{
  int        i, n, ierr,rank,size,*ii,start,stride;
  IS         is;
  PetscTruth flag;

  PetscInitialize(&argc,&argv,(char*)0,help);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&size);

  /*
     Test IS of size 0 
  */
  ierr = ISCreateStride(MPI_COMM_SELF,0,0,2,&is); CHKERRA(ierr);
  ierr = ISGetSize(is,&n); CHKERRA(ierr);
  if (n != 0) SETERRQ(1,0);
  ierr = ISStrideGetInfo(is,&start,&stride); CHKERRA(ierr);
  if (start != 0) SETERRQ(1,0);
  if (stride != 2) SETERRQ(1,0);
  ierr = ISStride(is,&flag); CHKERRA(ierr);
  if (flag != PETSC_TRUE) SETERRQ(1,0);
  ierr = ISGetIndices(is,&ii); CHKERRA(ierr);
  ierr = ISRestoreIndices(is,&ii); CHKERRA(ierr);
  ierr = ISDestroy(is); CHKERRA(ierr);

  /*
     Test ISGetIndices()
  */
  ierr = ISCreateStride(MPI_COMM_SELF,10000,-8,3,&is); CHKERRA(ierr);
  ierr = ISGetSize(is,&n); CHKERRA(ierr);
  ierr = ISGetIndices(is,&ii); CHKERRA(ierr);
  for (i=0; i<10000; i++) {
    if (ii[i] != -8 + 3*i) SETERRQ(1,0);
  }
  ierr = ISRestoreIndices(is,&ii); CHKERRA(ierr);

  PetscFinalize();
  return 0;
}
 







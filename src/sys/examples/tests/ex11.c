/*$Id: ex11.c,v 1.5 1999/10/24 14:01:38 bsmith Exp bsmith $*/

static char help[] = "Tests PetscSynchronizedPrintf() and PetscSynchronizedFPrintf().\n\n";

#include "petsc.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int rank,ierr;

  PetscInitialize(&argc,&argv,(char *)0,help);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"Greetings from %d\n",rank);CHKERRQ(ierr);
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD);CHKERRQ(ierr);

  ierr = PetscSynchronizedFPrintf(PETSC_COMM_WORLD,stderr,"Greetings again from %d\n",rank);CHKERRQ(ierr);
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD);CHKERRQ(ierr);
 
  PetscFinalize();
  return 0;
}
 

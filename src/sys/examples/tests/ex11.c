#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex11.c,v 1.3 1999/03/19 21:17:16 bsmith Exp bsmith $";
#endif

static char help[] = "Tests PetscSynchronizedPrintf() and PetscSynchronizedFPrintf().\n\n";

#include "petsc.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int rank,ierr;

  PetscInitialize(&argc,&argv,(char *)0,help);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRA(ierr);

  ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"Greetings from %d\n",rank);CHKERRA(ierr);
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD);CHKERRA(ierr);

  ierr = PetscSynchronizedFPrintf(PETSC_COMM_WORLD,stderr,"Greetings again from %d\n",rank);CHKERRA(ierr);
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD);CHKERRA(ierr);
 
  PetscFinalize();
  return 0;
}
 

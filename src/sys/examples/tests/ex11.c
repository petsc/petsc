#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex11.c,v 1.2 1999/02/03 04:29:37 bsmith Exp bsmith $";
#endif

static char help[] = "Tests PetscSynchronizedPrintf() and PetscSynchronizedFPrintf().\n\n";

#include "petsc.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int rank;

  PetscInitialize(&argc,&argv,(char *)0,help);
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

  PetscSynchronizedPrintf(PETSC_COMM_WORLD,"Greetings from %d\n",rank);
  PetscSynchronizedFlush(PETSC_COMM_WORLD);

  PetscSynchronizedFPrintf(PETSC_COMM_WORLD,stderr,"Greetings again from %d\n",rank);
  PetscSynchronizedFlush(PETSC_COMM_WORLD);
 
  PetscFinalize();
  return 0;
}
 

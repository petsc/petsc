#ifndef lint
static char vcid[] = "$Id: ex1.c,v 1.10 1996/04/13 15:14:31 curfman Exp $";
#endif
static char help[] = "Demonstrates the usage of PetscInitialize() and \n\
PetscFinalize().\n\n";
 
#include "petsc.h"
int main(int argc,char **argv)
{
  PetscInitialize(&argc,&argv,(char *)0,help);
  PetscSynchronizedPrintf(MPI_COMM_WORLD,"Hello World\n");
  PetscSynchronizedFlush(MPI_COMM_WORLD);
  PetscFinalize();
  return 0;
}

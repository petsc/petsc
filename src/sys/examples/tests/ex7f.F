#ifndef lint
static char vcid[] = "$Id: ex1.c,v 1.10 1996/04/13 15:14:31 curfman Exp $";
#endif

/*
     Formated test for PetscSetCommWorld()
*/

static char help[] = "Tests PetscSetCommWorld()\n\n";

#include "petsc.h"

int main( int argc, char **argv )
{
  int size;

  MPI_Init( &argc, &argv );
  PetscSetCommWorld(MPI_COMM_SELF);
  PetscInitialize(&argc, &argv,PETSC_NULL,help);
   
  MPI_Comm_size(PETSC_COMM_WORLD,&size);
  if (size != 1) SETERRQ(1,"main:Error from PetscSetCommWorld()");

  PetscFinalize();
  MPI_Finalize();
  return 0;
}

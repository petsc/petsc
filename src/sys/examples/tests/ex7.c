#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex7.c,v 1.4 1997/07/09 20:52:08 balay Exp bsmith $";
#endif

/*
     Formatted test for PetscSetCommWorld()
*/

static char help[] = "Tests PetscSetCommWorld()\n\n";

#include "petsc.h"

int main( int argc, char **argv )
{
  int size;

  MPI_Init( &argc, &argv );

  /*
    Create a seperate PETSc universe for each processor
  */
  PetscSetCommWorld(MPI_COMM_SELF);
  PetscInitialize(&argc, &argv,PETSC_NULL,help);
   
  MPI_Comm_size(PETSC_COMM_WORLD,&size);
  if (size != 1) SETERRQ(1,0,"main:Error from PetscSetCommWorld()");

  PetscFinalize();
  MPI_Finalize();
  return 0;
}

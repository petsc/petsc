
/*
     Formatted test for PetscSetCommWorld()
*/

static char help[] = "Tests PetscSetCommWorld().\n\n";

#include "petsc.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  int size,ierr;

  MPI_Init(&argc,&argv);

  /*
    Create a seperate PETSc universe for each processor
  */
  PetscSetCommWorld(MPI_COMM_SELF);
  PetscInitialize(&argc,&argv,PETSC_NULL,help);
   
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size != 1) SETERRQ(1,"main:Error from PetscSetCommWorld()");

  ierr = PetscFinalize();CHKERRQ(ierr);
  MPI_Finalize();
  return 0;
}

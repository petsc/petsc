/*
    Reads in individual PETSc matrix files for each processor and concatinates them
  together into a single file containing the entire matrix
*/
#include "petscmat.h"
#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  int          ierr;

  PetscInitialize(&argc,&argv,(char*) 0,0);
  ierr = MatFileMerge(PETSC_COMM_WORLD,"splitme","together");CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
 

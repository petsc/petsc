
static char help[] = "Tests various ADDA routines.\n\n";

#include "petscda.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  ADDA           adda;
  PetscInt       nodes[4] = {20, 20, 10, 10};
  PetscTruth     periodic[4] = {PETSC_TRUE, PETSC_TRUE, PETSC_FALSE, PETSC_FALSE};

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr); 

  /* Create distributed array and get vectors */
  ierr = ADDACreate(PETSC_COMM_WORLD, 4, nodes, PETSC_NULL, 2, periodic, &adda);CHKERRQ(ierr);

  /* Free memory */
  ierr = ADDADestroy(adda);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}

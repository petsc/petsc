
static char help[] = "Tests error message in DAGetColoring() with periodic boundary conditions. \n\n";


#include "petscda.h"
#include "petscmat.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  Mat            J;                   
  PetscErrorCode ierr; 
  DA             da;
  MatFDColoring  matfdcoloring = 0;
  ISColoring     iscoloring;

  PetscInitialize(&argc,&argv,(char *)0,help);
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DA) to manage parallel grid and vectors
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DACreate2d(PETSC_COMM_WORLD,DA_XPERIODIC,DA_STENCIL_BOX,-5,-5,
                    PETSC_DECIDE,PETSC_DECIDE,1,2,0,0,&da);CHKERRQ(ierr);
  ierr = DAGetMatrix(da,MATAIJ,&J);CHKERRQ(ierr);
  ierr = DAGetColoring(da,IS_COLORING_GHOSTED,MATAIJ,&iscoloring);CHKERRQ(ierr);
  ierr = MatFDColoringCreate(J,iscoloring,&matfdcoloring);CHKERRQ(ierr);
  ierr = ISColoringDestroy(iscoloring);CHKERRQ(ierr);

  /* free spaces */
  ierr = MatDestroy(J);CHKERRQ(ierr);
  ierr = MatFDColoringDestroy(matfdcoloring);CHKERRQ(ierr); 
  ierr = DADestroy(da);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

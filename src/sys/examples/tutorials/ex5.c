
static char help[] = "Demonstrates using the PetscBag Object\n\n";

/*T
   Concepts: bags;
   Processors: n
T*/
#include "petsc.h"
#include "petscbag.h"

/*
   Define a C struct that will contain my program's parameters.
   It MUST begin with the PetscBag struct.
*/
typedef struct {
  PetscBag    bag;
  PetscReal   rho;
  PetscScalar W;
  PetscInt    I;
  PetscTruth  T;
} MyBag;
 

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscBag       *bag;
  MyBag          *mybag;
  PetscViewer    viewer;

  /*
    Every PETSc routine should begin with the PetscInitialize() routine.
    argc, argv - These command line arguments are taken to extract the options
                 supplied to PETSc and options supplied to MPI.
    help       - When PETSc executable is invoked with the option -help, 
                 it prints the various options that can be applied at 
                 runtime.  The user can use the "help" variable place
                 additional help messages in this printout.
  */
  ierr = PetscInitialize(&argc,&argv,(char *)0,help);CHKERRQ(ierr);

  /* Create an empty bag */
  ierr  = PetscBagCreate(PETSC_COMM_WORLD,MyBag,&bag);CHKERRQ(ierr);
  mybag = (MyBag*)bag;

  ierr  = PetscBagSetName(bag,"MyBag","Contains parameters for my physics");CHKERRQ(ierr);
  ierr  = PetscBagRegisterReal(bag,&mybag->rho,3.0,"rho","Some bogus real parameter");CHKERRQ(ierr);
  ierr  = PetscBagRegisterScalar(bag,&mybag->W,5.0,"W","Some other bogus real parameter");CHKERRQ(ierr);
  ierr  = PetscBagRegisterInt(bag,&mybag->I,2,"I","Some other bogus int parameter");CHKERRQ(ierr);
  ierr  = PetscBagRegisterTruth(bag,&mybag->T,2,"T","Some bogus logical parameter");CHKERRQ(ierr);

  mybag->T = PETSC_FALSE;
  ierr = PetscBagView(bag,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscBagView(bag,PETSC_VIEWER_BINARY_WORLD);CHKERRQ(ierr);
  ierr = PetscBagDestroy(bag);CHKERRQ(ierr);

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"binaryoutput",PETSC_FILE_RDONLY,&viewer);CHKERRQ(ierr);
  ierr = PetscBagLoad(viewer,&bag);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
  ierr = PetscBagView(bag,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = PetscBagSetFromOptions(bag);CHKERRQ(ierr);
  ierr = PetscBagView(bag,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = PetscBagDestroy(bag);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}


static char help[] = "Demonstrates using the PetscBag Object\n\n";

/*T
   Concepts: bags;
   Processors: n
T*/
#include "petsc.h"
#include "petscbag.h"

typedef struct {
  PetscReal   x1,x2;
} TwoVec;

/*
   Define a C struct that will contain my program's parameters.
   It MUST begin with the PetscBag struct.
*/

typedef struct {
  PetscBag    bag;
  char        filename[PETSC_MAX_PATH_LEN];
  PetscReal   rho;
  PetscReal   a,b,c,d,e; /* in Parameter struct but not in bag */
  PetscScalar W;
  PetscInt    I;
  PetscTruth  T;
  TwoVec      pos;
} Parameter;
 

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscBag       *bag;
  Parameter      *params;
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
  ierr  = PetscBagCreate(PETSC_COMM_WORLD,Parameter,&bag);CHKERRQ(ierr);
  params = (Parameter*)bag;

  /* register variables, defaults, names, help strings */
  ierr  = PetscBagSetName(bag,"ParameterBag","contains parameters for simulations of top-secret, dangerous physics");CHKERRQ(ierr);
  ierr  = PetscBagRegisterString(bag,&params->filename,PETSC_MAX_PATH_LEN,"myfile","filename","Name of secret file");CHKERRQ(ierr);
  ierr  = PetscBagRegisterReal  (bag,&params->rho,3.0,"rho","Density, kg/m^3");CHKERRQ(ierr);
  ierr  = PetscBagRegisterScalar(bag,&params->W,  5.0,"W","Vertical velocity, m/sec");CHKERRQ(ierr);
  ierr  = PetscBagRegisterInt   (bag,&params->I,  2,"modes_x","Number of modes in x-direction");CHKERRQ(ierr);
  ierr  = PetscBagRegisterTruth (bag,&params->T,  PETSC_FALSE,"do_output","Write output file (yes/no)");CHKERRQ(ierr);
  ierr  = PetscBagRegisterReal  (bag,&params->pos.x1,1.0,"x1","x position");CHKERRQ(ierr);
  ierr  = PetscBagRegisterReal  (bag,&params->pos.x2,1.9,"x2","y position");CHKERRQ(ierr);

  /* get options from command line THIS IS NO LONGER NECESSARY */
  /* ierr = PetscBagSetFromOptions(bag);CHKERRQ(ierr); */

  /* write bag to stdio & file */
  ierr = PetscBagView(bag,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscBagView(bag,PETSC_VIEWER_BINARY_WORLD);CHKERRQ(ierr);
  ierr = PetscBagDestroy(bag);CHKERRQ(ierr);

  /* load bag from file & write to stdio */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"binaryoutput",PETSC_FILE_RDONLY,&viewer);CHKERRQ(ierr);
  ierr = PetscBagLoad(viewer,&bag);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
  ierr = PetscBagView(bag,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* reuse the parameter struct */
  params = (Parameter*)bag;
  PetscPrintf(PETSC_COMM_WORLD,"The value of rho after loading is: %f\n",params->rho);

  /* clean up and exit */
  ierr = PetscBagDestroy(bag);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}

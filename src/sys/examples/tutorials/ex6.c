static char help[] = "Demonstrates using the PetscViewerBinaryMatlab viewer\n\n";

/*T
   Concepts: viewers; bags;
   Processors: n
T*/
#include "petsc.h"
#include "petscda.h"
#include "petscbag.h"

typedef struct {
  char          filename[PETSC_MAX_PATH_LEN];
  PetscReal     ra;
  PetscInt      ia;
  PetscTruth    ta;
} Parameter;

typedef struct {
  PetscScalar u,v,w,p,T;
} Field;

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscBag       bag;
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
  ierr = PetscBagCreate(PETSC_COMM_WORLD,sizeof(Parameter),&bag);CHKERRQ(ierr);
  ierr = PetscBagGetData(bag,(void **)&params);CHKERRQ(ierr);

  /* fill bag: register variables, defaults, names, help strings */
  ierr = PetscBagSetName(bag,"ParameterBag","contains parameters for ex6");CHKERRQ(ierr);
  ierr = PetscBagRegisterString(bag,&params->filename,PETSC_MAX_PATH_LEN,"ex6_output","filename","Name of secret file");CHKERRQ(ierr);
  ierr = PetscBagRegisterReal  (bag,&params->ra,1.0,"param_1","The first parameter");CHKERRQ(ierr);
  ierr = PetscBagRegisterInt   (bag,&params->ia,5  ,"param_2","The second parameter");CHKERRQ(ierr);
  ierr = PetscBagRegisterTruth (bag,&params->ta,PETSC_TRUE,"do_output","Write output file (true/false)");CHKERRQ(ierr);

  /* 
     Write output file with PetscViewerBinaryMatlab viewer. 
     NOTE: the output generated with this viewer can be loaded into
     matlab using bin/matlab/PetscBinaryReadMatlab.m 
  */
  ierr = PetscViewerBinaryMatlabOpen(PETSC_COMM_WORLD,params->filename,&viewer); CHKERRQ(ierr);
  ierr = PetscViewerBinaryMatlabOutputBag(viewer,"params",bag); CHKERRQ(ierr);
  ierr = PetscViewerBinaryMatlabDestroy(viewer); CHKERRQ(ierr);
  
  /* clean up and exit */
  ierr = PetscBagDestroy(bag);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;  
}

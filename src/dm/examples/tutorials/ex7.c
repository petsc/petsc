static char help[] = "Demonstrates using the PetscViewerBinaryMatlab viewer\n\n";

/*T
   Concepts: viewers
   Concepts: bags
   Processors: n
T*/
#include <petscsys.h>
#include <petscdmda.h>
#include <petscbag.h>

typedef struct {
  char          filename[PETSC_MAX_PATH_LEN];
  PetscReal     ra;
  PetscInt      ia;
  PetscBool     ta;
} Parameter;

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscBag       bag;
  Parameter      *params;
  PetscViewer    viewer;
  DM             da;
  Vec            global,local;
  PetscMPIInt    rank;

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

  /* Create a DMDA and an associated vector */
  ierr = DMDACreate2d(PETSC_COMM_WORLD, DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_NONE,DMDA_STENCIL_BOX,10,10,
		    PETSC_DECIDE,PETSC_DECIDE,2,1,PETSC_NULL,PETSC_NULL,&da);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&global);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(da,&local);CHKERRQ(ierr);
  ierr = VecSet(global,-1.0);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da,global,INSERT_VALUES,local);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,global,INSERT_VALUES,local);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = VecScale(local,rank+1);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(da,local,ADD_VALUES,global);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(da,local,ADD_VALUES,global);CHKERRQ(ierr);

  /* Create an empty bag */
  ierr = PetscBagCreate(PETSC_COMM_WORLD,sizeof(Parameter),&bag);CHKERRQ(ierr);
  ierr = PetscBagGetData(bag,(void **)&params);CHKERRQ(ierr);

  /* fill bag: register variables, defaults, names, help strings */
  ierr = PetscBagSetName(bag,"ParameterBag","contains problem parameters");CHKERRQ(ierr);
  ierr = PetscBagRegisterString(bag,&params->filename,PETSC_MAX_PATH_LEN,"output_file","filename","Name of secret file");CHKERRQ(ierr);
  ierr = PetscBagRegisterReal  (bag,&params->ra,1.0,"param_1","The first parameter");CHKERRQ(ierr);
  ierr = PetscBagRegisterInt   (bag,&params->ia,5  ,"param_2","The second parameter");CHKERRQ(ierr);
  ierr = PetscBagRegisterBool (bag,&params->ta,PETSC_TRUE,"do_output","Write output file (true/false)");CHKERRQ(ierr);

  /* 
     Write output file with PetscViewerBinaryMatlab viewer. 
     NOTE: the output generated with this viewer can be loaded into
     matlab using bin/matlab/PetscBinaryReadMatlab.m 
  */
  ierr = PetscViewerBinaryMatlabOpen(PETSC_COMM_WORLD,params->filename,&viewer);CHKERRQ(ierr);
  ierr = PetscViewerBinaryMatlabOutputBag(viewer,"params",bag);CHKERRQ(ierr);
  ierr = DMDASetFieldName(da,0,"field1");CHKERRQ(ierr);
  ierr = DMDASetFieldName(da,1,"field2");CHKERRQ(ierr);
  ierr = PetscViewerBinaryMatlabOutputVecDA(viewer,"da1",global,da);CHKERRQ(ierr);
  ierr = PetscViewerBinaryMatlabDestroy(&viewer);CHKERRQ(ierr);
  
  /* clean up and exit */
  ierr = PetscBagDestroy(&bag);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = VecDestroy(&local);CHKERRQ(ierr);
  ierr = VecDestroy(&global);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;  
}

static char help[] = "Demonstrates using PetscViewerPushFormat(viewer,PETSC_VIEWER_BINARY_MATLAB)\n\n";

/*T
   Concepts: viewers
   Concepts: bags
   Processors: n
T*/
#include <petscsys.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscbag.h>

typedef struct {
  char      filename[PETSC_MAX_PATH_LEN];
  PetscReal ra;
  PetscInt  ia;
  PetscBool ta;
} Parameter;

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
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  /* Create a DMDA and an associated vector */
  ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,10,10,PETSC_DECIDE,PETSC_DECIDE,2,1,NULL,NULL,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
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
  ierr = PetscBagGetData(bag,(void**)&params);CHKERRQ(ierr);

  /* fill bag: register variables, defaults, names, help strings */
  ierr = PetscBagSetName(bag,"ParameterBag","contains problem parameters");CHKERRQ(ierr);
  ierr = PetscBagRegisterString(bag,&params->filename,PETSC_MAX_PATH_LEN,"output_file","filename","Name of secret file");CHKERRQ(ierr);
  ierr = PetscBagRegisterReal  (bag,&params->ra,1.0,"param_1","The first parameter");CHKERRQ(ierr);
  ierr = PetscBagRegisterInt   (bag,&params->ia,5,"param_2","The second parameter");CHKERRQ(ierr);
  ierr = PetscBagRegisterBool (bag,&params->ta,PETSC_TRUE,"do_output","Write output file (true/false)");CHKERRQ(ierr);

  /*
     Write output file with PETSC_VIEWER_BINARY_MATLAB format
     NOTE: the output generated with this viewer can be loaded into
     MATLAB using $PETSC_DIR/share/petsc/matlab/PetscReadBinaryMatlab.m
  */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,params->filename,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_BINARY_MATLAB);CHKERRQ(ierr);
  ierr = PetscBagView(bag,viewer);CHKERRQ(ierr);
  ierr = DMDASetFieldName(da,0,"field1");CHKERRQ(ierr);
  ierr = DMDASetFieldName(da,1,"field2");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)global,"da1");CHKERRQ(ierr);
  ierr = VecView(global,viewer);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  /* clean up and exit */
  ierr = PetscBagDestroy(&bag);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = VecDestroy(&local);CHKERRQ(ierr);
  ierr = VecDestroy(&global);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}


/*TEST

   test:

TEST*/

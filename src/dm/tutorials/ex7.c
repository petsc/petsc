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
  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
  /* Create a DMDA and an associated vector */
  CHKERRQ(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,10,10,PETSC_DECIDE,PETSC_DECIDE,2,1,NULL,NULL,&da));
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetUp(da));
  CHKERRQ(DMCreateGlobalVector(da,&global));
  CHKERRQ(DMCreateLocalVector(da,&local));
  CHKERRQ(VecSet(global,-1.0));
  CHKERRQ(DMGlobalToLocalBegin(da,global,INSERT_VALUES,local));
  CHKERRQ(DMGlobalToLocalEnd(da,global,INSERT_VALUES,local));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRQ(VecScale(local,rank+1));
  CHKERRQ(DMLocalToGlobalBegin(da,local,ADD_VALUES,global));
  CHKERRQ(DMLocalToGlobalEnd(da,local,ADD_VALUES,global));

  /* Create an empty bag */
  CHKERRQ(PetscBagCreate(PETSC_COMM_WORLD,sizeof(Parameter),&bag));
  CHKERRQ(PetscBagGetData(bag,(void**)&params));

  /* fill bag: register variables, defaults, names, help strings */
  CHKERRQ(PetscBagSetName(bag,"ParameterBag","contains problem parameters"));
  CHKERRQ(PetscBagRegisterString(bag,&params->filename,PETSC_MAX_PATH_LEN,"output_file","filename","Name of secret file"));
  CHKERRQ(PetscBagRegisterReal  (bag,&params->ra,1.0,"param_1","The first parameter"));
  CHKERRQ(PetscBagRegisterInt   (bag,&params->ia,5,"param_2","The second parameter"));
  CHKERRQ(PetscBagRegisterBool (bag,&params->ta,PETSC_TRUE,"do_output","Write output file (true/false)"));

  /*
     Write output file with PETSC_VIEWER_BINARY_MATLAB format
     NOTE: the output generated with this viewer can be loaded into
     MATLAB using $PETSC_DIR/share/petsc/matlab/PetscReadBinaryMatlab.m
  */
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,params->filename,FILE_MODE_WRITE,&viewer));
  CHKERRQ(PetscViewerPushFormat(viewer,PETSC_VIEWER_BINARY_MATLAB));
  CHKERRQ(PetscBagView(bag,viewer));
  CHKERRQ(DMDASetFieldName(da,0,"field1"));
  CHKERRQ(DMDASetFieldName(da,1,"field2"));
  CHKERRQ(PetscObjectSetName((PetscObject)global,"da1"));
  CHKERRQ(VecView(global,viewer));
  CHKERRQ(PetscViewerPopFormat(viewer));
  CHKERRQ(PetscViewerDestroy(&viewer));

  /* clean up and exit */
  CHKERRQ(PetscBagDestroy(&bag));
  CHKERRQ(DMDestroy(&da));
  CHKERRQ(VecDestroy(&local));
  CHKERRQ(VecDestroy(&global));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/

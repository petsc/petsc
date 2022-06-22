static char help[] = "Demonstrates using PetscViewerPushFormat(viewer,PETSC_VIEWER_BINARY_MATLAB)\n\n";

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
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  /* Create a DMDA and an associated vector */
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,10,10,PETSC_DECIDE,PETSC_DECIDE,2,1,NULL,NULL,&da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMCreateGlobalVector(da,&global));
  PetscCall(DMCreateLocalVector(da,&local));
  PetscCall(VecSet(global,-1.0));
  PetscCall(DMGlobalToLocalBegin(da,global,INSERT_VALUES,local));
  PetscCall(DMGlobalToLocalEnd(da,global,INSERT_VALUES,local));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCall(VecScale(local,rank+1));
  PetscCall(DMLocalToGlobalBegin(da,local,ADD_VALUES,global));
  PetscCall(DMLocalToGlobalEnd(da,local,ADD_VALUES,global));

  /* Create an empty bag */
  PetscCall(PetscBagCreate(PETSC_COMM_WORLD,sizeof(Parameter),&bag));
  PetscCall(PetscBagGetData(bag,(void**)&params));

  /* fill bag: register variables, defaults, names, help strings */
  PetscCall(PetscBagSetName(bag,"ParameterBag","contains problem parameters"));
  PetscCall(PetscBagRegisterString(bag,&params->filename,PETSC_MAX_PATH_LEN,"output_file","filename","Name of secret file"));
  PetscCall(PetscBagRegisterReal  (bag,&params->ra,1.0,"param_1","The first parameter"));
  PetscCall(PetscBagRegisterInt   (bag,&params->ia,5,"param_2","The second parameter"));
  PetscCall(PetscBagRegisterBool (bag,&params->ta,PETSC_TRUE,"do_output","Write output file (true/false)"));

  /*
     Write output file with PETSC_VIEWER_BINARY_MATLAB format
     NOTE: the output generated with this viewer can be loaded into
     MATLAB using $PETSC_DIR/share/petsc/matlab/PetscReadBinaryMatlab.m
  */
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,params->filename,FILE_MODE_WRITE,&viewer));
  PetscCall(PetscViewerPushFormat(viewer,PETSC_VIEWER_BINARY_MATLAB));
  PetscCall(PetscBagView(bag,viewer));
  PetscCall(DMDASetFieldName(da,0,"field1"));
  PetscCall(DMDASetFieldName(da,1,"field2"));
  PetscCall(PetscObjectSetName((PetscObject)global,"da1"));
  PetscCall(VecView(global,viewer));
  PetscCall(PetscViewerPopFormat(viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  /* clean up and exit */
  PetscCall(PetscBagDestroy(&bag));
  PetscCall(DMDestroy(&da));
  PetscCall(VecDestroy(&local));
  PetscCall(VecDestroy(&global));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/

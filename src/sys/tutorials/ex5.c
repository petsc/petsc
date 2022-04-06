
static char help[] = "Demonstrates using the PetscBag Object\n\n";

#include <petscsys.h>
#include <petscbag.h>
#include <petscviewer.h>

/*
  Enum variables can be stored in a bag but require a string array
  to name their fields.  The fourth entry in this example is the name
  of the enum, the fifth is the prefix (none in this case), and the last
  entry is the null string.
*/
typedef enum {
  THIS = 0, THAT = 1, THE_OTHER = 2
} YourChoice;
const char *EnumeratedChoices[] = {"THIS","THAT","THE_OTHER","EnumeratedChoices","",0};

/*
  Data structures can be used in a bag as long as they
  are declared in the bag with a variable, not with a pointer.
*/
typedef struct {
  PetscReal x1,x2;
} TwoVec;

/*
  Define a C struct that will contain my program's parameters.

  A PETSc bag is merely a representation of a C struct that can be printed, saved to a file and loaded from a file.
*/
typedef struct {
  PetscScalar   W;
  PetscReal     rho;
  TwoVec        pos;
  PetscInt      Ii;
  PetscInt      iarray[3];
  PetscReal     rarray[2];
  PetscBool     T;
  PetscBool     Tarray[3];
  PetscDataType dt;
  char          filename[PETSC_MAX_PATH_LEN];
  YourChoice    which;
} Parameter;

int main(int argc,char **argv)
{
  PetscBag       bag;
  Parameter      *params;
  PetscViewer    viewer;
  PetscBool      flg;
  char           filename[PETSC_MAX_PATH_LEN] = "binaryoutput";

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

  /* Create an empty bag */
  PetscCall(PetscBagCreate(PETSC_COMM_WORLD,sizeof(Parameter),&bag));
  PetscCall(PetscBagGetData(bag,(void**)&params));

  /* register variables, defaults, names, help strings */
  PetscCall(PetscBagSetName(bag,"ParameterBag","contains parameters for simulations of top-secret, dangerous physics"));
  PetscCall(PetscBagSetOptionsPrefix(bag, "pbag_"));
  PetscCall(PetscBagRegisterString(bag,&params->filename,PETSC_MAX_PATH_LEN,"myfile","filename","Name of secret file"));
  PetscCall(PetscBagRegisterReal  (bag,&params->rho,3.0,"rho","Density, kg/m^3"));
  PetscCall(PetscBagRegisterScalar(bag,&params->W,  5.0,"W","Vertical velocity, m/sec"));
  PetscCall(PetscBagRegisterInt   (bag,&params->Ii, 2,"modes_x","Number of modes in x-direction"));

  params->iarray[0] = 1;
  params->iarray[1] = 2;
  params->iarray[2] = 3;

  PetscCall(PetscBagRegisterIntArray(bag,&params->iarray, 3,"int_array","Int array with 3 locations"));

  params->rarray[0] = -1.0;
  params->rarray[1] = -2.0;

  PetscCall(PetscBagRegisterRealArray(bag,&params->rarray, 2,"real_array","Real array with 2 locations"));
  PetscCall(PetscBagRegisterBool (bag,&params->T,  PETSC_FALSE,"do_output","Write output file (yes/no)"));
  PetscCall(PetscBagRegisterBoolArray(bag,&params->Tarray, 3,"bool_array","Bool array with 3 locations"));
  PetscCall(PetscBagRegisterEnum  (bag,&params->dt, PetscDataTypes,(PetscEnum)PETSC_INT,"dt","meaningless datatype"));
  PetscCall(PetscBagRegisterReal  (bag,&params->pos.x1,1.0,"x1","x position"));
  PetscCall(PetscBagRegisterReal  (bag,&params->pos.x2,1.9,"x2","y position"));
  PetscCall(PetscBagRegisterEnum  (bag,&params->which, EnumeratedChoices, (PetscEnum)THAT, "choose","Express yourself by choosing among enumerated things"));

  /* This option allows loading user-provided PetscBag */
  PetscCall(PetscOptionsGetString(NULL,NULL,"-f",filename,sizeof(filename),&flg));
  if (!flg) {

    /* write bag to stdio & binary file */
    PetscCall(PetscBagView(bag,PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_WRITE,&viewer));
    PetscCall(PetscBagView(bag,viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }

  PetscCall(PetscMemzero(params,sizeof(Parameter)));

  /* load bag from file & write to stdio */
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer));
  PetscCall(PetscBagLoad(viewer,bag));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(PetscBagSetFromOptions(bag));
  PetscCall(PetscBagView(bag,PETSC_VIEWER_STDOUT_WORLD));

  /* reuse the parameter struct */
  PetscCall(PetscBagGetData(bag,(void**)&params));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"The value of rho after loading is: %f\n",(double)params->rho));

  /* clean up and exit */
  PetscCall(PetscBagDestroy(&bag));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      args: -pbag_rho 44 -pbag_do_output true
      requires: !complex

   test:
      suffix: yaml
      requires: !complex
      args: -options_file bag.yml -options_view
      filter: egrep -v "(options_left|options_view|malloc_dump|malloc_test|saws_port_auto_select|display|check_pointer_intensity|error_output_stdout|nox|vecscatter_mpi1|use_gpu_aware_mpi|checkstack)"
      localrunfiles: bag.yml

TEST*/

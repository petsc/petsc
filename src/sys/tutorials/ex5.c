
static char help[] = "Demonstrates using the PetscBag Object\n\n";

/*T
   Concepts: bags;
   Processors: n
T*/

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
  PetscErrorCode ierr;
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
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  /* Create an empty bag */
  CHKERRQ(PetscBagCreate(PETSC_COMM_WORLD,sizeof(Parameter),&bag));
  CHKERRQ(PetscBagGetData(bag,(void**)&params));

  /* register variables, defaults, names, help strings */
  CHKERRQ(PetscBagSetName(bag,"ParameterBag","contains parameters for simulations of top-secret, dangerous physics"));
  CHKERRQ(PetscBagSetOptionsPrefix(bag, "pbag_"));
  CHKERRQ(PetscBagRegisterString(bag,&params->filename,PETSC_MAX_PATH_LEN,"myfile","filename","Name of secret file"));
  CHKERRQ(PetscBagRegisterReal  (bag,&params->rho,3.0,"rho","Density, kg/m^3"));
  CHKERRQ(PetscBagRegisterScalar(bag,&params->W,  5.0,"W","Vertical velocity, m/sec"));
  CHKERRQ(PetscBagRegisterInt   (bag,&params->Ii, 2,"modes_x","Number of modes in x-direction"));

  params->iarray[0] = 1;
  params->iarray[1] = 2;
  params->iarray[2] = 3;

  CHKERRQ(PetscBagRegisterIntArray(bag,&params->iarray, 3,"int_array","Int array with 3 locations"));

  params->rarray[0] = -1.0;
  params->rarray[1] = -2.0;

  CHKERRQ(PetscBagRegisterRealArray(bag,&params->rarray, 2,"real_array","Real array with 2 locations"));
  CHKERRQ(PetscBagRegisterBool (bag,&params->T,  PETSC_FALSE,"do_output","Write output file (yes/no)"));
  CHKERRQ(PetscBagRegisterBoolArray(bag,&params->Tarray, 3,"bool_array","Bool array with 3 locations"));
  CHKERRQ(PetscBagRegisterEnum  (bag,&params->dt, PetscDataTypes,(PetscEnum)PETSC_INT,"dt","meaningless datatype"));
  CHKERRQ(PetscBagRegisterReal  (bag,&params->pos.x1,1.0,"x1","x position"));
  CHKERRQ(PetscBagRegisterReal  (bag,&params->pos.x2,1.9,"x2","y position"));
  CHKERRQ(PetscBagRegisterEnum  (bag,&params->which, EnumeratedChoices, (PetscEnum)THAT, "choose","Express yourself by choosing among enumerated things"));

  /* This option allows loading user-provided PetscBag */
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-f",filename,sizeof(filename),&flg));
  if (!flg) {

    /* write bag to stdio & binary file */
    CHKERRQ(PetscBagView(bag,PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_WRITE,&viewer));
    CHKERRQ(PetscBagView(bag,viewer));
    CHKERRQ(PetscViewerDestroy(&viewer));
  }

  CHKERRQ(PetscMemzero(params,sizeof(Parameter)));

  /* load bag from file & write to stdio */
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer));
  CHKERRQ(PetscBagLoad(viewer,bag));
  CHKERRQ(PetscViewerDestroy(&viewer));
  CHKERRQ(PetscBagSetFromOptions(bag));
  CHKERRQ(PetscBagView(bag,PETSC_VIEWER_STDOUT_WORLD));

  /* reuse the parameter struct */
  CHKERRQ(PetscBagGetData(bag,(void**)&params));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"The value of rho after loading is: %f\n",(double)params->rho));

  /* clean up and exit */
  CHKERRQ(PetscBagDestroy(&bag));
  ierr = PetscFinalize();
  return ierr;
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

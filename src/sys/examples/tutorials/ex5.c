
static char help[] = "Demonstrates using the PetscBag Object\n\n";

/*T
   Concepts: bags;
   Processors: n
T*/
#include <petscsys.h>
#include <petscbag.h>

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
  PetscReal   x1,x2;
} TwoVec;

/* 
  Define a C struct that will contain my program's parameters.
*/
typedef struct {
  PetscScalar   W;
  PetscReal     rho;
  TwoVec        pos; 
  PetscInt      Ii;
  PetscInt      iarray[3];
  PetscReal     rarray[2];
  PetscBool     T;
  PetscDataType dt;
  char          filename[PETSC_MAX_PATH_LEN];
  YourChoice    which;
} Parameter;
 

#undef __FUNCT__
#define __FUNCT__ "main"
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
  ierr = PetscInitialize(&argc,&argv,(char *)0,help);CHKERRQ(ierr);

  /* Create an empty bag */
  ierr   = PetscBagCreate(PETSC_COMM_WORLD,sizeof(Parameter),&bag);CHKERRQ(ierr);
  ierr   = PetscBagGetData(bag,(void **)&params);CHKERRQ(ierr);
  
  /* register variables, defaults, names, help strings */
  ierr = PetscBagSetName(bag,"ParameterBag","contains parameters for simulations of top-secret, dangerous physics");CHKERRQ(ierr);
  ierr = PetscBagSetOptionsPrefix(bag, "pbag_");CHKERRQ(ierr);
  ierr = PetscBagRegisterString(bag,&params->filename,PETSC_MAX_PATH_LEN,"myfile","filename","Name of secret file");CHKERRQ(ierr);
  ierr = PetscBagRegisterReal  (bag,&params->rho,3.0,"rho","Density, kg/m^3");CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag,&params->W,  5.0,"W","Vertical velocity, m/sec");CHKERRQ(ierr);
  ierr = PetscBagRegisterInt   (bag,&params->Ii, 2,"modes_x","Number of modes in x-direction");CHKERRQ(ierr);
  params->iarray[0] = 1;
  params->iarray[1] = 2;
  params->iarray[2] = 3;
  ierr = PetscBagRegisterIntArray(bag,&params->iarray, 3,"int_array","Int array with 3 locations");CHKERRQ(ierr);
  params->rarray[0] = -1.0;
  params->rarray[1] = -2.0;
  ierr = PetscBagRegisterRealArray(bag,&params->rarray, 2,"real_array","Real array with 2 locations");CHKERRQ(ierr);
  ierr = PetscBagRegisterBool (bag,&params->T,  PETSC_FALSE,"do_output","Write output file (yes/no)");CHKERRQ(ierr);
  ierr = PetscBagRegisterEnum  (bag,&params->dt, PetscDataTypes,(PetscEnum)PETSC_INT,"dt","meaningless datatype");CHKERRQ(ierr);
  ierr = PetscBagRegisterReal  (bag,&params->pos.x1,1.0,"x1","x position");CHKERRQ(ierr);
  ierr = PetscBagRegisterReal  (bag,&params->pos.x2,1.9,"x2","y position");CHKERRQ(ierr);
  ierr = PetscBagRegisterEnum  (bag,&params->which, EnumeratedChoices, (PetscEnum)THAT, "choose","Express yourself by choosing among enumerated things");CHKERRQ(ierr);

  
  /* This option allows loading user-provided PetscBag */
  ierr = PetscOptionsGetString(PETSC_NULL,"-f",filename,sizeof filename,&flg);CHKERRQ(ierr);
  if (!flg) {

    /* write bag to stdio & binary file */
    ierr = PetscBagView(bag,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
    ierr = PetscBagView(bag,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
  
  ierr = PetscMemzero(params,sizeof(Parameter));CHKERRQ(ierr);

  /* load bag from file & write to stdio */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = PetscBagLoad(viewer,bag);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = PetscBagSetFromOptions(bag);CHKERRQ(ierr);
  ierr = PetscBagView(bag,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* reuse the parameter struct */
  ierr = PetscBagGetData(bag,(void**)&params);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"The value of rho after loading is: %f\n",(double)params->rho);CHKERRQ(ierr);

#if defined(PETSC_USE_SOCKET_VIEWER)
  {
    PetscBool  flg;
    ierr = PetscOptionsName("-bag_view_socket","Sends bag to socket (can be read from matlab)","PetscBagView",&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PetscBagView(bag,PETSC_VIEWER_SOCKET_(PETSC_COMM_WORLD));CHKERRQ(ierr);
      ierr = PetscViewerFlush(PETSC_VIEWER_SOCKET_(PETSC_COMM_WORLD));CHKERRQ(ierr);
    }
  }
#endif

  /* clean up and exit */
  ierr = PetscBagDestroy(&bag);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}

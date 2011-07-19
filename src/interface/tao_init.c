#include "taosolver.h"   /*I "taosolver.h" I*/

PetscBool TaoInitializeCalled = PETSC_FALSE;
PetscBool TaoBeganPetsc = PETSC_FALSE;

static int tao_one=1;
static char* tao_executable = (char*)"tao";
static char** tao_executablePtr = &tao_executable;
#undef __FUNCT__
#define __FUNCT__ "TaoInitialize"
/*@C 
  TaoInitialize - Initializes the TAO component and many of the packages associated with it.

   Collective on MPI_COMM_WORLD

   Input Parameters:
+  argc - [optional] count of number of command line arguments
.  args - [optional] the command line arguments
.  file - [optional] PETSc database file, defaults to ~username/.petscrc
          (use TAO_NULL for default)
-  help - [optional] Help message to print, use TAO_NULL for no message

   Note:
   TaoInitialize() should always be called near the beginning of your 
   program.  However, this command should come after PetscInitialize()

   Level: beginner

.keywords: TAO_SOLVER, initialize

.seealso: TaoInitializeFortran(), TaoFinalize(), PetscInitialize()
@*/
PetscErrorCode TaoInitialize(int *argc, char ***args, const char file[], 
			     const char help[])
{
  PetscErrorCode ierr;

  if (TaoInitializeCalled) {
    return(0);
  }
  if (PetscInitializeCalled) {
    PetscInfo(0,"TAO successfully initialized.\n"); CHKERRQ(ierr);
  } else {
    if (argc&&args) {
      PetscInitialize(argc,args,file,help); 
    } else {
      PetscInitialize(&tao_one,&tao_executablePtr,0,0); 
    }
    TaoBeganPetsc=PETSC_TRUE;
  }
  if (!PetscInitializeCalled) {
    printf("Error initializing PETSc -- aborting.\n");
    exit(1);
  }
  ierr = TaoInitialize_DynamicLibraries();  CHKERRQ(ierr);
  TaoInitializeCalled = PETSC_TRUE;
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "TaoFinalize"
/*@
   TaoFinalize - Checks for options at the end of the TAO program
   and finalizes interfaces with other packages.

   Collective on MPI_COMM_WORLD

   Level: beginner

.keywords: finalize, exit, end

.seealso: TaoInitialize(), PetscFinalize()
@*/
PetscErrorCode TaoFinalize()
{
  PetscErrorCode ierr;
  TaoFinalize_DynamicLibraries();
  if (TaoBeganPetsc) {
    PetscFinalize();
  } 
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoInitialize_DynamicLibraries"
PetscErrorCode TaoInitialize_DynamicLibraries(void)
{
  PetscErrorCode ierr;
  char path[PETSC_MAX_PATH_LEN];
  PetscFunctionBegin;


  ierr = PetscStrcpy(path,TAO_LIB_DIR); CHKERRQ(ierr);
  ierr = PetscStrcat(path,"/libtaosolver"); CHKERRQ(ierr);
#ifdef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = PetscDLLibraryAppend(PETSC_COMM_WORLD,&DLLibrariesLoaded,path); CHKERRQ(ierr);
#endif

  ierr = PetscStrcpy(path,TAO_LIB_DIR); CHKERRQ(ierr);
  ierr = PetscStrcat(path,"/libtaolinesearch"); CHKERRQ(ierr);
#ifdef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = PetscDLLibraryAppend(PETSC_COMM_WORLD,&DLLibrariesLoaded,path); CHKERRQ(ierr);
#endif


  PetscFunctionReturn(0);
  
}    

#undef __FUNCT__
#define __FUNCT__ "TaoFinalize_DynamicLibraries"
PetscErrorCode TaoFinalize_DynamicLibraries(void)
{
    PetscFunctionBegin;
    PetscFunctionReturn(0);
    
}

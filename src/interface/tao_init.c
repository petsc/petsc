#include "taosolver.h"   /*I "taosolver.h" I*/

PetscBool TaoInitializeCalled = PETSC_FALSE;


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
  PetscFunctionBegin;
  if (TaoInitializeCalled) {
    PetscFunctionReturn(0);
  }
  TaoInitializeCalled = PETSC_TRUE;
  ierr = TaoInitialize_DynamicLibraries(); CHKERRQ(ierr);
  ierr = PetscInfo(0,"TAO successfully initialized.\n"); CHKERRQ(ierr);

  PetscFunctionReturn(0);
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
  PetscFunctionBegin;
  ierr = TaoFinalize_DynamicLibraries(); CHKERRQ(ierr);
  PetscFunctionReturn(0);
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

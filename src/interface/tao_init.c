#include "taosolver.h"   /*I "tao_solver.h" */

PetscTruth TaoInitializeCalled = PETSC_FALSE;

//static int TaoGlobalArgc=0;
//static char** TaoGlogalArgs = 0;

#undef __FUNCT__
#define __FUNCT__ "TaoInitialize"
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

#define PETSC_DLL

#include "petsc.h"
#include "petscdraw.h"
#include "petscsys.h"

#undef __FUNCT__  
#define __FUNCT__ "PetscInitializePackage" 
/*@C
  PetscInitializePackage - This function initializes everything in the main Petsc package. It is called
  from PetscDLLibraryRegister() when using dynamic libraries, and on the call to PetscInitialize()
  when using static libraries.

  Input Parameter:
  path - The dynamic library path, or PETSC_NULL

  Level: developer

.keywords: Petsc, initialize, package
.seealso: PetscInitialize()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscInitializePackage(char *path)
{
  static PetscTruth initialized = PETSC_FALSE;
  char              logList[256];
  char              *className;
  PetscTruth        opt;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (initialized) PetscFunctionReturn(0);
  initialized = PETSC_TRUE;
  /* Register Classes */
  ierr = PetscLogClassRegister(&PETSC_VIEWER_COOKIE, "Viewer");CHKERRQ(ierr);
  ierr = PetscLogClassRegister(&PETSC_DRAW_COOKIE,   "Draw");CHKERRQ(ierr);
  ierr = PetscLogClassRegister(&DRAWAXIS_COOKIE,     "Axis");CHKERRQ(ierr);
  ierr = PetscLogClassRegister(&DRAWLG_COOKIE,       "Line Graph");CHKERRQ(ierr);
  ierr = PetscLogClassRegister(&DRAWHG_COOKIE,       "Histogram");CHKERRQ(ierr);
  ierr = PetscLogClassRegister(&DRAWSP_COOKIE,       "Scatter Plot");CHKERRQ(ierr);
  ierr = PetscLogClassRegister(&PETSC_RANDOM_COOKIE, "Random Number Generator");CHKERRQ(ierr);
  /* Register Constructors */
  ierr = PetscDrawRegisterAll(path);CHKERRQ(ierr);
  ierr = PetscViewerRegisterAll(path);CHKERRQ(ierr);
  /* Register Events */
  ierr = PetscLogEventRegister(&PETSC_Barrier, "PetscBarrier", PETSC_COOKIE);CHKERRQ(ierr);
  /* Process info exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL, "-log_info_exclude", logList, 256, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList, "null", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogInfoDeactivateClass(0);CHKERRQ(ierr);
    }
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL, "-log_summary_exclude", logList, 256, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList, "null", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogEventDeactivateClass(0);CHKERRQ(ierr);
    }
  }
  /* Setup auxiliary packages */
#if defined(PETSC_HAVE_MATHEMATICA)
  ierr = PetscViewerMathematicaInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif
  ierr = PetscPLAPACKInitializePackage(PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PetscDLLibraryRegister" 
/*
  PetscDLLibraryRegister - This function is called when the dynamic library it is in is opened.

  This one registers all the draw and PetscViewer objects.

  Input Parameter:
  path - library path
 */
PetscErrorCode PETSC_DLLEXPORT PetscDLLibraryRegister(char *path)
{
  PetscErrorCode ierr;

  ierr = PetscInitializeNoArguments(); if (ierr) return 1;
  PetscFunctionBegin;
  /*
      If we got here then PETSc was properly loaded
  */
  ierr = PetscInitializePackage(path);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

/* --------------------------------------------------------------------------*/
static const char *contents = "PETSc Graphics and PetscViewer libraries. \n\
     ASCII, Binary, Sockets, X-windows, ...\n";
static const char *authors  = PETSC_AUTHOR_INFO;

#include "src/sys/src/utils/dlregis.h"

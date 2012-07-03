
/*
       Provides the calling sequences for all the basic PetscDraw routines.
*/
#include <../src/sys/draw/drawimpl.h>  /*I "petscdraw.h" I*/

EXTERN_C_BEGIN
#if defined(PETSC_HAVE_X)
extern PetscErrorCode PetscDrawCreate_X(PetscDraw);
#endif
extern PetscErrorCode PetscDrawCreate_Null(PetscDraw);
#if defined(PETSC_USE_WINDOWS_GRAPHICS)
extern PetscErrorCode PetscDrawCreate_Win32(PetscDraw);
#endif
EXTERN_C_END
  
#undef __FUNCT__  
#define __FUNCT__ "PetscDrawRegisterAll" 
/*@C
  PetscDrawRegisterAll - Registers all of the graphics methods in the PetscDraw package.

  Not Collective

  Level: developer

.seealso:  PetscDrawRegisterDestroy()
@*/
PetscErrorCode  PetscDrawRegisterAll(const char *path)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  
#if defined(PETSC_HAVE_X)
  ierr = PetscDrawRegisterDynamic(PETSC_DRAW_X,     path,"PetscDrawCreate_X",     PetscDrawCreate_X);CHKERRQ(ierr);
#elif defined(PETSC_USE_WINDOWS_GRAPHICS)
  ierr = PetscDrawRegisterDynamic(PETSC_DRAW_WIN32, path,"PetscDrawCreate_Win32", PetscDrawCreate_Win32);CHKERRQ(ierr);
#endif
  ierr = PetscDrawRegisterDynamic(PETSC_DRAW_NULL,  path,"PetscDrawCreate_Null",  PetscDrawCreate_Null);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



/*
       Provides the calling sequences for all the basic PetscDraw routines.
*/
#include <petsc/private/drawimpl.h>  /*I "petscdraw.h" I*/

PETSC_EXTERN PetscErrorCode PetscDrawCreate_Image(PetscDraw);
PETSC_EXTERN PetscErrorCode PetscDrawCreate_TikZ(PetscDraw);
#if defined(PETSC_HAVE_X)
PETSC_EXTERN PetscErrorCode PetscDrawCreate_X(PetscDraw);
#endif
PETSC_EXTERN PetscErrorCode PetscDrawCreate_Null(PetscDraw);
#if defined(PETSC_USE_WINDOWS_GRAPHICS)
PETSC_EXTERN PetscErrorCode PetscDrawCreate_Win32(PetscDraw);
#endif

PetscBool PetscDrawRegisterAllCalled = PETSC_FALSE;

/*@C
  PetscDrawRegisterAll - Registers all of the graphics methods in the PetscDraw package.

  Not Collective

  Level: developer

.seealso: `PetscDrawRegisterDestroy()`
@*/
PetscErrorCode  PetscDrawRegisterAll(void)
{
  PetscFunctionBegin;
  if (PetscDrawRegisterAllCalled) PetscFunctionReturn(0);
  PetscDrawRegisterAllCalled = PETSC_TRUE;

  PetscCall(PetscDrawRegister(PETSC_DRAW_IMAGE,    PetscDrawCreate_Image));
  PetscCall(PetscDrawRegister(PETSC_DRAW_TIKZ,     PetscDrawCreate_TikZ));
#if defined(PETSC_HAVE_X)
  PetscCall(PetscDrawRegister(PETSC_DRAW_X,        PetscDrawCreate_X));
#elif defined(PETSC_USE_WINDOWS_GRAPHICS)
  PetscCall(PetscDrawRegister(PETSC_DRAW_WIN32,    PetscDrawCreate_Win32));
#endif
  PetscCall(PetscDrawRegister(PETSC_DRAW_NULL,     PetscDrawCreate_Null));
  PetscFunctionReturn(0);
}

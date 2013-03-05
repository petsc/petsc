
/*
       Provides the calling sequences for all the basic PetscDraw routines.
*/
#include <petsc-private/drawimpl.h>  /*I "petscdraw.h" I*/

PETSC_EXTERN_C PetscErrorCode PetscDrawCreate_TikZ(PetscDraw);
#if defined(PETSC_HAVE_X)
PETSC_EXTERN_C PetscErrorCode PetscDrawCreate_X(PetscDraw);
#endif
#if defined(PETSC_HAVE_GLUT)
PETSC_EXTERN_C PetscErrorCode PetscDrawCreate_GLUT(PetscDraw);
#endif
#if defined(PETSC_HAVE_OPENGLES)
PETSC_EXTERN_C PetscErrorCode PetscDrawCreate_OpenGLES(PetscDraw);
#endif
PETSC_EXTERN_C PetscErrorCode PetscDrawCreate_Null(PetscDraw);
#if defined(PETSC_USE_WINDOWS_GRAPHICS)
PETSC_EXTERN_C PetscErrorCode PetscDrawCreate_Win32(PetscDraw);
#endif

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
  ierr = PetscDrawRegisterDynamic(PETSC_DRAW_TIKZ,     path,"PetscDrawCreate_TikZ",  PetscDrawCreate_TikZ);CHKERRQ(ierr);
#if defined(PETSC_HAVE_OPENGLES)
  ierr = PetscDrawRegisterDynamic(PETSC_DRAW_OPENGLES, path,"PetscDrawCreate_OpenGLES",  PetscDrawCreate_OpenGLES);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_GLUT)
  ierr = PetscDrawRegisterDynamic(PETSC_DRAW_GLUT,     path,"PetscDrawCreate_GLUT",  PetscDrawCreate_GLUT);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_X)
  ierr = PetscDrawRegisterDynamic(PETSC_DRAW_X,        path,"PetscDrawCreate_X",     PetscDrawCreate_X);CHKERRQ(ierr);
#elif defined(PETSC_USE_WINDOWS_GRAPHICS)
  ierr = PetscDrawRegisterDynamic(PETSC_DRAW_WIN32,    path,"PetscDrawCreate_Win32", PetscDrawCreate_Win32);CHKERRQ(ierr);
#endif
  ierr = PetscDrawRegisterDynamic(PETSC_DRAW_NULL,     path,"PetscDrawCreate_Null",  PetscDrawCreate_Null);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


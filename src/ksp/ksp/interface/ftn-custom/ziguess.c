#include <petsc/private/fortranimpl.h>
#include <petscksp.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define kspguessgettype_ KSPGUESSGETTYPE
#define kspguesssettype_ KSPGUESSSETTYPE
#define kspguessview_    KSPGUESSVIEW
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define kspguessgettype_ kspguessgettype
#define kspguesssettype_ kspguesssettype
#define kspguessview_    kspguessview
#endif

PETSC_EXTERN void PETSC_STDCALL kspguessgettype_(KSPGuess *kspguess,char* name PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  const char *tname;

  *ierr = KSPGuessGetType(*kspguess,&tname);if (*ierr) return;
  *ierr = PetscStrncpy(name,tname,len);
  FIXRETURNCHAR(PETSC_TRUE,name,len);

}

PETSC_EXTERN void PETSC_STDCALL kspguesssettype_(KSPGuess *kspguess,char* type PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = KSPGuessSetType(*kspguess,t);
  FREECHAR(type,t);
}

PETSC_EXTERN void PETSC_STDCALL kspguessview_(KSPGuess *kspguess,PetscViewer *viewer, PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = KSPGuessView(*kspguess,v);
}

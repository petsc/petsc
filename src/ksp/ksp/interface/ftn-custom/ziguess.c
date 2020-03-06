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

PETSC_EXTERN void kspguessgettype_(KSPGuess *kspguess,char* name,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  const char *tname;

  *ierr = KSPGuessGetType(*kspguess,&tname);if (*ierr) return;
  *ierr = PetscStrncpy(name,tname,len);
  FIXRETURNCHAR(PETSC_TRUE,name,len);

}

PETSC_EXTERN void kspguesssettype_(KSPGuess *kspguess,char* type,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = KSPGuessSetType(*kspguess,t);if (*ierr) return;
  FREECHAR(type,t);
}

PETSC_EXTERN void kspguessview_(KSPGuess *kspguess,PetscViewer *viewer, PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = KSPGuessView(*kspguess,v);
}

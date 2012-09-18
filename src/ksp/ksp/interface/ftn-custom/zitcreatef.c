#include <petsc-private/fortranimpl.h>
#include <petscksp.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define kspgettype_                KSPGETTYPE
#define kspsettype_                KSPSETTYPE
#define kspview_                   KSPVIEW
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define kspgettype_                kspgettype
#define kspsettype_                kspsettype
#define kspview_                   kspview
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL kspgettype_(KSP *ksp,CHAR name PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  const char *tname;

  *ierr = KSPGetType(*ksp,&tname);if (*ierr) return;
  *ierr = PetscStrncpy(name,tname,len);
  FIXRETURNCHAR(PETSC_TRUE,name,len);

}

void PETSC_STDCALL kspsettype_(KSP *ksp,CHAR type PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = KSPSetType(*ksp,t);
  FREECHAR(type,t);
}

void PETSC_STDCALL kspview_(KSP *ksp,PetscViewer *viewer, PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = KSPView(*ksp,v);
}

EXTERN_C_END

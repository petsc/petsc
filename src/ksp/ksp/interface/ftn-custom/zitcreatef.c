#include <petsc/private/fortranimpl.h>
#include <petscksp.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define kspgettype_                KSPGETTYPE
#define kspsettype_                KSPSETTYPE
#define kspview_                   KSPVIEW
#define kspviewfromoptions_        kspviewfromoptions
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define kspgettype_                kspgettype
#define kspsettype_                kspsettype
#define kspview_                   kspview
#define kspviewfromoptions_        KSPVIEWFROMOPTIONS
#endif

PETSC_EXTERN void PETSC_STDCALL kspgettype_(KSP *ksp,char* name PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  const char *tname;

  *ierr = KSPGetType(*ksp,&tname);if (*ierr) return;
  *ierr = PetscStrncpy(name,tname,len);
  FIXRETURNCHAR(PETSC_TRUE,name,len);

}

PETSC_EXTERN void PETSC_STDCALL kspsettype_(KSP *ksp,char* type PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = KSPSetType(*ksp,t);if (*ierr) return;
  FREECHAR(type,t);
}

PETSC_EXTERN void PETSC_STDCALL kspview_(KSP *ksp,PetscViewer *viewer, PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = KSPView(*ksp,v);
}

PETSC_EXTERN void PETSC_STDCALL kspviewfromoptions_(KSP *ao,PetscObject obj,char* type PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = KSPViewFromOptions(*ao,obj,t);if (*ierr) return;
  FREECHAR(type,t);
}

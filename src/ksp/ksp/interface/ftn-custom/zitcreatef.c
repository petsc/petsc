#include <petsc/private/fortranimpl.h>
#include <petscksp.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define kspgettype_                KSPGETTYPE
#define kspsettype_                KSPSETTYPE
#define kspview_                   KSPVIEW
#define kspviewfromoptions_        KSPVIEWFROMOPTIONS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define kspgettype_                kspgettype
#define kspsettype_                kspsettype
#define kspview_                   kspview
#define kspviewfromoptions_        kspviewfromoptions
#endif

PETSC_EXTERN void kspgettype_(KSP *ksp,char* name,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  const char *tname;

  *ierr = KSPGetType(*ksp,&tname);if (*ierr) return;
  *ierr = PetscStrncpy(name,tname,len);
  FIXRETURNCHAR(PETSC_TRUE,name,len);

}

PETSC_EXTERN void kspsettype_(KSP *ksp,char* type,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = KSPSetType(*ksp,t);if (*ierr) return;
  FREECHAR(type,t);
}

PETSC_EXTERN void kspview_(KSP *ksp,PetscViewer *viewer, PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = KSPView(*ksp,v);
}

PETSC_EXTERN void kspviewfromoptions_(KSP *ao,PetscObject obj,char* type,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = KSPViewFromOptions(*ao,obj,t);if (*ierr) return;
  FREECHAR(type,t);
}

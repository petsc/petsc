#include <petsc/private/fortranimpl.h>
#include <petscfv.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscfvsetcomponentname_ PETSCFVSETCOMPONENTNAME
#define petscfvview_ PETSCFVVIEW
#define petscfvsettype_ PETSCFVSETTYPE
#define petscfvviewfromoptions_ PETSCFVVIEWFROMOPTIONS
#define petsclimiterviewfromoptions_  PETSCLIMITERVIEWFROMOPTIONS
#define petsclimitersettype_ PETSCLIMITERSETTYPE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscfvsetcomponentname_ petscfvsetcomponentname
#define petscfvview_ petscfvview
#define petscfvsettype_ petscfvsettype
#define petscfvviewfromoptions_ petscfvviewfromoptions
#define petsclimiterviewfromoptions_  petsclimiterviewfromoptions
#define petsclimitersettype_  petsclimitersettype
#endif

PETSC_EXTERN void petscfvsetcomponentname_(PetscFV *fvm,PetscInt *comp,char* name,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *compname;

  FIXCHAR(name,len,compname);
  *ierr = PetscFVSetComponentName(*fvm,*comp,compname);if (*ierr) return;
  FREECHAR(name,compname);
}

PETSC_EXTERN void petscfvview_(PetscFV *fvm,PetscViewer *vin,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(vin,v);
  *ierr = PetscFVView(*fvm,v);
}

PETSC_EXTERN void petscfvsettype_(PetscFV *fvm,char* type_name,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(type_name,len,t);
  *ierr = PetscFVSetType(*fvm,t);if (*ierr) return;
  FREECHAR(type_name,t);
}

PETSC_EXTERN void petscfvviewfromoptions_(PetscFV *ao,PetscObject obj,char* type,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = PetscFVViewFromOptions(*ao,obj,t);if (*ierr) return;
  FREECHAR(type,t);
}

PETSC_EXTERN void petsclimiterviewfromoptions_(PetscLimiter *ao,PetscObject obj,char* type,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = PetscLimiterViewFromOptions(*ao,obj,t);if (*ierr) return;
  FREECHAR(type,t);
}

PETSC_EXTERN void petsclimitersettype_(PetscLimiter *lim,char *name,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T namelen)
{
  char *newname;

  FIXCHAR(name,namelen,newname);
  *ierr = PetscLimiterSetType(*lim,newname);if (*ierr) return;
  FREECHAR(name,newname);
}
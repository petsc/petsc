#include <petsc/private/fortranimpl.h>
#include <petscfe.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscspaceviewfromoptions_   PETSCSPACEVIEWFROMOPTIONS
#define petscdualspaceviewfromoptions_   PETSCDUALSPACEVIEWFROMOPTIONS
#define petscfeviewfromoptions_   PETSCFEVIEWFROMOPTIONS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscspaceviewfromoptions_   petscspaceviewfromoptions
#define petscdualspaceviewfromoptions_   petscdualspaceviewfromoptions
#define petscfeviewfromoptions_   petscfeviewfromoptions
#endif

PETSC_EXTERN void petscspaceviewfromoptions_(PetscSpace *ao,PetscObject obj,char* type,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = PetscSpaceViewFromOptions(*ao,obj,t);if (*ierr) return;
  FREECHAR(type,t);
}

PETSC_EXTERN void petscdualspaceviewfromoptions_(PetscDualSpace *ao,PetscObject obj,char* type,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = PetscDualSpaceViewFromOptions(*ao,obj,t);if (*ierr) return;
  FREECHAR(type,t);
}

PETSC_EXTERN void petscfeviewfromoptions_(PetscFE *ao,PetscObject obj,char* type,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = PetscFEViewFromOptions(*ao,obj,t);if (*ierr) return;
  FREECHAR(type,t);
}



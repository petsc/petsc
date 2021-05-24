#include <petsc/private/fortranimpl.h>
#include <petscpc.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define pcfactorsetmatorderingtype_  PCFACTORSETMATORDERINGTYPE
#define pcfactorsetmatsolvertype_ PCFACTORSETMATSOLVERTYPE
#define pcfactorgetmatsolvertype_ PCFACTORGETMATSOLVERTYPE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define pcfactorsetmatorderingtype_  pcfactorsetmatorderingtype
#define pcfactorsetmatsolvertype_ pcfactorsetmatsolvertype
#define pcfactorgetmatsolvertype_ pcfactorgetmatsolvertype
#endif

PETSC_EXTERN void pcfactorsetmatorderingtype_(PC *pc,char* ordering, PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(ordering,len,t);
  *ierr = PCFactorSetMatOrderingType(*pc,t);if (*ierr) return;
  FREECHAR(ordering,t);
}
PETSC_EXTERN void pcfactorsetmatsolvertype_(PC *pc,char* ordering, PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(ordering,len,t);
  *ierr = PCFactorSetMatSolverType(*pc,t);if (*ierr) return;
  FREECHAR(ordering,t);
}
PETSC_EXTERN void pcfactorgetmatsolvertype_(PC *mat,char* name,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  const char *tname;

  *ierr = PCFactorGetMatSolverType(*mat,&tname);if (*ierr) return;
  if (name != PETSC_NULL_CHARACTER_Fortran) {
    *ierr = PetscStrncpy(name,tname,len);if (*ierr) return;
  }
  FIXRETURNCHAR(PETSC_TRUE,name,len);
}

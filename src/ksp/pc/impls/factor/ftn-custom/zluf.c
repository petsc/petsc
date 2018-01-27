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

PETSC_EXTERN void PETSC_STDCALL pcfactorsetmatorderingtype_(PC *pc,char* ordering PETSC_MIXED_LEN(len), PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(ordering,len,t);
  *ierr = PCFactorSetMatOrderingType(*pc,t);
  FREECHAR(ordering,t);
}
PETSC_EXTERN void PETSC_STDCALL pcfactorsetmatsolvertype_(PC *pc,char* ordering PETSC_MIXED_LEN(len), PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(ordering,len,t);
  *ierr = PCFactorSetMatSolverType(*pc,t);
  FREECHAR(ordering,t);
}
PETSC_EXTERN void PETSC_STDCALL pcfactorgetmatsolvertype_(PC *mat,char* name PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  const char *tname;

  *ierr = PCFactorGetMatSolverType(*mat,&tname);if (*ierr) return;
  if (name != PETSC_NULL_CHARACTER_Fortran) {
    *ierr = PetscStrncpy(name,tname,len);if (*ierr) return;
  }
  FIXRETURNCHAR(PETSC_TRUE,name,len);
}

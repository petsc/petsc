#include <petsc/private/fortranimpl.h>
#include <petscpc.h>


#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define pcfactorsetmatorderingtype_  PCFACTORSETMATORDERINGTYPE
#define pcfactorsetmatsolverpackage_ PCFACTORSETMATSOLVERPACKAGE
#define pcfactorgetmatsolverpackage_ PCFACTORGETMATSOLVERPACKAGE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define pcfactorsetmatorderingtype_  pcfactorsetmatorderingtype
#define pcfactorsetmatsolverpackage_ pcfactorsetmatsolverpackage
#define pcfactorgetmatsolverpackage_ pcfactorgetmatsolverpackage
#endif

PETSC_EXTERN void PETSC_STDCALL pcfactorsetmatorderingtype_(PC *pc,char* ordering PETSC_MIXED_LEN(len), PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(ordering,len,t);
  *ierr = PCFactorSetMatOrderingType(*pc,t);
  FREECHAR(ordering,t);
}
PETSC_EXTERN void PETSC_STDCALL pcfactorsetmatsolverpackage_(PC *pc,char* ordering PETSC_MIXED_LEN(len), PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(ordering,len,t);
  *ierr = PCFactorSetMatSolverPackage(*pc,t);
  FREECHAR(ordering,t);
}
PETSC_EXTERN void PETSC_STDCALL pcfactorgetmatsolverpackage_(PC *mat,char* name PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  const char *tname;

  *ierr = PCFactorGetMatSolverPackage(*mat,&tname);if (*ierr) return;
  if (name != PETSC_NULL_CHARACTER_Fortran) {
    *ierr = PetscStrncpy(name,tname,len);if (*ierr) return;
  }
  FIXRETURNCHAR(PETSC_TRUE,name,len);
}

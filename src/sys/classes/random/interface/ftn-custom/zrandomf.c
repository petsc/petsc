#include <petsc/private/fortranimpl.h>
#include <petscsys.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscrandomsettype_                PETSCRANDOMSETTYPE
#define petscrandomgettype_                PETSCRANDOMGETTYPE
#define petscrandomsetseed_                PETSCRANDOMSETSEED
#define petscrandomgetseed_                PETSCRANDOMGETSEED
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscrandomsettype_                petscrandomsettype
#define petscrandomgettype_                petscrandomgettype
#define petscrandomsetseed_                petscrandomsetseed
#define petscrandomgetseed_                petscrandomgetseed
#endif

PETSC_EXTERN void PETSC_STDCALL  petscrandomgetseed_(PetscRandom *r,unsigned long *seed, PetscErrorCode *ierr )
{
  *ierr = PetscRandomGetSeed(*r,seed);
}
PETSC_EXTERN void PETSC_STDCALL  petscrandomsetseed_(PetscRandom *r,unsigned long *seed, PetscErrorCode *ierr )
{
  *ierr = PetscRandomSetSeed(*r,*seed);
}

PETSC_EXTERN void PETSC_STDCALL petscrandomsettype_(PetscRandom *rnd,char* type PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = PetscRandomSetType(*rnd,t);
  FREECHAR(type,t);
}

PETSC_EXTERN void PETSC_STDCALL petscrandomgettype_(PetscRandom *petscrandom,char* name PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  const char *tname;

  *ierr = PetscRandomGetType(*petscrandom,&tname);if (*ierr) return;
  *ierr = PetscStrncpy(name,tname,len);
  FIXRETURNCHAR(PETSC_TRUE,name,len);

}

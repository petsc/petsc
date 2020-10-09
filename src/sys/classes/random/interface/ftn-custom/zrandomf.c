#include <petsc/private/fortranimpl.h>
#include <petscsys.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscrandomsettype_                PETSCRANDOMSETTYPE
#define petscrandomgettype_                PETSCRANDOMGETTYPE
#define petscrandomsetseed_                PETSCRANDOMSETSEED
#define petscrandomgetseed_                PETSCRANDOMGETSEED
#define petscrandomviewfromoptions_        PETSCRANDOMVIEWFROMOPTIONS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscrandomsettype_                petscrandomsettype
#define petscrandomgettype_                petscrandomgettype
#define petscrandomsetseed_                petscrandomsetseed
#define petscrandomgetseed_                petscrandomgetseed
#define petscrandomviewfromoptions_        petscrandomviewfromoptions
#endif

PETSC_EXTERN void  petscrandomgetseed_(PetscRandom *r,unsigned long *seed, PetscErrorCode *ierr)
{
  *ierr = PetscRandomGetSeed(*r,seed);
}
PETSC_EXTERN void  petscrandomsetseed_(PetscRandom *r,unsigned long *seed, PetscErrorCode *ierr)
{
  *ierr = PetscRandomSetSeed(*r,*seed);
}

PETSC_EXTERN void petscrandomsettype_(PetscRandom *rnd,char* type,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = PetscRandomSetType(*rnd,t);if (*ierr) return;
  FREECHAR(type,t);
}

PETSC_EXTERN void petscrandomgettype_(PetscRandom *petscrandom,char* name,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  const char *tname;

  *ierr = PetscRandomGetType(*petscrandom,&tname);if (*ierr) return;
  *ierr = PetscStrncpy(name,tname,len);if (*ierr) return;
  FIXRETURNCHAR(PETSC_TRUE,name,len);

}
PETSC_EXTERN void petscrandomviewfromoptions_(PetscRandom *ao,PetscObject obj,char* type,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = PetscRandomViewFromOptions(*ao,obj,t);if (*ierr) return;
  FREECHAR(type,t);
}

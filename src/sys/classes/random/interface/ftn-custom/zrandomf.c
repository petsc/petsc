#include <petsc-private/fortranimpl.h>
#include <petscsys.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscrandomsettype_                PETSCRANDOMSETTYPE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscrandomsettype_                petscrandomsettype
#endif

void PETSC_STDCALL petscrandomsettype_(PetscRandom *rnd,CHAR type PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = PetscRandomSetType(*rnd,t);
  FREECHAR(type,t);
}

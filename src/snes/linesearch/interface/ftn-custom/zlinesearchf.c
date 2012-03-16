#include <private/fortranimpl.h>
#include <petscsnes.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petsclinesearchsettype_          PETSCLINESEARCHSETTYPE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petsclinesearchsettype_          petsclinesearchsettype
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL petsclinesearchsettype_(PetscLineSearch *linesearch,CHAR type PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = PetscLineSearchSetType(*linesearch,t);
  FREECHAR(type,t);
}
EXTERN_C_END

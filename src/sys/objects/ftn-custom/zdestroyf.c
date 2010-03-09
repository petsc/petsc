#include "private/fortranimpl.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petsctypecompare_          PETSCTYPECOMPARE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petsctypecompare_          petsctypecompare
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL petsctypecompare_(PetscObject *obj,CHAR type_name PETSC_MIXED_LEN(len),
                                     PetscTruth *same,PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *c1;
  FIXCHAR(type_name,len,c1);
  *ierr = PetscTypeCompare(*obj,c1,same);
  FREECHAR(type_name,c1);
}


EXTERN_C_END

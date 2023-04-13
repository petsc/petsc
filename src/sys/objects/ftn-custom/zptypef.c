#include <petsc/private/fortranimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define petscdatatypegetsize_ PETSCDATATYPEGETSIZE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define petscdatatypegetsize_ petscdatatypegetsize
#endif

PETSC_EXTERN void petscdatatypegetsize_(PetscDataType *ptype, size_t *size, PetscErrorCode *ierr)
{
  *ierr = PetscDataTypeGetSize(*ptype, size);
}

#include <petscis.h>
#include <petsc/private/ftnimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define isblockgetindices_     ISBLOCKGETINDICES
  #define isblockrestoreindices_ ISBLOCKRESTOREINDICES
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define isblockgetindices_     isblockgetindices
  #define isblockrestoreindices_ isblockrestoreindices
#endif

PETSC_EXTERN void isblockgetindices_(IS *x, F90Array1d *ptr, int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscInt *fa;
  PetscInt        len;
  *__ierr = ISBlockGetIndices(*x, &fa);
  if (*__ierr) return;
  *__ierr = ISBlockGetLocalSize(*x, &len);
  if (*__ierr) return;
  *__ierr = F90Array1dCreate((void *)fa, MPIU_INT, 1, len, ptr PETSC_F90_2PTR_PARAM(ptrd));
}
PETSC_EXTERN void isblockrestoreindices_(IS *x, F90Array1d *ptr, int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscInt *fa;
  *__ierr = F90Array1dAccess(ptr, MPIU_INT, (void **)&fa PETSC_F90_2PTR_PARAM(ptrd));
  if (*__ierr) return;
  *__ierr = F90Array1dDestroy(ptr, MPIU_INT PETSC_F90_2PTR_PARAM(ptrd));
  if (*__ierr) return;
  *__ierr = ISBlockRestoreIndices(*x, &fa);
}

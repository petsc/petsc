#include <petscis.h>
#include <petsc/private/ftnimpl.h>

#if PetscDefined(HAVE_FORTRAN_CAPS)
  #define isblockgetindices_     ISBLOCKGETINDICES
  #define isblockrestoreindices_ ISBLOCKRESTOREINDICES
#elif !PetscDefined(HAVE_FORTRAN_UNDERSCORE)
  #define isblockgetindices_     isblockgetindices
  #define isblockrestoreindices_ isblockrestoreindices
#endif

PETSC_EXTERN void isblockgetindices_(IS *is, F90Array1d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscInt *fa;
  PetscInt        len;
  *ierr = ISBlockGetIndices(*is, &fa);
  if (*ierr) return;
  *ierr = ISBlockGetLocalSize(*is, &len);
  if (*ierr) return;
  *ierr = F90Array1dCreate((void *)fa, MPIU_INT, 1, len, ptr PETSC_F90_2PTR_PARAM(ptrd));
}
PETSC_EXTERN void isblockrestoreindices_(IS *is, F90Array1d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscInt *fa;
  *ierr = F90Array1dAccess(ptr, MPIU_INT, (void **)&fa PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
  *ierr = F90Array1dDestroy(ptr, MPIU_INT PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
  *ierr = ISBlockRestoreIndices(*is, &fa);
}

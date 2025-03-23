#include <petscis.h>
#include <petsc/private/ftnimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define petsclayoutgetranges_     PETSCLAYOUTGETRANGES
  #define petsclayoutrestoreranges_ PETSCLAYOUTRESTORERANGES
  #define isgetindices_             ISGETINDICES
  #define isrestoreindices_         ISRESTOREINDICES
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define petsclayoutgetranges_     petsclayoutgetranges
  #define petsclayoutrestoreranges_ petsclayoutrestoreranges
  #define isgetindices_             isgetindices
  #define isrestoreindices_         isrestoreindices
#endif

PETSC_EXTERN void petsclayoutgetranges_(PetscLayout *map, F90Array1d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscInt *fa;
  PetscMPIInt     size;

  *ierr = PetscLayoutGetRanges(*map, &fa);
  if (*ierr) return;
  *ierr = MPI_Comm_size((*map)->comm, &size);
  if (*ierr) return;
  *ierr = F90Array1dCreate((void *)fa, MPIU_INT, 1, (PetscInt)size + 1, ptr PETSC_F90_2PTR_PARAM(ptrd));
}

PETSC_EXTERN void petsclayoutrestoreranges_(PetscLayout *map, F90Array1d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  *ierr = F90Array1dDestroy(ptr, MPIU_INT PETSC_F90_2PTR_PARAM(ptrd));
}

PETSC_EXTERN void isgetindices_(IS *x, F90Array1d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscInt *fa;
  PetscInt        len;

  *ierr = ISGetIndices(*x, &fa);
  if (*ierr) return;
  *ierr = ISGetLocalSize(*x, &len);
  if (*ierr) return;
  *ierr = F90Array1dCreate((void *)fa, MPIU_INT, 1, len, ptr PETSC_F90_2PTR_PARAM(ptrd));
}
PETSC_EXTERN void isrestoreindices_(IS *x, F90Array1d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscInt *fa;

  *ierr = F90Array1dAccess(ptr, MPIU_INT, (void **)&fa PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
  *ierr = F90Array1dDestroy(ptr, MPIU_INT PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
  *ierr = ISRestoreIndices(*x, &fa);
}

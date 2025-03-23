#include <petscis.h>
#include <petsc/private/ftnimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define islocaltoglobalmappinggetindices_          ISLOCALTOGLOBALMAPPINGGETINDICES
  #define islocaltoglobalmappingrestoreindices_      ISLOCALTOGLOBALMAPPINGRESTOREINDICES
  #define islocaltoglobalmappinggetblockindices_     ISLOCALTOGLOBALMAPPINGGETBLOCKINDICES
  #define islocaltoglobalmappingrestoreblockindices_ ISLOCALTOGLOBALMAPPINGRESTOREBLOCKINDICES
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define islocaltoglobalmappinggetindices_          islocaltoglobalmappinggetindices
  #define islocaltoglobalmappingrestoreindices_      islocaltoglobalmappingrestoreindices
  #define islocaltoglobalmappinggetblockindices_     islocaltoglobalmappinggetblockindices
  #define islocaltoglobalmappingrestoreblockindices_ islocaltoglobalmappingrestoreblockindices
#endif

PETSC_EXTERN void islocaltoglobalmappinggetindices_(ISLocalToGlobalMapping *da, F90Array1d *indices, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscInt *idx;
  PetscInt        n;
  *ierr = ISLocalToGlobalMappingGetIndices(*da, &idx);
  if (*ierr) return;
  *ierr = ISLocalToGlobalMappingGetSize(*da, &n);
  if (*ierr) return;
  *ierr = F90Array1dCreate((void *)idx, MPIU_INT, 1, n, indices PETSC_F90_2PTR_PARAM(ptrd));
}

PETSC_EXTERN void islocaltoglobalmappingrestoreindices_(ISLocalToGlobalMapping *da, F90Array1d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscInt *fa;

  *ierr = F90Array1dAccess(ptr, MPIU_INT, (void **)&fa PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
  *ierr = F90Array1dDestroy(ptr, MPIU_INT PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
  *ierr = ISLocalToGlobalMappingRestoreIndices(*da, &fa);
  if (*ierr) return;
}

PETSC_EXTERN void islocaltoglobalmappinggetblockindices_(ISLocalToGlobalMapping *da, F90Array1d *indices, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscInt *idx;
  PetscInt        n;
  *ierr = ISLocalToGlobalMappingGetBlockIndices(*da, &idx);
  if (*ierr) return;
  *ierr = ISLocalToGlobalMappingGetSize(*da, &n);
  if (*ierr) return;
  *ierr = F90Array1dCreate((void *)idx, MPIU_INT, 1, n, indices PETSC_F90_2PTR_PARAM(ptrd));
}

PETSC_EXTERN void islocaltoglobalmappingrestoreblockindices_(ISLocalToGlobalMapping *da, F90Array1d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  const PetscInt *fa;

  *ierr = F90Array1dAccess(ptr, MPIU_INT, (void **)&fa PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
  *ierr = F90Array1dDestroy(ptr, MPIU_INT PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
  *ierr = ISLocalToGlobalMappingRestoreBlockIndices(*da, &fa);
  if (*ierr) return;
}

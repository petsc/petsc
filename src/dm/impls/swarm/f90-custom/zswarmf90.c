#include <petsc/private/fortranimpl.h>
#include <petscdmswarm.h>
#include <petsc/private/f90impl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define dmswarmgetfield_     DMSWARMGETFIELD
  #define dmswarmrestorefield_ DMSWARMRESTOREFIELD
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
  #define dmswarmgetfield_     dmswarmgetfield
  #define dmswarmrestorefield_ dmswarmerstorefield
#endif

/* Definitions of Fortran Wrapper routines */

PETSC_EXTERN void dmswarmgetfield_(DM *dm, char *name, PetscInt *blocksize, PetscDataType *type, F90Array1d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd), PETSC_FORTRAN_CHARLEN_T lenN)
{
  PetscScalar *v;
  PetscInt     n;
  char        *fieldname;

  FIXCHAR(name, lenN, fieldname);
  *ierr = DMSwarmGetSize(*dm, &n);
  if (*ierr) return;
  *ierr = DMSwarmGetField(*dm, fieldname, blocksize, type, (void **)&v);
  if (*ierr) return;
  *ierr = F90Array1dCreate((void *)v, MPIU_SCALAR, 1, n, ptr PETSC_F90_2PTR_PARAM(ptrd));
  FREECHAR(name, fieldname);
}

PETSC_EXTERN void dmswarmrestorefield_(DM *dm, char *name, PetscInt *blocksize, PetscDataType *type, F90Array1d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd), PETSC_FORTRAN_CHARLEN_T lenN)
{
  PetscScalar *v;
  char        *fieldname;

  FIXCHAR(name, lenN, fieldname);
  *ierr = F90Array1dAccess(ptr, MPIU_SCALAR, (void **)&v PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
  *ierr = DMSwarmRestoreField(*dm, fieldname, blocksize, type, (void **)&v);
  if (*ierr) return;
  *ierr = F90Array1dDestroy(ptr, MPIU_SCALAR PETSC_F90_2PTR_PARAM(ptrd));
  FREECHAR(name, fieldname);
}

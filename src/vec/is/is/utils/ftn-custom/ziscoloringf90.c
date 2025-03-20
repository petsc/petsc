#include <petscis.h>
#include <petsc/private/ftnimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define iscoloringgetis_     ISCOLORINGGETIS
  #define iscoloringrestoreis_ ISCOLORINGRESTOREIS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define iscoloringgetis_     iscoloringgetis
  #define iscoloringrestoreis_ iscoloringrestoreis
#endif

PETSC_EXTERN void iscoloringgetis_(ISColoring *iscoloring, PetscCopyMode *mode, PetscInt *n, F90Array1d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  IS *lis;

  *ierr = ISColoringGetIS(*iscoloring, *mode, n, &lis);
  if (*ierr) return;
  *ierr = F90Array1dCreate(lis, MPIU_FORTRANADDR, 1, *n, ptr PETSC_F90_2PTR_PARAM(ptrd));
}

PETSC_EXTERN void iscoloringrestoreis_(ISColoring *iscoloring, PetscCopyMode *mode, F90Array1d *ptr, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  IS *is;

  *ierr = F90Array1dAccess(ptr, MPIU_FORTRANADDR, (void **)&is PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
  *ierr = F90Array1dDestroy(ptr, MPIU_FORTRANADDR PETSC_F90_2PTR_PARAM(ptrd));
  if (*ierr) return;
  *ierr = ISColoringRestoreIS(*iscoloring, *mode, &is);
}

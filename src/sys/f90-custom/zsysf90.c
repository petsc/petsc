#include <petsc/private/f90impl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscintarray1ddestroyf90_         PETSCINTARRAY1DDESTROYF90
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscintarray1ddestroyf90_         petscintarray1ddestroyf90
#endif

PETSC_EXTERN void petscintarray1ddestroyf90_(F90Array1d *ptr, int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscInt         *fa;

  *__ierr = F90Array1dAccess(ptr,MPIU_INT,(void**)&fa PETSC_F90_2PTR_PARAM(ptrd));if (*__ierr) return;
  *__ierr = F90Array1dDestroy(ptr,MPIU_INT PETSC_F90_2PTR_PARAM(ptrd));if (*__ierr) return;
  *__ierr = PetscFree(fa);
}

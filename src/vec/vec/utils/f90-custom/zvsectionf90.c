#include <petscvec.h>
#include <petscsection.h>
#include <petsc/private/f90impl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define vecsetvaluessectionf90_     VECSETVALUESSECTIONF90
#define vecgetvaluessectionf90_     VECGETVALUESSECTIONF90
#define vecrestorevaluessectionf90_ VECRESTOREVALUESSECTIONF90
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define vecsetvaluessectionf90_     vecsetvaluessectionf90
#define vecgetvaluessectionf90_     vecgetvaluessectionf90
#define vecrestorevaluessectionf90_ vecrestorevaluessectionf90
#endif

/* Definitions of Fortran Wrapper routines */

PETSC_EXTERN void vecsetvaluessectionf90_(Vec *v, PetscSection *section, PetscInt *point, F90Array1d *ptr, InsertMode *mode, int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *array;

  *__ierr = F90Array1dAccess(ptr, MPIU_SCALAR, (void**) &array PETSC_F90_2PTR_PARAM(ptrd));if (*__ierr) return;
  *__ierr = VecSetValuesSection(*v, *section, *point, array, *mode);
}

PETSC_EXTERN void vecgetvaluessectionf90_(Vec *v, PetscSection *section, PetscInt *point, F90Array1d *ptr, int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *fa;
  PetscInt    len;

  *__ierr = VecGetValuesSection(*v, *section, *point, &fa);if (*__ierr) return;
  *__ierr = PetscSectionGetDof(*section,*point,&len);if (*__ierr) return;
  *__ierr = F90Array1dCreate(fa,MPIU_SCALAR,1,len,ptr PETSC_F90_2PTR_PARAM(ptrd));
}

PETSC_EXTERN void vecrestorevaluessectionf90_(Vec *v, PetscSection *section, PetscInt *point, F90Array1d *ptr, int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *fa;

  *__ierr = F90Array1dAccess(ptr,MPIU_SCALAR,(void**)&fa PETSC_F90_2PTR_PARAM(ptrd));if (*__ierr) return;
  *__ierr = F90Array1dDestroy(ptr,MPIU_SCALAR PETSC_F90_2PTR_PARAM(ptrd));if (*__ierr) return;
}

#include <petsc-private/vecimpl.h>
#include <../src/sys/f90-src/f90impl.h>

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define vecsetvaluessection_ VECSETVALUESSECTION
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define vecsetvaluessection_ vecsetvaluessection
#endif

/* Definitions of Fortran Wrapper routines */
EXTERN_C_BEGIN

void PETSC_STDCALL vecsetvaluessection_(Vec *v, PetscSection *section, PetscInt *point, F90Array1d *ptr, InsertMode *mode, int *__ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscScalar *array;

  *__ierr = F90Array1dAccess(ptr, PETSC_SCALAR, (void **) &array PETSC_F90_2PTR_PARAM(ptrd));if (*__ierr) return;
  *__ierr = VecSetValuesSection(*v, *section, *point, array, *mode);
}

EXTERN_C_END

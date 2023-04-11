
#include <petscksp.h>
#include <petsc/private/f90impl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define kspgetresidualhistoryf90_ KSPGETRESIDUALHISTORYF90
  #define kspdestroy_               KSPDESTROY
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define kspgetresidualhistoryf90_ kspgetresidualhistoryf90
  #define kspdestroy_               kspdestroy
#endif

PETSC_EXTERN void kspgetresidualhistoryf90_(KSP *ksp, F90Array1d *indices, PetscInt *n, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscReal const *hist;
  *ierr = KSPGetResidualHistory(*ksp, &hist, n);
  if (*ierr) return;
  *ierr = F90Array1dCreate((void *)hist, MPIU_REAL, 1, *n, indices PETSC_F90_2PTR_PARAM(ptrd));
}

PETSC_EXTERN void kspdestroy_(KSP *x, int *ierr)
{
  PETSC_FORTRAN_OBJECT_F_DESTROYED_TO_C_NULL(x);
  *ierr = KSPDestroy(x);
  if (*ierr) return;
  PETSC_FORTRAN_OBJECT_C_NULL_TO_F_DESTROYED(x);
}

#include <petscksp.h>
#include <petsc/private/ftnimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define kspgetresidualhistory_ KSPGETRESIDUALHISTORY
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define kspgetresidualhistory_ kspgetresidualhistory
#endif

PETSC_EXTERN void kspgetresidualhistory_(KSP *ksp, F90Array1d *indices, PetscInt *n, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscReal const *hist;
  *ierr = KSPGetResidualHistory(*ksp, &hist, n);
  if (*ierr) return;
  *ierr = F90Array1dCreate((void *)hist, MPIU_REAL, 1, *n, indices PETSC_F90_2PTR_PARAM(ptrd));
}

PETSC_EXTERN void ksprestoreresidualhistory_(KSP *ksp, F90Array1d *indices, PetscInt *n, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  *ierr = F90Array1dDestroy(indices, MPIU_SCALAR PETSC_F90_2PTR_PARAM(ptrd));
}

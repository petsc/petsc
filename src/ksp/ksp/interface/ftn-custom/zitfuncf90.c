#include <petscksp.h>
#include <petsc/private/ftnimpl.h>

#if PetscDefined(HAVE_FORTRAN_CAPS)
  #define kspgetresidualhistory_     KSPGETRESIDUALHISTORY
  #define ksprestoreresidualhistory_ KSPRESTORERESIDUALHISTORY
#elif !PetscDefined(HAVE_FORTRAN_UNDERSCORE)
  #define kspgetresidualhistory_     kspgetresidualhistory
  #define ksprestoreresidualhistory_ ksprestoreresidualhistory
#endif

PETSC_EXTERN void kspgetresidualhistory_(KSP *ksp, F90Array1d *a, PetscInt *na, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscReal const *hist;
  *ierr = KSPGetResidualHistory(*ksp, &hist, na);
  if (*ierr) return;
  *ierr = F90Array1dCreate((void *)hist, MPIU_REAL, 1, *na, a PETSC_F90_2PTR_PARAM(ptrd));
}

PETSC_EXTERN void ksprestoreresidualhistory_(KSP *ksp, F90Array1d *a, PetscInt *na, int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  *ierr = F90Array1dDestroy(a, MPIU_REAL PETSC_F90_2PTR_PARAM(ptrd));
}

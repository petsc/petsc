#include <petscsnes.h>
#include <petsc/private/ftnimpl.h>

#if PetscDefined(HAVE_FORTRAN_CAPS)
  #define snesgetconvergencehistory_     SNESGETCONVERGENCEHISTORY
  #define snesrestoreconvergencehistory_ SNESRESTORECONVERGENCEHISTORY
#elif !PetscDefined(HAVE_FORTRAN_UNDERSCORE)
  #define snesgetconvergencehistory_     snesgetconvergencehistory
  #define snesrestoreconvergencehistory_ snesrestoreconvergencehistory
#endif

PETSC_EXTERN void snesgetconvergencehistory_(SNES *snes, F90Array1d *a, F90Array1d *its, PetscInt *na, int *ierr PETSC_F90_2PTR_PROTO(ptrd1) PETSC_F90_2PTR_PROTO(ptrd2))
{
  PetscReal *hist;
  PetscInt  *itsv, N;

  CHKFORTRANNULLINTEGER(na);
  *ierr = SNESGetConvergenceHistory(*snes, &hist, &itsv, &N);
  if (*ierr) return;
  *ierr = F90Array1dCreate(hist, MPIU_REAL, 1, N, a PETSC_F90_2PTR_PARAM(ptrd1));
  if (*ierr) return;
  *ierr = F90Array1dCreate(itsv, MPIU_INT, 1, N, its PETSC_F90_2PTR_PARAM(ptrd2));
  if (na) *na = N;
}

PETSC_EXTERN void snesrestoreconvergencehistory_(SNES *snes, F90Array1d *a, F90Array1d *its, PetscInt *na, int *ierr PETSC_F90_2PTR_PROTO(ptrd1) PETSC_F90_2PTR_PROTO(ptrd2))
{
  *ierr = F90Array1dDestroy(a, MPIU_REAL PETSC_F90_2PTR_PARAM(ptrd1));
  if (*ierr) return;
  *ierr = F90Array1dDestroy(its, MPIU_INT PETSC_F90_2PTR_PARAM(ptrd2));
}

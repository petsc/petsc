
#include <petscsnes.h>
#include <petsc/private/f90impl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define snesgetconvergencehistoryf90_ SNESGETCONVERGENCEHISTORYF90
  #define snesdestroy_                  SNESDESTROY
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define snesgetconvergencehistoryf90_ snesgetconvergencehistoryf90
  #define snesdestroy_                  snesdestroy
#endif

PETSC_EXTERN void snesgetconvergencehistoryf90_(SNES *snes, F90Array1d *r, F90Array1d *fits, PetscInt *n, int *ierr PETSC_F90_2PTR_PROTO(ptrd1) PETSC_F90_2PTR_PROTO(ptrd2))
{
  PetscReal *hist;
  PetscInt  *its;
  *ierr = SNESGetConvergenceHistory(*snes, &hist, &its, n);
  if (*ierr) return;
  *ierr = F90Array1dCreate(hist, MPIU_REAL, 1, *n, r PETSC_F90_2PTR_PARAM(ptrd1));
  if (*ierr) return;
  *ierr = F90Array1dCreate(its, MPIU_INT, 1, *n, fits PETSC_F90_2PTR_PARAM(ptrd2));
}

PETSC_EXTERN void snesdestroy_(SNES *x, int *ierr)
{
  PETSC_FORTRAN_OBJECT_F_DESTROYED_TO_C_NULL(x);
  *ierr = SNESDestroy(x);
  if (*ierr) return;
  PETSC_FORTRAN_OBJECT_C_NULL_TO_F_DESTROYED(x);
}

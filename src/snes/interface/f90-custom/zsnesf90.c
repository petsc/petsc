
#include <petscsnes.h>
#include <petsc/private/f90impl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define snesgetconvergencehistoryf90_     SNESGETCONVERGENCEHISTORYF90
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define snesgetconvergencehistoryf90_     snesgetconvergencehistoryf90
#endif

PETSC_EXTERN void snesgetconvergencehistoryf90_(SNES *snes,F90Array1d *r,F90Array1d *fits,PetscInt *n,int *ierr PETSC_F90_2PTR_PROTO(ptrd1) PETSC_F90_2PTR_PROTO(ptrd2))
{
  PetscReal *hist;
  PetscInt  *its;
  *ierr = SNESGetConvergenceHistory(*snes,&hist,&its,n); if (*ierr) return;
  *ierr = F90Array1dCreate(hist,MPIU_REAL,1,*n,r PETSC_F90_2PTR_PARAM(ptrd1)); if (*ierr) return;
  *ierr = F90Array1dCreate(its,MPIU_INT,1,*n,fits PETSC_F90_2PTR_PARAM(ptrd2));
}


#include <petscsnes.h>
#include <../src/sys/f90-src/f90impl.h>

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define snesgetconvergencehistoryf90_     SNESGETCONVERGENCEHISTORYF90
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define snesgetconvergencehistoryf90_     snesgetconvergencehistoryf90
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL snesgetconvergencehistoryf90_(SNES *snes,F90Array1d *r,F90Array1d *fits,PetscInt *n,int *ierr PETSC_F90_2PTR_PROTO(ptrd1) PETSC_F90_2PTR_PROTO(ptrd2))
{
  PetscReal *hist;
  PetscInt  *its;
  *ierr = SNESGetConvergenceHistory(*snes,&hist,&its,n); if (*ierr) return;
  *ierr = F90Array1dCreate(hist,PETSC_DOUBLE,1,*n,r PETSC_F90_2PTR_PARAM(ptrd1)); if (*ierr) return;
  *ierr = F90Array1dCreate(its,PETSC_INT,1,*n,fits PETSC_F90_2PTR_PARAM(ptrd2));
}
EXTERN_C_END

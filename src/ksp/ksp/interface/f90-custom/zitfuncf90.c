
#include <petscksp.h>
#include <petsc/private/f90impl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define kspgetresidualhistoryf90_     KSPGETRESIDUALHISTORYF90
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define kspgetresidualhistoryf90_     kspgetresidualhistoryf90
#endif

PETSC_EXTERN void kspgetresidualhistoryf90_(KSP *ksp,F90Array1d *indices,PetscInt *n,int *ierr PETSC_F90_2PTR_PROTO(ptrd))
{
  PetscReal *hist;
  *ierr = KSPGetResidualHistory(*ksp,&hist,n); if (*ierr) return;
  *ierr = F90Array1dCreate(hist,MPIU_REAL,1,*n,indices PETSC_F90_2PTR_PARAM(ptrd));
}


#include <petsc/private/fortranimpl.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petsclogflops_            PETSCLOGFLOPS
#define petscloggpuflops_         PETSCLOGGPUFLOPS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petsclogflops_            petsclogflops
#define petscloggpuflops_         petscloggpuflops
#endif

PETSC_EXTERN void petsclogflops_(PetscLogDouble *f,PetscErrorCode *ierr)
{
  *ierr = PetscLogFlops(*f);
}

PETSC_EXTERN void petscloggpuflops_(PetscLogDouble *n, PetscErrorCode *ierr)
{
  *ierr = PetscLogGpuFlops(*n);
}

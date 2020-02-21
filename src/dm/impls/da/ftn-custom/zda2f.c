#include <petsc/private/fortranimpl.h>
#include <petsc/private/dmdaimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dmdacreate2d_                  DMDACREATE2D
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dmdacreate2d_                  dmdacreate2d
#endif

PETSC_EXTERN void dmdacreate2d_(MPI_Comm *comm,DMBoundaryType *bx,DMBoundaryType *by,DMDAStencilType
                  *stencil_type,PetscInt *M,PetscInt *N,PetscInt *m,PetscInt *n,PetscInt *w,
                  PetscInt *s,PetscInt *lx,PetscInt *ly,DM *inra,PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(lx);
  CHKFORTRANNULLINTEGER(ly);
  *ierr = DMDACreate2d(MPI_Comm_f2c(*(MPI_Fint*)&*comm),*bx,*by,*stencil_type,*M,*N,*m,*n,*w,*s,lx,ly,inra);
}


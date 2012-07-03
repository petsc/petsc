
#include <petsc-private/fortranimpl.h>
#include <petscdmda.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dmdacreate3d_                  DMDACREATE3D
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dmdacreate3d_                  dmdacreate3d
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL dmdacreate3d_(MPI_Comm *comm,DMDABoundaryType *bx,DMDABoundaryType *by,DMDABoundaryType *bz,DMDAStencilType 
                 *stencil_type,PetscInt *M,PetscInt *N,PetscInt *P,PetscInt *m,PetscInt *n,PetscInt *p,
                 PetscInt *w,PetscInt *s,PetscInt *lx,PetscInt *ly,PetscInt *lz,DM *inra,PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(lx);
  CHKFORTRANNULLINTEGER(ly);
  CHKFORTRANNULLINTEGER(lz);
  *ierr = DMDACreate3d(MPI_Comm_f2c(*(MPI_Fint *)&*comm),*bx,*by,*bz,*stencil_type,
                        *M,*N,*P,*m,*n,*p,*w,*s,lx,ly,lz,inra);
}

EXTERN_C_END

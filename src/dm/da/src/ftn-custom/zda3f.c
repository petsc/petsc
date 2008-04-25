
#include "private/fortranimpl.h"
#include "petscda.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dacreate3d_                  DACREATE3D
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dacreate3d_                  dacreate3d
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL dacreate3d_(MPI_Comm *comm,DAPeriodicType *wrap,DAStencilType 
                 *stencil_type,PetscInt *M,PetscInt *N,PetscInt *P,PetscInt *m,PetscInt *n,PetscInt *p,
                 PetscInt *w,PetscInt *s,PetscInt *lx,PetscInt *ly,PetscInt *lz,DA *inra,PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(lx);
  CHKFORTRANNULLINTEGER(ly);
  CHKFORTRANNULLINTEGER(lz);
  *ierr = DACreate3d(MPI_Comm_f2c(*(MPI_Fint *)&*comm),*wrap,*stencil_type,
                        *M,*N,*P,*m,*n,*p,*w,*s,lx,ly,lz,inra);
}

EXTERN_C_END

/*
     Fortran interface for SDA routines.
*/
#include "private/fortranimpl.h"
#include "petscda.h"

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define sdacreate1d_          SDACREATE1D
#define sdacreate2d_          SDACREATE2D
#define sdacreate3d_          SDACREATE3D
#define sdagetghostcorners_   SDAGETGHOSTCORNERS
#define sdagetcorners_        SDAGETCORNERS
#define sdaarrayview_         SDAARRAYVIEW
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define sdacreate1d_          sdacreate1d
#define sdacreate2d_          sdacreate2d
#define sdacreate3d_          sdacreate3d
#define sdagetghostcorners_   sdagetghostcorners
#define sdagetcorners_        sdagetcorners
#define sdaarrayview_         sdaarrayview
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL sdaarrayview_(SDA *da,PetscScalar *values,PetscViewer *vin,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(vin,v);
  *ierr = SDAArrayView(*da,values,v);
}

void PETSC_STDCALL sdagetghostcorners_(SDA *da,PetscInt *x,PetscInt *y,PetscInt *z,PetscInt *m,PetscInt *n,PetscInt *p,PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(x);
  CHKFORTRANNULLINTEGER(y);
  CHKFORTRANNULLINTEGER(z);
  CHKFORTRANNULLINTEGER(m);
  CHKFORTRANNULLINTEGER(n);
  CHKFORTRANNULLINTEGER(p);
  *ierr = SDAGetGhostCorners(*da,x,y,z,m,n,p);
}

void PETSC_STDCALL sdagetcorners_(SDA *da,PetscInt *x,PetscInt *y,PetscInt *z,PetscInt *m,PetscInt *n,PetscInt *p,PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(x);
  CHKFORTRANNULLINTEGER(y);
  CHKFORTRANNULLINTEGER(z);
  CHKFORTRANNULLINTEGER(m);
  CHKFORTRANNULLINTEGER(n);
  CHKFORTRANNULLINTEGER(p);
  *ierr = SDAGetCorners(*da,x,y,z,m,n,p);
}

void PETSC_STDCALL sdacreate2d_(MPI_Comm *comm,DAPeriodicType *wrap,DAStencilType
                  *stencil_type,PetscInt *M,PetscInt *N,PetscInt *m,PetscInt *n,PetscInt *w,
                  PetscInt *s,PetscInt *lx,PetscInt *ly,SDA *inra,PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(lx);
  CHKFORTRANNULLINTEGER(ly);
  *ierr = SDACreate2d(
	    MPI_Comm_f2c(*(MPI_Fint *)&*comm),*wrap,
            *stencil_type,*M,*N,*m,*n,*w,*s,lx,ly,inra);
}

void PETSC_STDCALL sdacreate1d_(MPI_Comm *comm,DAPeriodicType *wrap,PetscInt *M,PetscInt *w,PetscInt *s,
                 PetscInt *lc,SDA *inra,PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(lc);
  *ierr = SDACreate1d(
	   MPI_Comm_f2c(*(MPI_Fint *)&*comm),*wrap,*M,*w,*s,lc,inra);
}

void PETSC_STDCALL sdacreate3d_(MPI_Comm *comm,DAPeriodicType *wrap,DAStencilType 
                 *stencil_type,PetscInt *M,PetscInt *N,PetscInt *P,PetscInt *m,PetscInt *n,PetscInt *p,
                 PetscInt *w,PetscInt *s,PetscInt *lx,PetscInt *ly,PetscInt *lz,SDA *inra,PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(lx);
  CHKFORTRANNULLINTEGER(ly);
  CHKFORTRANNULLINTEGER(lz);
  *ierr = SDACreate3d(
	   MPI_Comm_f2c(*(MPI_Fint *)&*comm),*wrap,*stencil_type,
           *M,*N,*P,*m,*n,*p,*w,*s,lx,ly,lz,inra);
}

EXTERN_C_END

/*$Id: sda2f.c,v 1.15 2001/08/06 21:18:49 bsmith Exp $*/
/*
     Fortran interface for SDA routines.
*/
#include "src/fortran/custom/zpetsc.h"
#include "petscda.h"

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define sdadestroy_           SDADESTROY
#define sdalocaltolocalbegin_ SDALOCALTOLOCALBEGIN
#define sdalocaltolocalend_   SDALOCALTOLOCALEND
#define sdacreate1d_          SDACREATE1D
#define sdacreate2d_          SDACREATE2D
#define sdacreate3d_          SDACREATE3D
#define sdagetghostcorners_   SDAGETGHOSTCORNERS
#define sdagetcorners_        SDAGETCORNERS
#define sdaarrayview_         SDAARRAYVIEW
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define sdadestroy_           sdadestroy
#define sdalocaltolocalbegin_ sdalocaltolocalbegin
#define sdalocaltolocalend_   sdalocaltolocalend
#define sdacreate1d_          sdacreate1d
#define sdacreate2d_          sdacreate2d
#define sdacreate3d_          sdacreate3d
#define sdagetghostcorners_   sdagetghostcorners
#define sdagetcorners_        sdagetcorners
#define sdaarrayview_         sdaarrayview
#endif

extern int SDAArrayView(SDA,PetscScalar*,PetscViewer);

EXTERN_C_BEGIN
void sdaarrayview_(SDA *da,PetscScalar *values,PetscViewer *vin,int *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(vin,v);
  *ierr = SDAArrayView(*da,values,v);
}

void sdagetghostcorners_(SDA *da,int *x,int *y,int *z,int *m,int *n,int *p,int *ierr)
{
  CHKFORTRANNULLINTEGER(x);
  CHKFORTRANNULLINTEGER(y);
  CHKFORTRANNULLINTEGER(z);
  CHKFORTRANNULLINTEGER(m);
  CHKFORTRANNULLINTEGER(n);
  CHKFORTRANNULLINTEGER(p);
  *ierr = SDAGetGhostCorners(*da,x,y,z,m,n,p);
}

void sdagetcorners_(SDA *da,int *x,int *y,int *z,int *m,int *n,int *p,int *ierr)
{
  CHKFORTRANNULLINTEGER(x);
  CHKFORTRANNULLINTEGER(y);
  CHKFORTRANNULLINTEGER(z);
  CHKFORTRANNULLINTEGER(m);
  CHKFORTRANNULLINTEGER(n);
  CHKFORTRANNULLINTEGER(p);
  *ierr = SDAGetCorners(*da,x,y,z,m,n,p);
}

void sdadestroy_(SDA *sda,int *__ierr)
{
  *__ierr = SDADestroy((SDA)PetscToPointer(sda));
  PetscRmPointer(sda);
}

void sdalocaltolocalbegin_(SDA *sda,PetscScalar *g,InsertMode *mode,PetscScalar *l,
                           int *__ierr)
{
  *__ierr = SDALocalToLocalBegin((SDA)PetscToPointer(sda),g,*mode,l);
}

void sdalocaltolocalend_(SDA *sda,PetscScalar *g,InsertMode *mode,PetscScalar *l,
                         int *__ierr){
  *__ierr = SDALocalToLocalEnd((SDA)PetscToPointer(sda),g,*mode,l);
}

void sdacreate2d_(MPI_Comm *comm,DAPeriodicType *wrap,DAStencilType
                  *stencil_type,int *M,int *N,int *m,int *n,int *w,
                  int *s,int *lx,int *ly,SDA *inra,int *ierr)
{
  CHKFORTRANNULLINTEGER(lx);
  CHKFORTRANNULLINTEGER(ly);
  *ierr = SDACreate2d(
	    (MPI_Comm)PetscToPointerComm(*comm),*wrap,
            *stencil_type,*M,*N,*m,*n,*w,*s,lx,ly,inra);
}

void sdacreate1d_(MPI_Comm *comm,DAPeriodicType *wrap,int *M,int *w,int *s,
                 int *lc,SDA *inra,int *ierr)
{
  CHKFORTRANNULLINTEGER(lc);
  *ierr = SDACreate1d(
	   (MPI_Comm)PetscToPointerComm(*comm),*wrap,*M,*w,*s,lc,inra);
}

void sdacreate3d_(MPI_Comm *comm,DAPeriodicType *wrap,DAStencilType 
                 *stencil_type,int *M,int *N,int *P,int *m,int *n,int *p,
                 int *w,int *s,int *lx,int *ly,int *lz,SDA *inra,int *ierr)
{
  CHKFORTRANNULLINTEGER(lx);
  CHKFORTRANNULLINTEGER(ly);
  CHKFORTRANNULLINTEGER(lz);
  *ierr = SDACreate3d(
	   (MPI_Comm)PetscToPointerComm(*comm),*wrap,*stencil_type,
           *M,*N,*P,*m,*n,*p,*w,*s,lx,ly,lz,inra);
}

EXTERN_C_END

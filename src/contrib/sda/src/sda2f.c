
#ifndef lint
static char vcid[] = "$Id: sda2f.c,v 1.5 1997/02/04 21:26:55 bsmith Exp bsmith $";
#endif
/*
     Fortran interface for SDA routines.
*/
#include "src/fortran/custom/zpetsc.h"

#include "src/contrib/sda/src/sda.h"

#ifdef HAVE_FORTRAN_CAPS
#define sdadestroy_           SDADESTROY
#define sdalocaltolocalbegin_ SDALOCALTOLOCALBEGIN
#define sdalocaltolocalend_   SDALOCALTOLOCALEND
#define sdacreate1d_          SDACREATE1D
#define sdacreate2d_          SDACREATE2D
#define sdacreate3d_          SDACREATE3D
#elif !defined(HAVE_FORTRAN_UNDERSCORE)
#define sdadestroy_           sdadestroy
#define sdalocaltolocalbegin_ sdalocaltolocalbegin
#define sdalocaltolocalend_   sdalocaltolocalend
#define sdacreate1d_          sdacreate1d
#define sdacreate2d_          sdacreate2d
#define sdacreate3d_          sdacreate3d
#endif

#if defined(__cplusplus)
extern "C" {
#endif

void sdadestroy_(SDA *sda, int *__ierr )
{
  *__ierr = SDADestroy((SDA)PetscToPointer(*(int*) sda));
  PetscRmPointer(*(int*)(sda));
}

void sdalocaltolocalbegin_(SDA *sda,Scalar *g,InsertMode *mode,Scalar *l,
                           int *__ierr )
{
  *__ierr = SDALocalToLocalBegin((SDA)PetscToPointer(*(int*)sda),g,*mode,l);
}

void sdalocaltolocalend_(SDA *sda,Scalar *g,InsertMode *mode,Scalar *l, 
                         int *__ierr ){
  *__ierr = SDALocalToLocalEnd((SDA)PetscToPointer(*(int*)sda),g,*mode,l);
}

void sdacreate2d_(MPI_Comm comm,DAPeriodicType *wrap,DAStencilType
                  *stencil_type,int *M,int *N,int *m,int *n,int *w,
                  int *s,int *lx,int *ly,SDA *inra, int *__ierr )
{
  SDA da;
  *__ierr = SDACreate2d(
	    (MPI_Comm)PetscToPointerComm( *(int*)(comm) ),*wrap,
            *stencil_type,*M,*N,*m,*n,*w,*s,lx,ly,&da);
  *(int*) inra = PetscFromPointer(da);
}

void sdacreate1d_(MPI_Comm comm,DAPeriodicType *wrap,int *M,int *w,int *s,
                 int *lc,SDA *inra, int *__ierr )
{
  SDA da;
  *__ierr = SDACreate1d(
	   (MPI_Comm)PetscToPointerComm( *(int*)(comm) ),*wrap,*M,*w,*s,lc,&da);
  *(int*) inra = PetscFromPointer(da);
}

void sdacreate3d_(MPI_Comm comm,DAPeriodicType *wrap,DAStencilType 
                 *stencil_type,int *M,int *N,int *P,int *m,int *n,int *p,
                 int *w,int *s,SDA *inra, int *__ierr )
{
  SDA da;
  *__ierr = SDACreate3d(
	   (MPI_Comm)PetscToPointerComm(*(int*)(comm)),*wrap,*stencil_type,
           *M,*N,*P,*m,*n,*p,*w,*s,&da);
  *(int*) inra = PetscFromPointer(da);
}

#if defined(__cplusplus)
}
#endif

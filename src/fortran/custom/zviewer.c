/*$Id: zviewer.c,v 1.24 2000/01/11 21:03:48 bsmith Exp bsmith $*/

#include "src/fortran/custom/zpetsc.h"
#include "petsc.h"

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define viewerdestroy_        VIEWERDESTROY
#define viewerasciiopen_      VIEWERASCIIOPEN
#define viewersetformat_      VIEWERSETFORMAT
#define viewerpushformat_     VIEWERPUSHFORMAT
#define viewerpopformat_      VIEWERPOPFORMAT
#define viewerbinaryopen_     VIEWERBINARYOPEN
#define viewersocketopen_     VIEWERSOCKETOPEN
#define viewerstringopen_     VIEWERSTRINGOPEN
#define viewerdrawopen_       VIEWERDRAWOPEN
#define viewerbinarysettype_  VIEWERBINARYSETTYPE
#define viewersetfilename_    VIEWERSETFILENAME
#define viewersocketputscalar_ VIEWERSOCKETPUTSCALAR
#define viewersocketputint_    VIEWERSOCKETPUTINT
#define viewersocketputreal_   VIEWERSOCKETPUTREAL
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define viewersocketputscalar_ viewersocketputscalar
#define viewersocketputint_    viewersocketputint
#define viewersocketputreal_   viewersocketputreal
#define viewerdestroy_        viewerdestroy
#define viewerasciiopen_      viewerasciiopen
#define viewersetformat_      viewersetformat
#define viewerpushformat_     viewerpushformat
#define viewerpopformat_      viewerpopformat
#define viewerbinaryopen_     viewerbinaryopen
#define viewersocketopen_     viewersocketopen
#define viewerstringopen_     viewerstringopen
#define viewerdrawopen_       viewerdrawopen
#define viewerbinarysettype_  viewerbinarysettype
#define viewersetfilename_    viewersetfilename
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL viewersocketputscalar(Viewer *viewer,int *m,int *n,Scalar *s,int *ierr)
{
  *ierr = ViewerSocketPutScalar(*viewer,*m,*n,s);
}

void PETSC_STDCALL viewersocketputreal(Viewer *viewer,int *m,int *n,PetscReal *s,int *ierr)
{
  *ierr = ViewerSocketPutReal(*viewer,*m,*n,s);
}

void PETSC_STDCALL viewersocketputint(Viewer *viewer,int *m,int *s,int *ierr)
{
  *ierr = ViewerSocketPutInt(*viewer,*m,s);
}

void PETSC_STDCALL viewersetfilename_(Viewer *viewer,CHAR name,int *ierr,int len1)
{
  char   *c1;
  FIXCHAR(name,len1,c1);
  *ierr = ViewerSetFilename(*viewer,c1);
  FREECHAR(name,c1);
}

void PETSC_STDCALL  viewerbinarysettype_(Viewer *viewer,ViewerBinaryType *type,int *ierr)
{
  *ierr = ViewerBinarySetType(*viewer,*type);
}

void PETSC_STDCALL viewersocketopen_(MPI_Comm *comm,CHAR name,int *port,Viewer *lab,
                       int *ierr,int len1)
{
  char   *c1;
  FIXCHAR(name,len1,c1);
  *ierr = ViewerSocketOpen((MPI_Comm)PetscToPointerComm(*comm),
     c1,*port,lab);
  FREECHAR(name,c1);
}

void PETSC_STDCALL viewerbinaryopen_(MPI_Comm *comm,CHAR name,ViewerBinaryType *type,
                           Viewer *binv,int *ierr,int len1)
{
  char   *c1;
  FIXCHAR(name,len1,c1);
  *ierr = ViewerBinaryOpen((MPI_Comm)PetscToPointerComm(*comm),c1,*type,binv);
  FREECHAR(name,c1);
}

void PETSC_STDCALL viewerasciiopen_(MPI_Comm *comm,CHAR name,Viewer *lab,int *ierr,int len1)
{
  char   *c1;
  FIXCHAR(name,len1,c1);
  *ierr = ViewerASCIIOpen((MPI_Comm)PetscToPointerComm(*comm),c1,lab);
  FREECHAR(name,c1);
}

void PETSC_STDCALL viewersetformat_(Viewer *vin,int *format,CHAR name,int *ierr,int len1)
{
  Viewer v;
  char   *c1;
  PetscPatchDefaultViewers_Fortran(vin,v);
  FIXCHAR(name,len1,c1);
  *ierr = ViewerSetFormat(v,*format,c1);
}

void PETSC_STDCALL viewerpushformat_(Viewer *vin,int *format,CHAR name,int *ierr,int len1)
{
  Viewer v;
  char   *c1;
  PetscPatchDefaultViewers_Fortran(vin,v);
  FIXCHAR(name,len1,c1);
  *ierr = ViewerPushFormat(v,*format,c1);
}

void PETSC_STDCALL viewerpopformat_(Viewer *vin,int *ierr)
{
  Viewer v;
  PetscPatchDefaultViewers_Fortran(vin,v);
  *ierr = ViewerPopFormat(v);
}

void PETSC_STDCALL viewerdestroy_(Viewer *v,int *ierr)
{
  *ierr = ViewerDestroy(*v);
}

void PETSC_STDCALL viewerstringopen_(MPI_Comm *comm,CHAR name,int *len,Viewer *str,int *ierr,int len1)
{
#if defined(PETSC_USES_CPTOFCD)
  *ierr = ViewerStringOpen((MPI_Comm)PetscToPointerComm(*comm),_fcdtocp(name),*len,str);
#else
  *ierr = ViewerStringOpen((MPI_Comm)PetscToPointerComm(*comm),name,*len,str);
#endif
}
  
void PETSC_STDCALL viewerdrawopen_(MPI_Comm *comm,CHAR display,CHAR title,int *x,int*y,int*w,int*h,Viewer *v,
                      int *ierr,int len1,int len2)
{
  char   *c1,*c2;

  FIXCHAR(display,len1,c1);
  FIXCHAR(title,len2,c2);
  *ierr = ViewerDrawOpen((MPI_Comm)PetscToPointerComm(*comm),c1,c2,*x,*y,*w,*h,v);
  FREECHAR(display,c1);
  FREECHAR(title,c2);
}

EXTERN_C_END



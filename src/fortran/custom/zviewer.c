/*$Id: zviewer.c,v 1.30 2001/01/19 23:25:08 balay Exp bsmith $*/

#include "src/fortran/custom/zpetsc.h"
#include "petsc.h"

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define viewerdestroy_        PETSC_VIEWERDESTROY
#define viewerasciiopen_      PETSC_VIEWERASCIIOPEN
#define viewersetformat_      PETSC_VIEWERSETFORMAT
#define viewerpushformat_     PETSC_VIEWERPUSHFORMAT
#define viewerpopformat_      PETSC_VIEWERPOPFORMAT
#define viewerbinaryopen_     PETSC_VIEWERBINARYOPEN
#define viewersocketopen_     PETSC_VIEWERSOCKETOPEN
#define viewerstringopen_     PETSC_VIEWERSTRINGOPEN
#define viewerdrawopen_       PETSC_VIEWERDRAWOPEN
#define viewerbinarysettype_  PETSC_VIEWERBINARYSETTYPE
#define viewersetfilename_    PETSC_VIEWERSETFILENAME
#define viewersocketputscalar_ PETSC_VIEWERSOCKETPUTSCALAR
#define viewersocketputint_    PETSC_VIEWERSOCKETPUTINT
#define viewersocketputreal_   PETSC_VIEWERSOCKETPUTREAL
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

void PETSC_STDCALL viewersocketputscalar(PetscViewer *viewer,int *m,int *n,Scalar *s,int *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = PetscViewerSocketPutScalar(v,*m,*n,s);
}

void PETSC_STDCALL viewersocketputreal(PetscViewer *viewer,int *m,int *n,PetscReal *s,int *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = PetscViewerSocketPutReal(v,*m,*n,s);
}

void PETSC_STDCALL viewersocketputint(PetscViewer *viewer,int *m,int *s,int *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = PetscViewerSocketPutInt(v,*m,s);
}

void PETSC_STDCALL viewersetfilename_(PetscViewer *viewer,CHAR name PETSC_MIXED_LEN(len),
                                      int *ierr PETSC_END_LEN(len))
{
  char   *c1;
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  FIXCHAR(name,len,c1);
  *ierr = PetscViewerSetFilename(v,c1);
  FREECHAR(name,c1);
}

void PETSC_STDCALL  viewerbinarysettype_(PetscViewer *viewer,PetscViewerBinaryType *type,int *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = PetscViewerBinarySetType(v,*type);
}

void PETSC_STDCALL viewersocketopen_(MPI_Comm *comm,CHAR name PETSC_MIXED_LEN(len),int *port,PetscViewer *lab,int *ierr PETSC_END_LEN(len))
{
  char   *c1;
  FIXCHAR(name,len,c1);
  *ierr = PetscViewerSocketOpen((MPI_Comm)PetscToPointerComm(*comm),c1,*port,lab);
  FREECHAR(name,c1);
}

void PETSC_STDCALL viewerbinaryopen_(MPI_Comm *comm,CHAR name PETSC_MIXED_LEN(len),PetscViewerBinaryType *type,
                           PetscViewer *binv,int *ierr PETSC_END_LEN(len))
{
  char   *c1;
  FIXCHAR(name,len,c1);
  *ierr = PetscViewerBinaryOpen((MPI_Comm)PetscToPointerComm(*comm),c1,*type,binv);
  FREECHAR(name,c1);
}

void PETSC_STDCALL viewerasciiopen_(MPI_Comm *comm,CHAR name PETSC_MIXED_LEN(len),PetscViewer *lab,
                                    int *ierr PETSC_END_LEN(len))
{
  char   *c1;
  FIXCHAR(name,len,c1);
  *ierr = PetscViewerASCIIOpen((MPI_Comm)PetscToPointerComm(*comm),c1,lab);
  FREECHAR(name,c1);
}

void PETSC_STDCALL viewersetformat_(PetscViewer *vin,PetscViewerFormat *format,int *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(vin,v);
  *ierr = PetscViewerSetFormat(v,*format);
}

void PETSC_STDCALL viewerpushformat_(PetscViewer *vin,PetscViewerFormat *format,int *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(vin,v);
  *ierr = PetscViewerPushFormat(v,*format);
}

void PETSC_STDCALL viewerpopformat_(PetscViewer *vin,int *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(vin,v);
  *ierr = PetscViewerPopFormat(v);
}

void PETSC_STDCALL viewerdestroy_(PetscViewer *v,int *ierr)
{
  *ierr = PetscViewerDestroy(*v);
}

void PETSC_STDCALL viewerstringopen_(MPI_Comm *comm,CHAR name PETSC_MIXED_LEN(len1),int *len,PetscViewer *str,
                                     int *ierr PETSC_END_LEN(len1))
{
#if defined(PETSC_USES_CPTOFCD)
  *ierr = PetscViewerStringOpen((MPI_Comm)PetscToPointerComm(*comm),_fcdtocp(name),*len,str);
#else
  *ierr = PetscViewerStringOpen((MPI_Comm)PetscToPointerComm(*comm),name,*len,str);
#endif
}
  
void PETSC_STDCALL viewerdrawopen_(MPI_Comm *comm,CHAR display PETSC_MIXED_LEN(len1),
                   CHAR title PETSC_MIXED_LEN(len2),int *x,int*y,int*w,int*h,PetscViewer *v,
                   int *ierr PETSC_END_LEN(len1) PETSC_END_LEN(len2))
{
  char   *c1,*c2;

  FIXCHAR(display,len1,c1);
  FIXCHAR(title,len2,c2);
  *ierr = PetscViewerDrawOpen((MPI_Comm)PetscToPointerComm(*comm),c1,c2,*x,*y,*w,*h,v);
  FREECHAR(display,c1);
  FREECHAR(title,c2);
}

EXTERN_C_END



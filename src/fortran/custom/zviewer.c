#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: zviewer.c,v 1.12 1998/10/05 20:43:29 balay Exp bsmith $";
#endif

#include "src/fortran/custom/zpetsc.h"
#include "petsc.h"

#ifdef HAVE_FORTRAN_CAPS
#define viewerdestroy_        VIEWERDESTROY
#define viewerfileopenascii_  VIEWERFILEOPENASCII
#define viewersetformat_      VIEWERSETFORMAT
#define viewerpushformat_     VIEWERPUSHFORMAT
#define viewerpopformat_      VIEWERPOPFORMAT
#define viewerfileopenbinary_ VIEWERFILEOPENBINARY
#define viewermatlabopen_     VIEWERMATLABOPEN
#define viewerstringopen_     VIEWERSTRINGOPEN
#define viewerdrawopenx_      VIEWERDRAWOPENX
#elif !defined(HAVE_FORTRAN_UNDERSCORE)
#define viewerdestroy_        viewerdestroy
#define viewerfileopenascii_  viewerfileopenascii
#define viewersetformat_      viewersetformat
#define viewerpushformat_     viewerpushformat
#define viewerpopformat_      viewerpopformat
#define viewerfileopenbinary_ viewerfileopenbinary
#define viewermatlabopen_     viewermatlabopen
#define viewerstringopen_     viewerstringopen
#define viewerdrawopenx_      viewerdrawopenx
#endif

#if defined(__cplusplus)
extern "C" {
#endif

void viewermatlabopen_(MPI_Comm *comm,CHAR name,int *port,Viewer *lab, 
                       int *__ierr,int len1 )
{
  char   *c1;
  FIXCHAR(name,len1,c1);
  *__ierr = ViewerMatlabOpen((MPI_Comm)PetscToPointerComm(*comm),
     c1,*port,lab);
  FREECHAR(name,c1);
}

void viewerfileopenbinary_(MPI_Comm *comm,CHAR name,ViewerBinaryType *type,
                           Viewer *binv, int *__ierr,int len1 )
{
  char   *c1;
  FIXCHAR(name,len1,c1);
  *__ierr = ViewerFileOpenBinary(
                 (MPI_Comm)PetscToPointerComm(*comm),c1,*type,binv);
  FREECHAR(name,c1);
}

void viewerfileopenascii_(MPI_Comm *comm,CHAR name,Viewer *lab, int *__ierr,
                          int len1 )
{
  char   *c1;
  FIXCHAR(name,len1,c1);
  *__ierr = ViewerFileOpenASCII((MPI_Comm)PetscToPointerComm(*comm),c1,lab);
  FREECHAR(name,c1);
}

void viewersetformat_(Viewer *vin,int *format,CHAR name,int *__ierr,int len1)
{
  Viewer v;
  char   *c1;
  PetscPatchDefaultViewers_Fortran(vin,v);
  FIXCHAR(name,len1,c1);
  *__ierr = ViewerSetFormat(v,*format,c1);
}

void viewerpushformat_(Viewer *vin,int *format,CHAR name,int *__ierr,int len1)
{
  Viewer v;
  char   *c1;
  PetscPatchDefaultViewers_Fortran(vin,v);
  FIXCHAR(name,len1,c1);
  *__ierr = ViewerPushFormat(v,*format,c1);
}

void viewerpopformat_(Viewer *vin,int *__ierr)
{
  Viewer v;
  PetscPatchDefaultViewers_Fortran(vin,v);
  *__ierr = ViewerPopFormat(v);
}

void viewerdestroy_(Viewer *v, int *__ierr )
{
  *__ierr = ViewerDestroy(*v);
}

void viewerstringopen_(MPI_Comm *comm,CHAR name,int *len, Viewer *str,int *__ierr,int len1)
{
#if defined(USES_CPTOFCD)
  *__ierr = ViewerStringOpen((MPI_Comm)PetscToPointerComm(*comm),_fcdtocp(name),*len,str);
#else
  *__ierr = ViewerStringOpen((MPI_Comm)PetscToPointerComm(*comm),name,*len,str);
#endif
}
  
void viewerdrawopenx_(MPI_Comm *comm,CHAR display,CHAR title, int *x,int*y,int*w,int*h,Viewer *v,
                      int *__ierr,int len1,int len2)
{
  char   *c1,*c2;

  FIXCHAR(display,len1,c1);
  FIXCHAR(title,len2,c2);
  *__ierr = ViewerDrawOpenX((MPI_Comm)PetscToPointerComm(*comm),c1,c2,*x,*y,*w,*h,v);
  FREECHAR(display,c1);
  FREECHAR(title,c2);
}

#if defined(__cplusplus)
}
#endif



#ifndef lint
static char vcid[] = "$Id: zviewer.c,v 1.6 1996/03/23 16:56:55 bsmith Exp bsmith $";
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

void viewermatlabopen_(MPI_Comm comm,CHAR name,int *port,Viewer *lab, 
                       int *__ierr,int len1 )
{
  Viewer vv;
  char   *c1;
  FIXCHAR(name,len1,c1);
  *__ierr = ViewerMatlabOpen((MPI_Comm)PetscToPointerComm(*(int*)(comm)),
     c1,*port,&vv);
  *(int*) lab = PetscFromPointer(vv);
  FREECHAR(name,c1);
}

void viewerfileopenbinary_(MPI_Comm comm,CHAR name,ViewerBinaryType *type,
                           Viewer *binv, int *__ierr,int len1 )
{
  Viewer vv;
  char   *c1;
  FIXCHAR(name,len1,c1);
  *__ierr = ViewerFileOpenBinary(
                 (MPI_Comm)PetscToPointerComm(*(int*)(comm)),c1,*type,&vv);
  *(int*) binv = PetscFromPointer(vv);
  FREECHAR(name,c1);
}

void viewerfileopenascii_(MPI_Comm comm,CHAR name,Viewer *lab, int *__ierr,
                          int len1 )
{
  Viewer vv;
  char   *c1;
  FIXCHAR(name,len1,c1);
  *__ierr = ViewerFileOpenASCII((MPI_Comm)PetscToPointerComm(*(int*)(comm)),
     c1,&vv);
  *(int*) lab = PetscFromPointer(vv);
  FREECHAR(name,c1);
}

void viewersetformat_(Viewer v,int *format,CHAR name,int *__ierr,int len1)
{
  char   *c1;
  PetscPatchDefaultViewers_Fortran(v);
  FIXCHAR(name,len1,c1);
  *__ierr = ViewerSetFormat(v,*format,c1);
}

void viewerpushformat_(Viewer v,int *format,CHAR name,int *__ierr,int len1)
{
  char   *c1;
  PetscPatchDefaultViewers_Fortran(v);
  FIXCHAR(name,len1,c1);
  *__ierr = ViewerPushFormat(v,*format,c1);
}

void viewerpopformat_(Viewer v,int *__ierr)
{
  PetscPatchDefaultViewers_Fortran(v);
  *__ierr = ViewerPopFormat(v);
}

void viewerdestroy_(Viewer v, int *__ierr )
{
  *__ierr = ViewerDestroy((Viewer)PetscToPointer( *(int*)(v) ));
  PetscRmPointer(*(int*)(v) );
}

void viewerstringopen_(MPI_Comm comm,CHAR name,int *len, Viewer *str,int *__ierr,int len1)
{
  Viewer vv;
#if defined(USES_CPTOFCD)
  *__ierr = ViewerStringOpen((MPI_Comm)PetscToPointerComm(*(int*)(comm)),_fcdtocp(name),*len,&vv);
#else
  *__ierr = ViewerStringOpen((MPI_Comm)PetscToPointerComm(*(int*)(comm)),name,*len,&vv);
#endif
  *(int*) str = PetscFromPointer(vv);
}
  
void viewerdrawopenx_(MPI_Comm comm,CHAR display,CHAR title, int *x,int*y,int*w,int*h,Viewer *v,
                      int *__ierr,int len1,int len2)
{
  char   *c1,*c2;
  Viewer vv;

  FIXCHAR(display,len1,c1);
  FIXCHAR(title,len2,c2);
  *__ierr = ViewerDrawOpenX((MPI_Comm)PetscToPointerComm(*(int*)(comm)),c1,c2,*x,*y,*w,*h,&vv);
  FREECHAR(display,c1);
  FREECHAR(title,c2);
  *(int*) v = PetscFromPointer(vv);
}

#if defined(__cplusplus)
}
#endif


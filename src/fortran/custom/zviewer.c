
#ifndef lint
static char vcid[] = "$Id: zviewer.c,v 1.5 1996/03/04 21:30:31 bsmith Exp bsmith $";
#endif

#include "zpetsc.h"
#include "petsc.h"

#ifdef HAVE_FORTRAN_CAPS
#define viewerdestroy_        VIEWERDESTROY
#define viewerfileopenascii_  VIEWERFILEOPENASCII
#define viewersetformat_      VIEWERSETFORMAT
#define viewerfileopenbinary_ VIEWERFILEOPENBINARY
#define viewermatlabopen_     VIEWERMATLABOPEN
#elif !defined(HAVE_FORTRAN_UNDERSCORE)
#define viewerdestroy_        viewerdestroy
#define viewerfileopenascii_  viewerfileopenascii
#define viewersetformat_      viewersetformat
#define viewerfileopenbinary_ viewerfileopenbinary
#define viewermatlabopen_     viewermatlabopen
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
  *__ierr = ViewerMatlabOpen((MPI_Comm)MPIR_ToPointer_Comm(*(int*)(comm)),
     c1,*port,&vv);
  *(int*) lab = MPIR_FromPointer(vv);
  FREECHAR(name,c1);
}

void viewerfileopenbinary_(MPI_Comm comm,CHAR name,ViewerBinaryType *type,
                           Viewer *binv, int *__ierr,int len1 )
{
  Viewer vv;
  char   *c1;
  FIXCHAR(name,len1,c1);
  *__ierr = ViewerFileOpenBinary(
                 (MPI_Comm)MPIR_ToPointer_Comm(*(int*)(comm)),c1,*type,&vv);
  *(int*) binv = MPIR_FromPointer(vv);
  FREECHAR(name,c1);
}

void viewerfileopenascii_(MPI_Comm comm,CHAR name,Viewer *lab, int *__ierr,
                          int len1 )
{
  Viewer vv;
  char   *c1;
  FIXCHAR(name,len1,c1);
  *__ierr = ViewerFileOpenASCII((MPI_Comm)MPIR_ToPointer_Comm(*(int*)(comm)),
     c1,&vv);
  *(int*) lab = MPIR_FromPointer(vv);
  FREECHAR(name,c1);
}

void viewersetformat_(Viewer v,int *format,CHAR name,int *__ierr,int len1)
{
  char   *c1;
  FIXCHAR(name,len1,c1);
  *__ierr = ViewerSetFormat((Viewer)MPIR_ToPointer(*(int*)(v)),*format,c1);
  FREECHAR(name,c1);
}

void viewerdestroy_(Viewer v, int *__ierr )
{
  *__ierr = ViewerDestroy((Viewer)MPIR_ToPointer( *(int*)(v) ));
  MPIR_RmPointer(*(int*)(v) );
}

#if defined(__cplusplus)
}
#endif

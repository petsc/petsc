#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: zviewer.c,v 1.18 1999/03/23 21:54:16 balay Exp bsmith $";
#endif

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
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
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

#undef __FUNC__ 
#define __FUNC__ ""
void viewersetfilename_(Viewer *viewer, CHAR name, int *__ierr,int len1)
{
  char   *c1;
  FIXCHAR(name,len1,c1);
  *__ierr = ViewerSetFilename(*viewer,c1);
  FREECHAR(name,c1);
}

#undef __FUNC__ 
#define __FUNC__ ""
void  viewerbinarysettype_(Viewer *viewer,ViewerBinaryType *type, int *__ierr)
{
  *__ierr = ViewerBinarySetType(*viewer,*type);
}

void viewersocketopen_(MPI_Comm *comm,CHAR name,int *port,Viewer *lab, 
                       int *__ierr,int len1 )
{
  char   *c1;
  FIXCHAR(name,len1,c1);
  *__ierr = ViewerSocketOpen((MPI_Comm)PetscToPointerComm(*comm),
     c1,*port,lab);
  FREECHAR(name,c1);
}

void viewerbinaryopen_(MPI_Comm *comm,CHAR name,ViewerBinaryType *type,
                           Viewer *binv, int *__ierr,int len1 )
{
  char   *c1;
  FIXCHAR(name,len1,c1);
  *__ierr = ViewerBinaryOpen((MPI_Comm)PetscToPointerComm(*comm),c1,*type,binv);
  FREECHAR(name,c1);
}

void viewerasciiopen_(MPI_Comm *comm,CHAR name,Viewer *lab, int *__ierr,int len1)
{
  char   *c1;
  FIXCHAR(name,len1,c1);
  *__ierr = ViewerASCIIOpen((MPI_Comm)PetscToPointerComm(*comm),c1,lab);
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
  
void viewerdrawopen_(MPI_Comm *comm,CHAR display,CHAR title, int *x,int*y,int*w,int*h,Viewer *v,
                      int *__ierr,int len1,int len2)
{
  char   *c1,*c2;

  FIXCHAR(display,len1,c1);
  FIXCHAR(title,len2,c2);
  *__ierr = ViewerDrawOpen((MPI_Comm)PetscToPointerComm(*comm),c1,c2,*x,*y,*w,*h,v);
  FREECHAR(display,c1);
  FREECHAR(title,c2);
}

EXTERN_C_END



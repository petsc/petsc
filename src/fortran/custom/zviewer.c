
#ifndef lint
static char vcid[] = "$Id: zviewer.c,v 1.2 1995/10/26 22:01:47 bsmith Exp bsmith $";
#endif

#include "zpetsc.h"
#include "petsc.h"

#ifdef FORTRANCAPS
#define viewerdestroy_        VIEWERDESTROY
#define viewerfileopenascii_  VIEWERFILEOPENASCII
#define viewerfilesetformat_  VIEWERFILESETFORMAT
#define viewerfileopenbinary_ VIEWERFILEOPENBINARY
#define viewermatlabopen_     VIEWERMATLABOPEN
#elif !defined(FORTRANUNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define viewerdestroy_        viewerdestroy
#define viewerfileopenascii_  viewerfileopenascii
#define viewerfilesetformat_  viewerfilesetformat
#define viewerfileopenbinary_ viewerfileopenbinary
#define viewermatlabopen_     viewermatlabopen
#endif

void viewermatlabopen_(MPI_Comm comm,char *name,int *port,Viewer *lab, int *__ierr,
                       int len1 )
{
  Viewer vv;
  char   *c1;
  if (!name[len1] == 0) {
    c1 = (char *) PetscMalloc( (len1+1)*sizeof(char)); 
    PetscStrncpy(c1,name,len1);
    c1[len1] = 0;
  } else c1 = name;
  *__ierr = ViewerMatlabOpen((MPI_Comm)MPIR_ToPointer(*(int*)(comm)),c1,*port,&vv);
  *(int*) lab = MPIR_FromPointer(vv);
  if (c1 != name) PetscFree(c1);
}

void viewerfileopenbinary_(MPI_Comm comm,char *name,ViewerBinaryType *type,
                           Viewer *binv, int *__ierr,int len1 )
{
  Viewer vv;
  char   *c1;
  if (!name[len1] == 0) {
    c1 = (char *) PetscMalloc( (len1+1)*sizeof(char)); 
    PetscStrncpy(c1,name,len1);
    c1[len1] = 0;
  } else c1 = name;
  *__ierr = ViewerFileOpenBinary(
                     (MPI_Comm)MPIR_ToPointer(*(int*)(comm)),c1,*type,&vv);
  *(int*) binv = MPIR_FromPointer(vv);
  if (c1 != name) PetscFree(c1);
}

void viewerfileopenascii_(MPI_Comm comm,char *name,Viewer *lab, int *__ierr,int len1 )
{
  Viewer vv;
  char   *c1;
  if (!name[len1] == 0) {
    c1 = (char *) PetscMalloc( (len1+1)*sizeof(char)); 
    PetscStrncpy(c1,name,len1);
    c1[len1] = 0;
  } else c1 = name;
  *__ierr = ViewerFileOpenASCII((MPI_Comm)MPIR_ToPointer(*(int*)(comm)),c1,&vv);
  *(int*) lab = MPIR_FromPointer(vv);
  if (c1 != name) PetscFree(c1);
}
void viewerfilesetformat_(Viewer v,int *format,char *name, int *__ierr,int len1 )
{
  char   *c1;
  if (!name[len1] == 0) {
    c1 = (char *) PetscMalloc( (len1+1)*sizeof(char)); 
    PetscStrncpy(c1,name,len1);
    c1[len1] = 0;
  } else c1 = name;
  *__ierr = ViewerFileSetFormat((Viewer)MPIR_ToPointer( *(int*)(v) ),*format,c1);
  if (c1 != name) PetscFree(c1);
}

void viewerdestroy_(Viewer v, int *__ierr )
{
  *__ierr = ViewerDestroy((Viewer)MPIR_ToPointer( *(int*)(v) ));
  MPIR_RmPointer(*(int*)(v) );
}

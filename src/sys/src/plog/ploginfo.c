/*$Id: ploginfo.c,v 1.13 1999/10/01 21:20:44 bsmith Exp bsmith $*/
/*
      PLogInfo() is contained in a different file from the other profiling to 
   allow it to be replaced at link time by an alternative routine.
*/
#include "petsc.h"        /*I    "petsc.h"   I*/
#include <stdarg.h>
#include <sys/types.h>
#include "sys.h"
#if defined(PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#if defined(PETSC_HAVE_MALLOC_H) && !defined(__cplusplus)
#include <malloc.h>
#endif
#include "pinclude/petscfix.h"

extern int  PLogPrintInfo,PLogPrintInfoNull;
extern int  PLogInfoFlags[];
extern FILE *PLogInfoFile;

/*
   If the option -log_history was used, then all printed PLogInfo() 
  messages are also printed to the history file, called by default
  .petschistory in ones home directory.
*/
extern FILE *petsc_history;

#undef __FUNC__  
#define __FUNC__ "PLogInfo"
/*@C
    PLogInfo - Logs informative data, which is printed to standard output
    or a file when the option -log_info <file> is specified.

    Collective over PetscObject argument

    Input Parameter:
+   vobj - object most closely associated with the logging statement
-   message - logging message, using standard "printf" format

    Options Database Key:
$    -log_info : activates printing of PLogInfo() messages 

    Level: intermediate

    Fortran Note:
    This routine is not supported in Fortran.

    Example of Usage:
$
$     Mat A
$     double alpha
$     PLogInfo(A,"Matrix uses parameter alpha=%g\n",alpha);
$

.keywords: information, printing, monitoring

.seealso: PLogInfoAllow()
@*/
int PLogInfo(void *vobj,const char message[],...)  
{
  va_list     Argp;
  int         rank,urank,len,ierr;
  PetscObject obj = (PetscObject) vobj;
  char        string[256];

  PetscFunctionBegin;
  if (obj) PetscValidHeader(obj);
  if (!PLogPrintInfo) PetscFunctionReturn(0);
  if (!PLogPrintInfoNull && !vobj) PetscFunctionReturn(0);
  if (obj && !PLogInfoFlags[obj->cookie - PETSC_COOKIE - 1]) PetscFunctionReturn(0);
  if (!obj) rank = 0;
  else      {ierr = MPI_Comm_rank(obj->comm,&rank);CHKERRQ(ierr);} 
  if (rank) PetscFunctionReturn(0);

  ierr = MPI_Comm_rank(MPI_COMM_WORLD,&urank);CHKERRQ(ierr);
  va_start( Argp, message );
  sprintf(string,"[%d]",urank); 
  ierr = PetscStrlen(string,&len);CHKERRQ(ierr);
#if defined(PETSC_HAVE_VPRINTF_CHAR)
  vsprintf(string+len,message,(char *)Argp);
#else
  vsprintf(string+len,message,Argp);
#endif
  fprintf(PLogInfoFile,"%s",string);
  fflush(PLogInfoFile);
  if (petsc_history) {
#if defined(PETSC_HAVE_VPRINTF_CHAR)
    vfprintf(petsc_history,message,(char *)Argp);
#else
    vfprintf(petsc_history,message,Argp);
#endif
  }
  va_end( Argp );
  PetscFunctionReturn(0);
}


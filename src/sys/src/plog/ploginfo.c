#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ploginfo.c,v 1.4 1998/04/13 18:01:08 bsmith Exp bsmith $";
#endif
/*
      PLogInfo() is contained in a different file from the other profiling to 
   allow it to be replaced at link time by an alternative routine.
*/
#include "petsc.h"        /*I    "petsc.h"   I*/
#include <stdarg.h>
#include <sys/types.h>
#include "sys.h"
#if defined(HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#if defined(HAVE_MALLOC_H) && !defined(__cplusplus)
#include <malloc.h>
#endif
#include "pinclude/pviewer.h"
#include "pinclude/petscfix.h"

extern int PLogPrintInfo,PLogPrintInfoNull;
extern int PLogInfoFlags[];

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
    when the option -log_info is specified.

    Collective over PetscObject argument

    Input Parameter:
+   vobj - object most closely associated with the logging statement
-   message - logging message, using standard "printf" format

    Options Database Key:
$    -log_info : activates printing of PLogInfo() messages 

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
int PLogInfo(void *vobj,char *message,...)
{
  va_list     Argp;
  int         rank,urank,len;
  PetscObject obj = (PetscObject) vobj;
  char        string[256];

  PetscFunctionBegin;
  if (obj) PetscValidHeader(obj);
  if (!PLogPrintInfo) PetscFunctionReturn(0);
  if (!PLogPrintInfoNull && !vobj) PetscFunctionReturn(0);
  if (obj && !PLogInfoFlags[obj->cookie - PETSC_COOKIE - 1]) PetscFunctionReturn(0);
  if (!obj) rank = 0;
  else      {MPI_Comm_rank(obj->comm,&rank);} 
  if (rank) PetscFunctionReturn(0);

  MPI_Comm_rank(MPI_COMM_WORLD,&urank);
  va_start( Argp, message );
  sprintf(string,"[%d]",urank); len = PetscStrlen(string);
#if (__GNUC__ == 2 && __GNUC_MINOR__ >= 7 && defined(PARCH_freebsd) )
  vsprintf(string+len,message,(char *)Argp);
#else
  vsprintf(string+len,message,Argp);
#endif
  fprintf(stdout,"%s",string);
  fflush(stdout);
  if (petsc_history) {
#if (__GNUC__ == 2 && __GNUC_MINOR__ >= 7 && defined(PARCH_freebsd) )
    vfprintf(petsc_history,message,(char *)Argp);
#else
    vfprintf(petsc_history,message,Argp);
#endif
  }
  va_end( Argp );
  PetscFunctionReturn(0);
}


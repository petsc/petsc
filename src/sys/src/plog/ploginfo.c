/*$Id: ploginfo.c,v 1.22 2001/03/23 23:20:50 balay Exp $*/
/*
      PetscLogInfo() is contained in a different file from the other profiling to 
   allow it to be replaced at link time by an alternative routine.
*/
#include "petsc.h"        /*I    "petsc.h"   I*/
#include <stdarg.h>
#include <sys/types.h>
#include "petscsys.h"
#if defined(HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#if defined(HAVE_MALLOC_H) && !defined(__cplusplus)
#include <malloc.h>
#endif
#include "petscfix.h"

extern PetscTruth PetscLogPrintInfo,PetscLogPrintInfoNull;
extern int        PetscLogInfoFlags[];
extern FILE       *PetscLogInfoFile;

/*
   If the option -log_history was used, then all printed PetscLogInfo() 
  messages are also printed to the history file, called by default
  .petschistory in ones home directory.
*/
extern FILE *petsc_history;

#undef __FUNCT__  
#define __FUNCT__ "PetscLogInfo"
/*@C
    PetscLogInfo - Logs informative data, which is printed to standard output
    or a file when the option -log_info <file> is specified.

    Collective over PetscObject argument

    Input Parameter:
+   vobj - object most closely associated with the logging statement
-   message - logging message, using standard "printf" format

    Options Database Key:
$    -log_info : activates printing of PetscLogInfo() messages 

    Level: intermediate

    Fortran Note:
    This routine is not supported in Fortran.

    Example of Usage:
$
$     Mat A
$     double alpha
$     PetscLogInfo(A,"Matrix uses parameter alpha=%g\n",alpha);
$

   Concepts: runtime information

.seealso: PetscLogInfoAllow()
@*/
int PetscLogInfo(void *vobj,const char message[],...)  
{
  va_list     Argp;
  int         rank,urank,len,ierr;
  PetscObject obj = (PetscObject)vobj;
  char        string[256];

  PetscFunctionBegin;
  if (obj) PetscValidHeader(obj);
  if (!PetscLogPrintInfo) PetscFunctionReturn(0);
  if (!PetscLogPrintInfoNull && !vobj) PetscFunctionReturn(0);
  if (obj && !PetscLogInfoFlags[obj->cookie - PETSC_COOKIE - 1]) PetscFunctionReturn(0);
  if (!obj) rank = 0;
  else      {ierr = MPI_Comm_rank(obj->comm,&rank);CHKERRQ(ierr);} 
  if (rank) PetscFunctionReturn(0);

  ierr = MPI_Comm_rank(MPI_COMM_WORLD,&urank);CHKERRQ(ierr);
  va_start(Argp,message);
  sprintf(string,"[%d]",urank); 
  ierr = PetscStrlen(string,&len);CHKERRQ(ierr);
#if defined(HAVE_VPRINTF_CHAR)
  vsprintf(string+len,message,(char *)Argp);
#else
  vsprintf(string+len,message,Argp);
#endif
  fprintf(PetscLogInfoFile,"%s",string);
  fflush(PetscLogInfoFile);
  if (petsc_history) {
#if defined(HAVE_VPRINTF_CHAR)
    vfprintf(petsc_history,message,(char *)Argp);
#else
    vfprintf(petsc_history,message,Argp);
#endif
  }
  va_end(Argp);
  PetscFunctionReturn(0);
}


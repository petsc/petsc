#ifndef lint
static char vcid[] = "$Id: err.c,v 1.41 1996/04/01 02:55:25 curfman Exp curfman $";
#endif
/*
       The default error handlers and code that allows one to change
   error handlers.
*/
#include "petsc.h"           /*I "petsc.h" I*/
#include <stdio.h>           /*I <stdio.h> I*/
#if defined(HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#include "pinclude/petscfix.h"

struct EH {
  int    cookie;
  int    (*handler)(int, char*,char *,int,char*,void *);
  void   *ctx;
  struct EH* previous;
};

static struct EH* eh = 0;

/*@C
   PetscAbortErrorHandler - Error handler that calls abort on error. 
   This routine is very useful when running in the debugger, because the 
   user can look directly at the stack frames and the variables.

   Input Parameters:
.  line - the line number of the error (indicated by __LINE__)
.  file - the file in which the error was detected (indicated by __FILE__)
.  dir - the directory of the file (indicated by __DIR__)
.  message - an error text string, usually just printed to the screen
.  number - the user-provided error number
.  ctx - error handler context

   Options Database Keys:
$   -on_error_abort
$   -start_in_debugger [noxterm,dbx,xxgdb]  [-display name]
$       Starts all processes in the debugger and uses 
$       PetscAbortErrorHandler().  By default the 
$       debugger is gdb; alternatives are dbx and xxgdb.

   Notes:
   Most users need not directly employ this routine and the other error 
   handlers, but can instead use the simplified interface SETERRQ, which
   has the calling sequence
$     SETERRQ(number,message)

   Notes for experienced users:
   Use PetscPushErrorHandler() to set the desired error handler.  The
   currently available PETSc error handlers are
$    PetscTraceBackErrorHandler()
$    PetscAttachDebuggerErrorHandler()
$    PetscAbortErrorHandler()

.keywords: abort, error, handler

.seealso: PetscPuchErrorHandler(), PetscTraceBackErrorHandler(), 
          PetscAttachDebuggerErrorHandler()
@*/
int PetscAbortErrorHandler(int line,char* dir,char *file,int number,
                           char *message,void *ctx)
{
  abort(); return 0;
}
/*@C
   PetscTraceBackErrorHandler - Default error handler routine that generates
   a traceback on error detection.

   Input Parameters:
.  line - the line number of the error (indicated by __LINE__)
.  file - the file in which the error was detected (indicated by __FILE__)
.  dir - the directory of the file (indicated by __DIR__)
.  message - an error text string, usually just printed to the screen
.  number - the user-provided error number
.  ctx - error handler context

   Notes:
   Most users need not directly employ this routine and the other error 
   handlers, but can instead use the simplified interface SETERRQ, which has 
   the calling sequence
$     SETERRQ(number,message)

   Notes for experienced users:
   Use PetscPushErrorHandler() to set the desired error handler.  The
   currently available PETSc error handlers are
$    PetscTraceBackErrorHandler()
$    PetscAttachDebuggerErrorHandler()
$    PetscAbortErrorHandler()
$    PetscStopErrorHandler()

.keywords: default, error, handler, traceback

.seealso:  PetscPushErrorHandler(), PetscAttachDebuggerErrorHandler(), 
          PetscAbortErrorHandler()
 @*/
int PetscTraceBackErrorHandler(int line,char *dir,char *file,int number,
                             char *message,void *ctx)
{
  int        tid,flg;

  MPI_Comm_rank(MPI_COMM_WORLD,&tid);
  if (number == PETSC_ERR_MEM) {
    if (!dir) fprintf(stderr,"[%d]PETSC ERROR: %s line # %d\n",tid,file,line);
    else      fprintf(stderr,"[%d]PETSC ERROR: %s%s line # %d\n",tid,dir,file,line);
    fprintf(stderr,"[%d]PETSC ERROR: Out of memory. This could be due to\n",tid);
    fprintf(stderr,"[%d]PETSC ERROR: allocating too large an object or\n",tid);
    fprintf(stderr,"[%d]PETSC ERROR: bleeding by not properly destroying\n",tid);
    fprintf(stderr,"[%d]PETSC ERROR: unneeded objects.\n",tid);
    OptionsHasName(PETSC_NULL,"-trdump",&flg);
    if (flg) {
      PetscTrDump(stderr);
    }
    else {
      fprintf(stderr,"[%d]PETSC ERROR: Try running with -trdump. \n",tid);
    }
    number = 1;
  }
  else if (number == PETSC_ERR_SUP) {
    if (!dir) fprintf(stderr,"[%d]PETSC ERROR: %s line # %d\n",tid,file,line);
    else      fprintf(stderr,"[%d]PETSC ERROR: %s%s line # %d\n",tid,dir,file,line);
    fprintf(stderr,"[%d]PETSC ERROR: %s: No support for this operation\n",tid,message);
    fprintf(stderr,"[%d]PETSC ERROR: for this object type!\n",tid);
    number = 1;
  }
  else if (number == PETSC_ERR_SIG) {
    fprintf(stderr,"[%d]PETSC ERROR: ",tid);
    fprintf(stderr,"%s %s\n",file,message);
  }
  else if (number == PETSC_ERR_SIZ) {
    if (!dir) fprintf(stderr,"[%d]PETSC ERROR: %s line # %d\n",tid,file,line);
    else      fprintf(stderr,"[%d]PETSC ERROR: %s%s line # %d\n",tid,dir,file,line);
    fprintf(stderr,"[%d]PETSC ERROR: %s: Nonconforming object sizes!\n",tid,message);
    number = 1;
  }
  else {
    fprintf(stderr,"[%d]PETSC ERROR: ",tid);
    if (!dir) {
      if (!message) fprintf(stderr,"%s line # %d\n",file,line);
      else fprintf(stderr,"%s line # %d %s\n",file,line,message);
    }
    else   {
      if (!message) fprintf(stderr,"%s%s line # %d\n",dir,file,line);
      else fprintf(stderr,"%s%s line # %d %s\n",dir,file,line,message);
    }
  }
  return number;
}

/*@C
   PetscStopErrorHandler - Calls MPI_abort() and exists.

   Input Parameters:
.  line - the line number of the error (indicated by __LINE__)
.  file - the file in which the error was detected (indicated by __FILE__)
.  dir - the directory of the file (indicated by __DIR__)
.  message - an error text string, usually just printed to the screen
.  number - the user-provided error number
.  ctx - error handler context

   Notes:
   Most users need not directly employ this routine and the other error 
   handlers, but can instead use the simplified interface SETERRQ, which has 
   the calling sequence
$     SETERRQ(number,message)

   Notes for experienced users:
   Use PetscPushErrorHandler() to set the desired error handler.  The
   currently available PETSc error handlers are
$    PetscTraceBackErrorHandler()
$    PetscStopErrorHandler()
$    PetscAttachDebuggerErrorHandler()
$    PetscAbortErrorHandler()

.keywords: default, error, handler, traceback

.seealso:  PetscPushErrorHandler(), PetscAttachDebuggerErrorHandler(), 
           PetscAbortErrorHandler(), PetscTraceBackErrorHandler()
 @*/
int PetscStopErrorHandler(int line,char *dir,char *file,int number,
                             char *message,void *ctx)
{
  int        tid,flg;

  MPI_Comm_rank(MPI_COMM_WORLD,&tid);
  if (number == PETSC_ERR_MEM) {
    if (!dir) fprintf(stderr,"[%d]PETSC ERROR: %s line # %d\n",tid,file,line);
    else      fprintf(stderr,"[%d]PETSC ERROR: %s%s line # %d\n",tid,dir,file,line);
    fprintf(stderr,"[%d]PETSC ERROR: Out of memory. This could be due to\n",tid);
    fprintf(stderr,"[%d]PETSC ERROR: allocating too large an object or\n",tid);
    fprintf(stderr,"[%d]PETSC ERROR: bleeding by not properly destroying\n",tid);
    fprintf(stderr,"[%d]PETSC ERROR: unneeded objects.\n",tid);
    OptionsHasName(PETSC_NULL,"-trdump",&flg);
    if (flg) {
      PetscTrDump(stderr);
    }
    else {
      fprintf(stderr,"[%d]PETSC ERROR: Try running with -trdump. \n",tid);
    }
    number = 1;
  }
  else if (number == PETSC_ERR_SUP) {
    if (!dir) fprintf(stderr,"[%d]PETSC ERROR: %s line # %d\n",tid,file,line);
    else      fprintf(stderr,"[%d]PETSC ERROR: %s%s line # %d\n",tid,dir,file,line);
    fprintf(stderr,"[%d]PETSC ERROR: %s: No support for this operation\n",tid,message);
    fprintf(stderr,"[%d]PETSC ERROR: for this object type!\n",tid);
    number = 1;
  }
  else if (number == PETSC_ERR_SIG) {
    fprintf(stderr,"[%d]PETSC ERROR: ",tid);
    fprintf(stderr,"%s %s\n",file,message);
  }
  else {
    fprintf(stderr,"[%d]PETSC ERROR: ",tid);
    if (!dir) {
      if (!message) fprintf(stderr,"%s line # %d\n",file,line);
      else fprintf(stderr,"%s line # %d %s\n",file,line,message);
    }
    else   {
      if (!message) fprintf(stderr,"%s%s line # %d\n",dir,file,line);
      else fprintf(stderr,"%s%s line # %d %s\n",dir,file,line,message);
    }
  }
  MPI_Abort(MPI_COMM_WORLD,number);
  return 0;
}

/*@C
   PetscPushErrorHandler - Sets a routine to be called on detection of errors.

   Input Parameters:
.  func - error handler routine

   Calling sequence of func:
   int func (int line,char *dir,char *file,char* message, int number);

.  line - the line number of the error (indicated by __LINE__)
.  file - the file in which the error was detected (indicated by __FILE__)
.  dir - the directory of the file (indicated by __DIR__)
.  message - an error text string, usually just printed to the screen
.  number - the user-provided error number

   Fortran Note:
   This routine is not supported in Fortran.

.seealso: PetscPopErrorHandler()
@*/
int PetscPushErrorHandler(int (*handler)(int,char*,char*,int,char*,void*),
                          void *ctx )
{
  struct  EH *neweh = (struct EH*) PetscMalloc(sizeof(struct EH)); CHKPTRQ(neweh);
  if (eh) {neweh->previous = eh;} 
  else {neweh->previous = 0;}
  neweh->handler = handler;
  neweh->ctx     = ctx;
  eh = neweh;
  return 0;
}
/*@C
   PetscPopErrorHandler - Removes the latest error handler that was 
   pushed with PetscPushErrorHandler().

   Fortran Note:
   This routine is not supported in Fortran.

.keywords: pop, error, handler

.seealso: PetscPushErrorHandler()
@*/
int PetscPopErrorHandler()
{
  struct EH *tmp;
  if (!eh) return 0;
  tmp = eh;
  eh = eh->previous;
  PetscFree(tmp);

  return 0;
}
/*@C
   PetscError - Routine that is called when an error has been detected, 
   usually called through the macro SETERRQ().

   Input Parameters:
.  line - the line number of the error (indicated by __LINE__)
.  dir - the directory of file (indicated by __DIR__)
.  file - the file in which the error was detected (indicated by __FILE__)
.  message - an error text string, usually just printed to the screen
.  number - the user-provided error number

   Notes:
   Most users need not directly use this routine and the error handlers, but
   can instead use the simplified interface SETERRQ, which has the calling 
   sequence
$     SETERRQ(number,message)

   Experienced users can set the error handler with PetscPushErrorHandler().

.keywords: error, SETERRQ, SETERRA

.seealso: PetscTraceBackErrorHandler(), PetscPushErrorHandler()
@*/
int PetscError(int line,char *dir,char *file,int number,char *message)
{
  if (!eh) return PetscTraceBackErrorHandler(line,dir,file,number,message,0);
  else  return (*eh->handler)(line,dir,file,number,message,eh->ctx);
}

/*
     Useful functions for debugging
*/
int IntView(int N,int* idx,Viewer viewer)
{
  int j,i,n = N/20, p = N % 20;

  for ( i=0; i<n; i++ ) {
    PetscPrintf(MPI_COMM_SELF,"%d:",20*i);
    for ( j=0; j<20; j++ ) {
       PetscPrintf(MPI_COMM_SELF," %d",idx[i*20+j]);
    }
    PetscPrintf(MPI_COMM_SELF,"\n");
  }
  if (p) {
    PetscPrintf(MPI_COMM_SELF,"%d:",20*n);
    for ( i=0; i<p; i++ ) { PetscPrintf(MPI_COMM_SELF," %d",idx[20*n+i]);}
    PetscPrintf(MPI_COMM_SELF,"\n");
  }
  return 0;
}
int DoubleView(int N,double* idx,Viewer viewer)
{
  int j,i,n = N/5, p = N % 5;

  for ( i=0; i<n; i++ ) {
    PetscPrintf(MPI_COMM_SELF,"%d:",5*i);
    for ( j=0; j<5; j++ ) {
       PetscPrintf(MPI_COMM_SELF," %6.4e",idx[i*5+j]);
    }
    PetscPrintf(MPI_COMM_SELF,"\n");
  }
  if (p) {
    PetscPrintf(MPI_COMM_SELF,"%d:",5*n);
    for ( i=0; i<p; i++ ) { PetscPrintf(MPI_COMM_SELF," %6.4e",idx[5*n+i]);}
    PetscPrintf(MPI_COMM_SELF,"\n");
  }
  return 0;
}














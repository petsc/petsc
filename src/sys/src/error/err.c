#ifndef lint
static char vcid[] = "$Id: err.c,v 1.53 1997/01/01 13:48:51 bsmith Exp balay $";
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
  int    (*handler)(int, char*,char*,char *,int,int,char*,void *);
  void   *ctx;
  struct EH* previous;
};

static struct EH* eh = 0;

#undef __FUNC__  
#define __FUNC__ "PetscAbortErrorHandler"
/*@C
   PetscAbortErrorHandler - Error handler that calls abort on error. 
   This routine is very useful when running in the debugger, because the 
   user can look directly at the stack frames and the variables.

   Input Parameters:
.  line - the line number of the error (indicated by __LINE__)
.  function - function where error occured (indicated by __FUNC__)
.  file - the file in which the error was detected (indicated by __FILE__)
.  dir - the directory of the file (indicated by __DIR__)
.  message - an error text string, usually just printed to the screen
.  number - the generic error number
.  p - specific error number
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
$     SETERRQ(number,p,message)

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
int PetscAbortErrorHandler(int line,char *function,char *file,char* dir,int number,int p,
                           char *message,void *ctx)
{
  abort(); return 0;
}

#undef __FUNC__  
#define __FUNC__ "PetscTraceBackErrorHandler"
/*@C
   PetscTraceBackErrorHandler - Default error handler routine that generates
   a traceback on error detection.

   Input Parameters:
.  line - the line number of the error (indicated by __LINE__)
.  function - the function where error is detected (indicated by __FUNC__)
.  file - the file in which the error was detected (indicated by __FILE__)
.  dir - the directory of the file (indicated by __DIR__)
.  message - an error text string, usually just printed to the screen
.  number - the generic error number
.  p - specific error number
.  ctx - error handler context

   Notes:
   Most users need not directly employ this routine and the other error 
   handlers, but can instead use the simplified interface SETERRQ, which has 
   the calling sequence
$     SETERRQ(number,p,message)

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
int PetscTraceBackErrorHandler(int line,char *fun,char* file,char *dir,int number,int p,
                               char *message,void *ctx)
{
  int rank,flg;

  if (!fun)     fun = "unknownfunction";
  if (!dir)     dir = " ";
  if (!message) message = " ";

  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  if (number == PETSC_ERR_MEM) {
    fprintf(stderr,"[%d]PETSC ERROR: %s() line %d in %s%s\n",rank,fun,line,dir,file);
    fprintf(stderr,"[%d]PETSC ERROR:   Out of memory. This could be due to allocating\n",rank);
    fprintf(stderr,"[%d]PETSC ERROR:   too large an object or bleeding by not properly\n",rank);
    fprintf(stderr,"[%d]PETSC ERROR:   destroying unneeded objects.\n",rank);
    OptionsHasName(PETSC_NULL,"-trdump",&flg);
    if (flg) {
      PetscTrDump(stderr);
    }
    else {
      fprintf(stderr,"[%d]PETSC ERROR:   Try running with -trdump for more information. \n",rank);
    }
    number = 1;
  }
  else if (number == PETSC_ERR_SUP) {
    fprintf(stderr,"[%d]PETSC ERROR: %s() line %d in %s%s\n",rank,fun,line,dir,file);
    fprintf(stderr,"[%d]PETSC ERROR:   %s: No support for this operation\n",rank,message);
    fprintf(stderr,"[%d]PETSC ERROR:   for this object type!\n",rank);
    number = 1;
  }
  else if (number == PETSC_ERR_SIG) {
    fprintf(stderr,"[%d]PETSC ERROR: %s() line %d in %s%s %s\n",rank,fun,line,dir,file,message);
  }
  else if (number == PETSC_ERR_ARG_SIZ) {
    fprintf(stderr,"[%d]PETSC ERROR: %s() line %d in %s%s\n",rank,fun,line,dir,file);
    fprintf(stderr,"[%d]PETSC ERROR:   %s: Nonconforming object sizes!\n",rank,message);
    number = 1;
  }
  else {
    fprintf(stderr,"[%d]PETSC ERROR: %s() line %d in %s%s %s\n",rank,fun,line,dir,file,message);
  }
  return number;
}

#undef __FUNC__  
#define __FUNC__ "PetscStopErrorHandler"
/*@C
   PetscStopErrorHandler - Calls MPI_abort() and exists.

   Input Parameters:
.  line - the line number of the error (indicated by __LINE__)
.  fun - the function where the error occurred (indicated by __FUNC__)
.  file - the file in which the error was detected (indicated by __FILE__)
.  dir - the directory of the file (indicated by __DIR__)
.  message - an error text string, usually just printed to the screen
.  number - the generic error number
.  p - the specific error number
.  ctx - error handler context

   Notes:
   Most users need not directly employ this routine and the other error 
   handlers, but can instead use the simplified interface SETERRQ, which has 
   the calling sequence
$     SETERRQ(number,p,message)

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
int PetscStopErrorHandler(int line,char *fun,char *file,char *dir,int number,int p,
                             char *message,void *ctx)
{
  int rank, flg;

  if (!fun)     fun = "unknownfunction";
  if (!dir)     dir = " ";
  if (!message) message = " ";

  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  if (number == PETSC_ERR_MEM) {
    fprintf(stderr,"[%d]PETSC ERROR: %s() line %d in %s%s\n",rank,fun,line,dir,file);
    fprintf(stderr,"[%d]PETSC ERROR:   Out of memory. This could be due to allocating\n",rank);
    fprintf(stderr,"[%d]PETSC ERROR:   too large an object or bleeding by not properly\n",rank);
    fprintf(stderr,"[%d]PETSC ERROR:   destroying unneeded objects.\n",rank);
    OptionsHasName(PETSC_NULL,"-trdump",&flg);
    if (flg) {
      PetscTrDump(stderr);
    }
    else {
      fprintf(stderr,"[%d]PETSC ERROR:   Try running with -trdump for more information. \n",rank);
    }
    number = 1;
  }
  else if (number == PETSC_ERR_SUP) {
    fprintf(stderr,"[%d]PETSC ERROR: %s() line %d in %s%s\n",rank,fun,line,dir,file);
    fprintf(stderr,"[%d]PETSC ERROR:   %s: No support for this operation\n",rank,message);
    fprintf(stderr,"[%d]PETSC ERROR:   for this object type!\n",rank);
    number = 1;
  }
  else if (number == PETSC_ERR_SIG) {
    fprintf(stderr,"[%d]PETSC ERROR: %s() line %d in %s%s %s\n",rank,fun,line,dir,file,message);
  }
  else {
    fprintf(stderr,"[%d]PETSC ERROR: %s() line %d in %s%s %s\n",rank,fun,line,dir,file,message);
  }
  MPI_Abort(PETSC_COMM_WORLD,number);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PetscPushErrorHandler"
/*@C
   PetscPushErrorHandler - Sets a routine to be called on detection of errors.

   Input Parameters:
.  func - error handler routine

   Calling sequence of func:
   int func (int line,char *func,char *file,char *dir,int number,int p,char *message);

.  function - the function where the error occured (indicated by __FUNC__)
.  line - the line number of the error (indicated by __LINE__)
.  file - the file in which the error was detected (indicated by __FILE__)
.  dir - the directory of the file (indicated by __DIR__)
.  message - an error text string, usually just printed to the screen
.  number - the generic error number (see list defined in include/petscerror.h)
.  p - the specific error number

   Fortran Note:
   This routine is not supported in Fortran.

.seealso: PetscPopErrorHandler()
@*/
int PetscPushErrorHandler(int (*handler)(int,char *,char*,char*,int,int,char*,void*),void *ctx )
{
  struct  EH *neweh = (struct EH*) PetscMalloc(sizeof(struct EH)); CHKPTRQ(neweh);
  if (eh) {neweh->previous = eh;} 
  else {neweh->previous = 0;}
  neweh->handler = handler;
  neweh->ctx     = ctx;
  eh = neweh;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PetscPopErrorHandler"
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

#undef __FUNC__  
#define __FUNC__ "PetscError"
/*@C
   PetscError - Routine that is called when an error has been detected, 
   usually called through the macro SETERRQ().

   Input Parameters:
.  line - the line number of the error (indicated by __LINE__)
.  function - the function where the error occured (indicated by __FUNC__)
.  dir - the directory of file (indicated by __DIR__)
.  file - the file in which the error was detected (indicated by __FILE__)
.  message - an error text string, usually just printed to the screen
.  number - the generic error number
.  p - the specific error number

   Notes:
   Most users need not directly use this routine and the error handlers, but
   can instead use the simplified interface SETERRQ, which has the calling 
   sequence
$     SETERRQ(number,p,message)

   Experienced users can set the error handler with PetscPushErrorHandler().

.keywords: error, SETERRQ, SETERRA

.seealso: PetscTraceBackErrorHandler(), PetscPushErrorHandler()
@*/
int PetscError(int line,char *function,char* file,char *dir,int number,int p,char *message)
{
  if (!eh) return PetscTraceBackErrorHandler(line,function,file,dir,number,p,message,0);
  else  return (*eh->handler)(line,function,file,dir,number,p,message,eh->ctx);
}

#undef __FUNC__  
#define __FUNC__ "PetscIntView"
/*@C
    PetscIntView - Prints an array of integers, useful for debugging.

  Input Parameters:
.   N - number of integers in array
.   idx - array of integers
.   viewer - location to print array,  VIEWER_STDOUT_WORLD, VIEWER_STDOUT_SELF or 0

  Notes:
    If using a viewer with more then one processor you must call PetscSynchronizedFlush()
   after this call to get all processors to print to the screen.

.seealso: PetscDoubleView() 
@*/
int PetscIntView(int N,int* idx,Viewer viewer)
{
  int      j,i,n = N/20, p = N % 20,ierr;
  MPI_Comm comm;

  if (viewer) PetscValidHeader(viewer);
  PetscValidIntPointer(idx);

  if (viewer) {
    ierr = PetscObjectGetComm((PetscObject) viewer,&comm); CHKERRQ(ierr);
  } else {
    comm = MPI_COMM_SELF;
  }

  for ( i=0; i<n; i++ ) {
    PetscSynchronizedPrintf(comm,"%d:",20*i);
    for ( j=0; j<20; j++ ) {
       PetscSynchronizedPrintf(comm," %d",idx[i*20+j]);
    }
    PetscSynchronizedPrintf(comm,"\n");
  }
  if (p) {
    PetscSynchronizedPrintf(comm,"%d:",20*n);
    for ( i=0; i<p; i++ ) { PetscSynchronizedPrintf(comm," %d",idx[20*n+i]);}
    PetscSynchronizedPrintf(comm,"\n");
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "PetscDoubleView"
/*@C
    PetscDoubleView - Prints an array of double, useful for debugging.

  Input Parameters:
.   N - number of doubles in array
.   idx - array of doubles
.   viewer - location to print array,  VIEWER_STDOUT_WORLD, VIEWER_STDOUT_SELF or 0

  Notes:
    If using a viewer with more then one processor you must call PetscSynchronizedFlush()
   after this call to get all processors to print to the screen.

.seealso: PetscIntView() 
@*/
int PetscDoubleView(int N,double* idx,Viewer viewer)
{
  int      j,i,n = N/5, p = N % 5,ierr;
  MPI_Comm comm;

  if (viewer) PetscValidHeader(viewer);
  PetscValidScalarPointer(idx);

  if (viewer) {
    ierr = PetscObjectGetComm((PetscObject) viewer,&comm); CHKERRQ(ierr);
  } else {
    comm = MPI_COMM_SELF;
  }

  for ( i=0; i<n; i++ ) {
    for ( j=0; j<5; j++ ) {
       PetscSynchronizedPrintf(comm," %6.4e",idx[i*5+j]);
    }
    PetscSynchronizedPrintf(comm,"\n");
  }
  if (p) {
    PetscSynchronizedPrintf(comm,"%d:",5*n);
    for ( i=0; i<p; i++ ) { PetscSynchronizedPrintf(comm," %6.4e",idx[5*n+i]);}
    PetscSynchronizedPrintf(comm,"\n");
  }
  return 0;
}











#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: err.c,v 1.73 1998/04/03 21:46:07 balay Exp bsmith $";
#endif
/*
       The default error handlers and code that allows one to change
   error handlers.
*/
#include "petsc.h"           /*I "petsc.h" I*/
#if defined(HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#include "pinclude/petscfix.h"
#include "pinclude/pviewer.h"

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
.  func - function where error occured (indicated by __FUNC__)
.  file - the file in which the error was detected (indicated by __FILE__)
.  dir - the directory of the file (indicated by __SDIR__)
.  mess - an error text string, usually just printed to the screen
.  n - the generic error number
.  p - specific error number
.  ctx - error handler context

   Not Collective

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
$     SETERRQ(number,p,mess)

   Notes for experienced users:
   Use PetscPushErrorHandler() to set the desired error handler.  The
   currently available PETSc error handlers are
$    PetscTraceBackErrorHandler()
$    PetscAttachDebuggerErrorHandler()
$    PetscAbortErrorHandler()

.keywords: abort, error, handler

.seealso: PetscPushErrorHandler(), PetscTraceBackErrorHandler(), 
          PetscAttachDebuggerErrorHandler()
@*/
int PetscAbortErrorHandler(int line,char *func,char *file,char* dir,int n,int p,char *mess,void *ctx)
{
  PetscFunctionBegin;

  abort(); 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscTraceBackErrorHandler"
/*@C
   PetscTraceBackErrorHandler - Default error handler routine that generates
   a traceback on error detection.

   Input Parameters:
.  line - the line number of the error (indicated by __LINE__)
.  func - the function where error is detected (indicated by __FUNC__)
.  file - the file in which the error was detected (indicated by __FILE__)
.  dir - the directory of the file (indicated by __SDIR__)
.  mess - an error text string, usually just printed to the screen
.  n - the generic error number
.  p - specific error number
.  ctx - error handler context

   Not Collective

   Notes:
   Most users need not directly employ this routine and the other error 
   handlers, but can instead use the simplified interface SETERRQ, which has 
   the calling sequence
$     SETERRQ(number,p,mess)

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
int PetscTraceBackErrorHandler(int line,char *fun,char* file,char *dir,int n,int p,char *mess,void *ctx)
{
  int        rank,flg1,flg2;
  PLogDouble mem,rss;

  PetscFunctionBegin;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  if (n == PETSC_ERR_MEM) {
    (*PetscErrorPrintf)("[%d]PETSC ERROR: %s() line %d in %s%s\n",rank,fun,line,dir,file);
    (*PetscErrorPrintf)("[%d]PETSC ERROR:   Out of memory. This could be due to allocating\n",rank);
    (*PetscErrorPrintf)("[%d]PETSC ERROR:   too large an object or bleeding by not properly\n",rank);
    (*PetscErrorPrintf)("[%d]PETSC ERROR:   destroying unneeded objects.\n",rank);
    PetscTrSpace(&mem,PETSC_NULL,PETSC_NULL); PetscGetResidentSetSize(&rss);
    OptionsHasName(PETSC_NULL,"-trdump",&flg1);
    OptionsHasName(PETSC_NULL,"-trmalloc_log",&flg2);
    if (flg2) {
      PetscTrLogDump(stderr);
    } else if (flg1) {
      (*PetscErrorPrintf)("[%d]PETSC ERROR:   Memory allocated %d Memory used by process %d\n",rank,(int)mem,(int)rss);
      PetscTrDump(stderr);
    }  else {
      (*PetscErrorPrintf)("[%d]PETSC ERROR:   Memory allocated %d Memory used by process %d\n",rank,(int)mem,(int)rss);
      (*PetscErrorPrintf)("[%d]PETSC ERROR:   Try running with -trdump or -trmalloc_log for info.\n",rank);
    }
    n = 1;
  } else if (n == PETSC_ERR_SUP) {
    if (!mess) mess = " ";
    (*PetscErrorPrintf)("[%d]PETSC ERROR: %s() line %d in %s%s\n",rank,fun,line,dir,file);
    (*PetscErrorPrintf)("[%d]PETSC ERROR: No support for this operation for this object type!\n",rank);
    (*PetscErrorPrintf)("[%d]PETSC ERROR: %s\n",rank,mess);
    n = 1;
  } else if (n == PETSC_ERR_SIG) {
    (*PetscErrorPrintf)("[%d]PETSC ERROR: %s() line %d in %s%s %s\n",rank,fun,line,dir,file,mess);
  } else if (n == PETSC_ERR_ARG_SIZ) {
    if (!mess) mess = " ";
    (*PetscErrorPrintf)("[%d]PETSC ERROR: %s() line %d in %s%s\n",rank,fun,line,dir,file);
    (*PetscErrorPrintf)("[%d]PETSC ERROR:   %s: Nonconforming object sizes!\n",rank,mess);
    n = 1;
  } else {
    if (mess) {
      (*PetscErrorPrintf)("[%d]PETSC ERROR: %s() line %d in %s%s\n    %s\n",rank,fun,line,dir,file,mess);
    } else {
      (*PetscErrorPrintf)("[%d]PETSC ERROR: %s() line %d in %s%s\n",rank,fun,line,dir,file);
    }
  }
  PetscFunctionReturn(n);
}

#undef __FUNC__  
#define __FUNC__ "PetscStopErrorHandler"
/*@C
   PetscStopErrorHandler - Calls MPI_abort() and exists.

   Input Parameters:
.  line - the line number of the error (indicated by __LINE__)
.  fun - the function where the error occurred (indicated by __FUNC__)
.  file - the file in which the error was detected (indicated by __FILE__)
.  dir - the directory of the file (indicated by __SDIR__)
.  mess - an error text string, usually just printed to the screen
.  n - the generic error number
.  p - the specific error number
.  ctx - error handler context

   Not Collective

   Notes:
   Most users need not directly employ this routine and the other error 
   handlers, but can instead use the simplified interface SETERRQ, which has 
   the calling sequence
$     SETERRQ(n,p,mess)

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
int PetscStopErrorHandler(int line,char *fun,char *file,char *dir,int n,int p,char *mess,void *ctx)
{
  int        rank, flg1, flg2;
  PLogDouble mem,rss;

  PetscFunctionBegin;
  if (!mess) mess = " ";

  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  if (n == PETSC_ERR_MEM) {
    (*PetscErrorPrintf)("[%d]PETSC ERROR: %s() line %d in %s%s\n",rank,fun,line,dir,file);
    (*PetscErrorPrintf)("[%d]PETSC ERROR:   Out of memory. This could be due to allocating\n",rank);
    (*PetscErrorPrintf)("[%d]PETSC ERROR:   too large an object or bleeding by not properly\n",rank);
    (*PetscErrorPrintf)("[%d]PETSC ERROR:   destroying unneeded objects.\n",rank);
    PetscTrSpace(&mem,PETSC_NULL,PETSC_NULL); PetscGetResidentSetSize(&rss);
    OptionsHasName(PETSC_NULL,"-trdump",&flg1);
    OptionsHasName(PETSC_NULL,"-trmalloc_log",&flg2);
    if (flg2) {
      PetscTrLogDump(stderr);
    } else if (flg1) {
      (*PetscErrorPrintf)("[%d]PETSC ERROR:   Memory allocated %d Memory used by process %d\n",rank,(int)mem,(int)rss);
      PetscTrDump(stderr);
    }  else {
      (*PetscErrorPrintf)("[%d]PETSC ERROR:   Memory allocated %d Memory used by process %d\n",rank,(int)mem,(int)rss);
      (*PetscErrorPrintf)("[%d]PETSC ERROR:   Try running with -trdump or -trmalloc_log for info.\n",rank);
    }
    n = 1;
  } else if (n == PETSC_ERR_SUP) {
    (*PetscErrorPrintf)("[%d]PETSC ERROR: %s() line %d in %s%s\n",rank,fun,line,dir,file);
    (*PetscErrorPrintf)("[%d]PETSC ERROR: No support for this operation for this object type!\n",rank);
    (*PetscErrorPrintf)("[%d]PETSC ERROR: %s\n",rank,mess);
    n = 1;
  } else if (n == PETSC_ERR_SIG) {
    (*PetscErrorPrintf)("[%d]PETSC ERROR: %s() line %d in %s%s %s\n",rank,fun,line,dir,file,mess);
  } else {
    (*PetscErrorPrintf)("[%d]PETSC ERROR: %s() line %d in %s%s\n    %s\n",rank,fun,line,dir,file,mess);
  }
  MPI_Abort(PETSC_COMM_WORLD,n);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscPushErrorHandler"
/*@C
   PetscPushErrorHandler - Sets a routine to be called on detection of errors.

   Input Parameters:
.  func - error handler routine

   Not Collective

   Calling sequence of func:
   int func (int line,char *func,char *file,char *dir,int n,int p,char *mess);

.  func - the function where the error occured (indicated by __FUNC__)
.  line - the line number of the error (indicated by __LINE__)
.  file - the file in which the error was detected (indicated by __FILE__)
.  dir - the directory of the file (indicated by __SDIR__)
.  mess - an error text string, usually just printed to the screen
.  n - the generic error number (see list defined in include/petscerror.h)
.  p - the specific error number

   Fortran Note:
   This routine is not supported in Fortran.

.seealso: PetscPopErrorHandler()
@*/
int PetscPushErrorHandler(int (*handler)(int,char *,char*,char*,int,int,char*,void*),void *ctx )
{
  struct  EH *neweh = (struct EH*) PetscMalloc(sizeof(struct EH)); CHKPTRQ(neweh);

  PetscFunctionBegin;
  if (eh) {neweh->previous = eh;} 
  else    {neweh->previous = 0;}
  neweh->handler = handler;
  neweh->ctx     = ctx;
  eh = neweh;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscPopErrorHandler"
/*@C
   PetscPopErrorHandler - Removes the latest error handler that was 
   pushed with PetscPushErrorHandler().

   Not Collective

   Fortran Note:
   This routine is not supported in Fortran.

.keywords: pop, error, handler

.seealso: PetscPushErrorHandler()
@*/
int PetscPopErrorHandler(void)
{
  struct EH *tmp;

  PetscFunctionBegin;
  if (!eh) PetscFunctionReturn(0);
  tmp = eh;
  eh  = eh->previous;
  PetscFree(tmp);

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscError"
/*@C
   PetscError - Routine that is called when an error has been detected, 
   usually called through the macro SETERRQ().

   Input Parameters:
.  line - the line number of the error (indicated by __LINE__)
.  func - the function where the error occured (indicated by __FUNC__)
.  dir - the directory of file (indicated by __SDIR__)
.  file - the file in which the error was detected (indicated by __FILE__)
.  mess - an error text string, usually just printed to the screen
.  n - the generic error number
.  p - the specific error number

   Not Collective

   Notes:
   Most users need not directly use this routine and the error handlers, but
   can instead use the simplified interface SETERRQ, which has the calling 
   sequence
$     SETERRQ(n,p,mess)

   Experienced users can set the error handler with PetscPushErrorHandler().

.keywords: error, SETERRQ, SETERRA

.seealso: PetscTraceBackErrorHandler(), PetscPushErrorHandler()
@*/
int PetscError(int line,char *func,char* file,char *dir,int n,int p,char *mess)
{
  int ierr;

  PetscFunctionBegin;
  if (!eh)     ierr = PetscTraceBackErrorHandler(line,func,file,dir,n,p,mess,0);
  else         ierr = (*eh->handler)(line,func,file,dir,n,p,mess,eh->ctx);
  PetscFunctionReturn(ierr);
}

#undef __FUNC__  
#define __FUNC__ "PetscIntView"
/*@C
    PetscIntView - Prints an array of integers; useful for debugging.

    Input Parameters:
.   N - number of integers in array
.   idx - array of integers
.   viewer - location to print array,  VIEWER_STDOUT_WORLD, VIEWER_STDOUT_SELF or 0

   Collective on Viewer

    Notes:
    If using a viewer with more than one processor, you must call PetscSynchronizedFlush()
    after this call to get all processors to print to the screen.

.seealso: PetscDoubleView() 
@*/
int PetscIntView(int N,int* idx,Viewer viewer)
{
  int        j,i,n = N/20, p = N % 20,ierr;
  MPI_Comm   comm;
  ViewerType vtype;
  FILE       *file;

  PetscFunctionBegin;
  if (!viewer) {
    viewer = VIEWER_STDOUT_SELF;
  }
  PetscValidHeader(viewer);
  PetscValidIntPointer(idx);
  ierr = PetscObjectGetComm((PetscObject) viewer,&comm); CHKERRQ(ierr);

  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  if (vtype == ASCII_FILE_VIEWER || vtype == ASCII_FILES_VIEWER) {
    ierr = ViewerASCIIGetPointer(viewer,&file);CHKERRQ(ierr);

    for ( i=0; i<n; i++ ) {
      PetscSynchronizedFPrintf(comm,file,"%d:",20*i);
      for ( j=0; j<20; j++ ) {
        PetscSynchronizedFPrintf(comm,file," %d",idx[i*20+j]);
      }
      PetscSynchronizedFPrintf(comm,file,"\n");
    }
    if (p) {
      PetscSynchronizedFPrintf(comm,file,"%d:",20*n);
      for ( i=0; i<p; i++ ) { PetscSynchronizedFPrintf(comm,file," %d",idx[20*n+i]);}
      PetscSynchronizedFPrintf(comm,file,"\n");
    }
    PetscSynchronizedFlush(comm);
  } else if (vtype == MATLAB_VIEWER) {
    int *array,*sizes,rank,size,Ntotal,*displs;

    MPI_Comm_rank(comm,&rank);
    MPI_Comm_size(comm,&size);

    if (size > 1) {
      
      if (rank) {
        MPI_Gather(&N,1,MPI_INT,0,0,MPI_INT,0,comm);
        MPI_Gatherv(idx,N,MPI_INT,0,0,0,MPI_INT,0,comm);
      } else {
        sizes = (int *) PetscMalloc(size*sizeof(int));CHKPTRQ(sizes);
        MPI_Gather(&N,1,MPI_INT,sizes,1,MPI_INT,0,comm);
        Ntotal    = sizes[0]; 
        displs    = (int *) PetscMalloc(size*sizeof(int));CHKPTRQ(sizes);
        displs[0] = 0;
        for (i=1; i<size; i++) {
          Ntotal    += sizes[i];
          displs[i] =  displs[i-1] + sizes[i-1];
        }
        array  = (int *) PetscMalloc(Ntotal*sizeof(int));CHKPTRQ(array);
        MPI_Gatherv(idx,N,MPI_INT,array,sizes,displs,MPI_INT,0,comm);
        ierr = ViewerMatlabPutInt_Private(viewer,Ntotal,array);CHKERRQ(ierr);
        PetscFree(sizes);
        PetscFree(displs);
        PetscFree(array);
      }
    } else {
      ierr = ViewerMatlabPutInt_Private(viewer,N,idx);CHKERRQ(ierr);
    }
  } else {
    SETERRQ(1,1,"Cannot handle that viewer");
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscDoubleView"
/*@C
    PetscDoubleView - Prints an array of doubles; useful for debugging.

    Input Parameters:
.   N - number of doubles in array
.   idx - array of doubles
.   viewer - location to print array,  VIEWER_STDOUT_WORLD, VIEWER_STDOUT_SELF or 0

   Collective on Viewer

   Notes:
   If using a viewer with more than one processor, you must call PetscSynchronizedFlush()
   after this call to get all processors to print to the screen.

.seealso: PetscIntView() 
@*/
int PetscDoubleView(int N,double* idx,Viewer viewer)
{
  int        j,i,n = N/5, p = N % 5,ierr;
  MPI_Comm   comm;
  ViewerType vtype;
  FILE       *file;

  PetscFunctionBegin;
  if (!viewer) viewer = VIEWER_STDOUT_SELF;
  PetscValidHeader(viewer);
  PetscValidScalarPointer(idx);
  ierr = PetscObjectGetComm((PetscObject) viewer,&comm); CHKERRQ(ierr);

  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  if (vtype == ASCII_FILE_VIEWER || vtype == ASCII_FILES_VIEWER) {
    ierr = ViewerASCIIGetPointer(viewer,&file);CHKERRQ(ierr);

    for ( i=0; i<n; i++ ) {
      for ( j=0; j<5; j++ ) {
         PetscSynchronizedFPrintf(comm,file," %6.4e",idx[i*5+j]);
      }
      PetscSynchronizedFPrintf(comm,file,"\n");
    }
    if (p) {
      PetscSynchronizedFPrintf(comm,file,"%d:",5*n);
      for ( i=0; i<p; i++ ) { PetscSynchronizedFPrintf(comm,file," %6.4e",idx[5*n+i]);}
      PetscSynchronizedFPrintf(comm,file,"\n");
    }
    PetscSynchronizedFlush(comm);
  } else if (vtype == MATLAB_VIEWER) {
    int    *sizes,rank,size,Ntotal,*displs;
    double *array;

    MPI_Comm_rank(comm,&rank);
    MPI_Comm_size(comm,&size);

    if (size > 1) {
      
      if (rank) {
        MPI_Gather(&N,1,MPI_INT,0,0,MPI_INT,0,comm);
        MPI_Gatherv(idx,N,MPI_DOUBLE,0,0,0,MPI_DOUBLE,0,comm);
      } else {
        sizes = (int *) PetscMalloc(size*sizeof(int));CHKPTRQ(sizes);
        MPI_Gather(&N,1,MPI_INT,sizes,1,MPI_INT,0,comm);
        Ntotal    = sizes[0]; 
        displs    = (int *) PetscMalloc(size*sizeof(int));CHKPTRQ(sizes);
        displs[0] = 0;
        for (i=1; i<size; i++) {
          Ntotal    += sizes[i];
          displs[i] =  displs[i-1] + sizes[i-1];
        }
        array  = (double *) PetscMalloc(Ntotal*sizeof(double));CHKPTRQ(array);
        MPI_Gatherv(idx,N,MPI_DOUBLE,array,sizes,displs,MPI_DOUBLE,0,comm);
        ierr = ViewerMatlabPutDouble_Private(viewer,Ntotal,1,array);CHKERRQ(ierr);
        PetscFree(sizes);
        PetscFree(displs);
        PetscFree(array);
      }
    } else {
      ierr = ViewerMatlabPutDouble_Private(viewer,N,1,idx);CHKERRQ(ierr);
    }
  } else {
    SETERRQ(1,1,"Cannot handle that viewer");
  }
  PetscFunctionReturn(0);
}











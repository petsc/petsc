/*$Id: err.c,v 1.130 2001/09/07 15:24:29 bsmith Exp $*/
/*
      Code that allows one to set the error handlers
*/
#include "petsc.h"           /*I "petsc.h" I*/
#include "petscsys.h"
#include <stdarg.h>
#if defined(PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#endif

typedef struct _EH *EH;
struct _EH {
  int    cookie;
  int    (*handler)(int,char*,char*,char *,int,int,char*,void *);
  void   *ctx;
  EH     previous;
};

static EH eh = 0;

#undef __FUNCT__  
#define __FUNCT__ "PetscEmacsClientErrorHandler" 
/*@C
   PetscEmacsClientErrorHandler - Error handler that uses the emacsclient program to 
    load the file where the error occured. Then calls the "previous" error handler.

   Not Collective

   Input Parameters:
+  line - the line number of the error (indicated by __LINE__)
.  func - the function where error is detected (indicated by __FUNCT__)
.  file - the file in which the error was detected (indicated by __FILE__)
.  dir - the directory of the file (indicated by __SDIR__)
.  mess - an error text string, usually just printed to the screen
.  n - the generic error number
.  p - specific error number
-  ctx - error handler context

   Options Database Key:
.   -on_error_emacs <machinename>

   Level: developer

   Notes:
   You must put (server-start) in your .emacs file for the emacsclient software to work

   Most users need not directly employ this routine and the other error 
   handlers, but can instead use the simplified interface SETERRQ, which has 
   the calling sequence
$     SETERRQ(number,p,mess)

   Notes for experienced users:
   Use PetscPushErrorHandler() to set the desired error handler.  The
   currently available PETSc error handlers include PetscTraceBackErrorHandler(),
   PetscAttachDebuggerErrorHandler(), PetscAbortErrorHandler(), and PetscStopErrorHandler()

   Concepts: emacs^going to on error
   Concepts: error handler^going to line in emacs

.seealso:  PetscPushErrorHandler(), PetscAttachDebuggerErrorHandler(), 
          PetscAbortErrorHandler()
 @*/
int PetscEmacsClientErrorHandler(int line,char *fun,char* file,char *dir,int n,int p,char *mess,void *ctx)
{
  int         ierr;
  char        command[PETSC_MAX_PATH_LEN],*pdir;
  FILE        *fp;

  PetscFunctionBegin;
  /* Note: don't check error codes since this an error handler :-) */
  ierr = PetscGetPetscDir(&pdir);CHKERRQ(ierr);
  sprintf(command,"emacsclient +%d %s/%s%s\n",line,pdir,dir,file);
  ierr = PetscPOpen(MPI_COMM_WORLD,(char*)ctx,command,"r",&fp);
  ierr = PetscFClose(MPI_COMM_WORLD,fp);
  ierr = PetscPopErrorHandler(); /* remove this handler from the stack of handlers */
  if (!eh)     ierr = PetscTraceBackErrorHandler(line,fun,file,dir,n,p,mess,0);
  else         ierr = (*eh->handler)(line,fun,file,dir,n,p,mess,eh->ctx);
  PetscFunctionReturn(ierr);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscPushErrorHandler" 
/*@C
   PetscPushErrorHandler - Sets a routine to be called on detection of errors.

   Not Collective

   Input Parameters:
+  handler - error handler routine
-  ctx - optional handler context that contains information needed by the handler (for 
         example file pointers for error messages etc.)

   Calling sequence of handler:
$    int handler(int line,char *func,char *file,char *dir,int n,int p,char *mess,void *ctx);

+  func - the function where the error occured (indicated by __FUNCT__)
.  line - the line number of the error (indicated by __LINE__)
.  file - the file in which the error was detected (indicated by __FILE__)
.  dir - the directory of the file (indicated by __SDIR__)
.  n - the generic error number (see list defined in include/petscerror.h)
.  p - the specific error number
.  mess - an error text string, usually just printed to the screen
-  ctx - the error handler context

   Options Database Keys:
+   -on_error_attach_debugger <noxterm,gdb or dbx>
-   -on_error_abort

   Level: intermediate

   Fortran Note:
   This routine is not supported in Fortran.

.seealso: PetscPopErrorHandler(), PetscAttachDebuggerErrorHandler(), PetscAbortErrorHandler(), PetscTraceBackErrorHandler()

@*/
int PetscPushErrorHandler(int (*handler)(int,char *,char*,char*,int,int,char*,void*),void *ctx)
{
  EH  neweh;
  int ierr;

  PetscFunctionBegin;
  ierr = PetscNew(struct _EH,&neweh);CHKERRQ(ierr);
  if (eh) {neweh->previous = eh;} 
  else    {neweh->previous = 0;}
  neweh->handler = handler;
  neweh->ctx     = ctx;
  eh             = neweh;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscPopErrorHandler" 
/*@C
   PetscPopErrorHandler - Removes the latest error handler that was 
   pushed with PetscPushErrorHandler().

   Not Collective

   Level: intermediate

   Fortran Note:
   This routine is not supported in Fortran.

   Concepts: error handler^setting

.seealso: PetscPushErrorHandler()
@*/
int PetscPopErrorHandler(void)
{
  EH  tmp;
  int ierr;

  PetscFunctionBegin;
  if (!eh) PetscFunctionReturn(0);
  tmp  = eh;
  eh   = eh->previous;
  ierr = PetscFree(tmp);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
 
char PetscErrorBaseMessage[1024];

#undef __FUNCT__  
#define __FUNCT__ "PetscError" 
/*@C
   PetscError - Routine that is called when an error has been detected, 
   usually called through the macro SETERRQ().

   Not Collective

   Input Parameters:
+  line - the line number of the error (indicated by __LINE__)
.  func - the function where the error occured (indicated by __FUNCT__)
.  dir - the directory of file (indicated by __SDIR__)
.  file - the file in which the error was detected (indicated by __FILE__)
.  mess - an error text string, usually just printed to the screen
.  n - the generic error number
.  p - 1 indicates the error was initially detected, 0 indicates this is a traceback from a 
   previously detected error
-  mess - formatted message string - aka printf

  Level: intermediate

   Notes:
   Most users need not directly use this routine and the error handlers, but
   can instead use the simplified interface SETERRQ, which has the calling 
   sequence
$     SETERRQ(n,mess)

   Experienced users can set the error handler with PetscPushErrorHandler().

   Concepts: error^setting condition

.seealso: PetscTraceBackErrorHandler(), PetscPushErrorHandler(), SETERRQ(), CHKERRQ(), CHKMEMQ(), SETERRQ1(), SETERRQ2()
@*/
int PetscError(int line,char *func,char* file,char *dir,int n,int p,char *mess,...)
{
  va_list     Argp;
  int         ierr;
  char        buf[2048],*lbuf = 0;
  PetscTruth  ismain,isunknown;

  PetscFunctionBegin;
  /* Compose the message evaluating the print format */
  if (mess) {
    va_start(Argp,mess);
#if defined(PETSC_HAVE_VPRINTF_CHAR)
    vsprintf(buf,mess,(char *)Argp);
#else
    vsprintf(buf,mess,Argp);
#endif
    va_end(Argp);
    lbuf = buf;
    if (p == 1) {
      PetscStrncpy(PetscErrorBaseMessage,lbuf,1023);
    }
  }

  if (!eh)     ierr = PetscTraceBackErrorHandler(line,func,file,dir,n,p,lbuf,0);
  else         ierr = (*eh->handler)(line,func,file,dir,n,p,lbuf,eh->ctx);

  /* 
      If this is called from the main() routine we call MPI_Abort() instead of 
    return to allow the parallel program to be properly shutdown.

    Since this is in the error handler we don't check the errors below. Of course,
    PetscStrncmp() does its own error checking which is problamatic
  */
  PetscStrncmp(func,"main",4,&ismain);
  PetscStrncmp(func,"unknown",7,&isunknown);
  if (ismain || isunknown) {
    MPI_Abort(PETSC_COMM_WORLD,ierr);
  }
  PetscFunctionReturn(ierr);
}

/* -------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "PetscIntView" 
/*@C
    PetscIntView - Prints an array of integers; useful for debugging.

    Collective on PetscViewer

    Input Parameters:
+   N - number of integers in array
.   idx - array of integers
-   viewer - location to print array,  PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_STDOUT_SELF or 0

  Level: intermediate

.seealso: PetscRealView() 
@*/
int PetscIntView(int N,int idx[],PetscViewer viewer)
{
  int        j,i,n = N/20,p = N % 20,ierr;
  PetscTruth isascii,issocket;
  MPI_Comm   comm;

  PetscFunctionBegin;
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_SELF;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_COOKIE);
  PetscValidIntPointer(idx);
  ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);

  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&isascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_SOCKET,&issocket);CHKERRQ(ierr);
  if (isascii) {
    for (i=0; i<n; i++) {
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"%d:",20*i);CHKERRQ(ierr);
      for (j=0; j<20; j++) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer," %d",idx[i*20+j]);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"\n");CHKERRQ(ierr);
    }
    if (p) {
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"%d:",20*n);CHKERRQ(ierr);
      for (i=0; i<p; i++) { ierr = PetscViewerASCIISynchronizedPrintf(viewer," %d",idx[20*n+i]);CHKERRQ(ierr);}
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  } else if (issocket) {
    int *array,*sizes,rank,size,Ntotal,*displs;

    ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
    ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);

    if (size > 1) {
      if (rank) {
        ierr = MPI_Gather(&N,1,MPI_INT,0,0,MPI_INT,0,comm);CHKERRQ(ierr);
        ierr = MPI_Gatherv(idx,N,MPI_INT,0,0,0,MPI_INT,0,comm);CHKERRQ(ierr);
      } else {
	ierr      = PetscMalloc(size*sizeof(int),&sizes);CHKERRQ(ierr);
        ierr      = MPI_Gather(&N,1,MPI_INT,sizes,1,MPI_INT,0,comm);CHKERRQ(ierr);
        Ntotal    = sizes[0]; 
	ierr      = PetscMalloc(size*sizeof(int),&displs);CHKERRQ(ierr);
        displs[0] = 0;
        for (i=1; i<size; i++) {
          Ntotal    += sizes[i];
          displs[i] =  displs[i-1] + sizes[i-1];
        }
	ierr = PetscMalloc(Ntotal*sizeof(int),&array);CHKERRQ(ierr);
        ierr = MPI_Gatherv(idx,N,MPI_INT,array,sizes,displs,MPI_INT,0,comm);CHKERRQ(ierr);
        ierr = PetscViewerSocketPutInt(viewer,Ntotal,array);CHKERRQ(ierr);
        ierr = PetscFree(sizes);CHKERRQ(ierr);
        ierr = PetscFree(displs);CHKERRQ(ierr);
        ierr = PetscFree(array);CHKERRQ(ierr);
      }
    } else {
      ierr = PetscViewerSocketPutInt(viewer,N,idx);CHKERRQ(ierr);
    }
  } else {
    SETERRQ(1,"Cannot handle that PetscViewer");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscRealView" 
/*@C
    PetscRealView - Prints an array of doubles; useful for debugging.

    Collective on PetscViewer

    Input Parameters:
+   N - number of doubles in array
.   idx - array of doubles
-   viewer - location to print array,  PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_STDOUT_SELF or 0

  Level: intermediate

.seealso: PetscIntView() 
@*/
int PetscRealView(int N,PetscReal idx[],PetscViewer viewer)
{
  int        j,i,n = N/5,p = N % 5,ierr;
  PetscTruth isascii,issocket;
  MPI_Comm   comm;

  PetscFunctionBegin;
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_SELF;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_COOKIE);
  PetscValidScalarPointer(idx);
  ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);

  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&isascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_SOCKET,&issocket);CHKERRQ(ierr);
  if (isascii) {
    for (i=0; i<n; i++) {
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"%2d:",5*i);CHKERRQ(ierr);
      for (j=0; j<5; j++) {
         ierr = PetscViewerASCIISynchronizedPrintf(viewer," %12.4e",idx[i*5+j]);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"\n");CHKERRQ(ierr);
    }
    if (p) {
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"%2d:",5*n);CHKERRQ(ierr);
      for (i=0; i<p; i++) { PetscViewerASCIISynchronizedPrintf(viewer," %12.4e",idx[5*n+i]);CHKERRQ(ierr);}
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  } else if (issocket) {
    int    *sizes,rank,size,Ntotal,*displs;
    PetscReal *array;

    ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
    ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);

    if (size > 1) {
      if (rank) {
        ierr = MPI_Gather(&N,1,MPI_INT,0,0,MPI_INT,0,comm);CHKERRQ(ierr);
        ierr = MPI_Gatherv(idx,N,MPI_DOUBLE,0,0,0,MPI_DOUBLE,0,comm);CHKERRQ(ierr);
      } else {
	ierr   = PetscMalloc(size*sizeof(int),&sizes);CHKERRQ(ierr);
        ierr   = MPI_Gather(&N,1,MPI_INT,sizes,1,MPI_INT,0,comm);CHKERRQ(ierr);
        Ntotal = sizes[0]; 
	ierr   = PetscMalloc(size*sizeof(int),&displs);CHKERRQ(ierr);
        displs[0] = 0;
        for (i=1; i<size; i++) {
          Ntotal    += sizes[i];
          displs[i] =  displs[i-1] + sizes[i-1];
        }
	ierr = PetscMalloc(Ntotal*sizeof(PetscReal),&array);CHKERRQ(ierr);
        ierr = MPI_Gatherv(idx,N,MPI_DOUBLE,array,sizes,displs,MPI_DOUBLE,0,comm);CHKERRQ(ierr);
        ierr = PetscViewerSocketPutReal(viewer,Ntotal,1,array);CHKERRQ(ierr);
        ierr = PetscFree(sizes);CHKERRQ(ierr);
        ierr = PetscFree(displs);CHKERRQ(ierr);
        ierr = PetscFree(array);CHKERRQ(ierr);
      }
    } else {
      ierr = PetscViewerSocketPutReal(viewer,N,1,idx);CHKERRQ(ierr);
    }
  } else {
    SETERRQ(1,"Cannot handle that PetscViewer");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscScalarView" 
/*@C
    PetscScalarView - Prints an array of scalars; useful for debugging.

    Collective on PetscViewer

    Input Parameters:
+   N - number of scalars in array
.   idx - array of scalars
-   viewer - location to print array,  PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_STDOUT_SELF or 0

  Level: intermediate

.seealso: PetscIntView(), PetscRealView()
@*/
int PetscScalarView(int N,PetscScalar idx[],PetscViewer viewer)
{
  int        j,i,n = N/3,p = N % 3,ierr;
  PetscTruth isascii,issocket;
  MPI_Comm   comm;

  PetscFunctionBegin;
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_SELF;
  PetscValidHeader(viewer);
  PetscValidScalarPointer(idx);
  ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);

  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&isascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_SOCKET,&issocket);CHKERRQ(ierr);
  if (isascii) {
    for (i=0; i<n; i++) {
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"%2d:",3*i);CHKERRQ(ierr);
      for (j=0; j<3; j++) {
#if defined (PETSC_USE_COMPLEX)
        ierr = PetscViewerASCIISynchronizedPrintf(viewer," (%12.4e,%12.4e)",
                                 PetscRealPart(idx[i*3+j]),PetscImaginaryPart(idx[i*3+j]));CHKERRQ(ierr);
#else       
        ierr = PetscViewerASCIISynchronizedPrintf(viewer," %12.4e",idx[i*3+j]);CHKERRQ(ierr);
#endif
      }
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"\n");CHKERRQ(ierr);
    }
    if (p) {
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"%2d:",3*n);CHKERRQ(ierr);
      for (i=0; i<p; i++) { 
#if defined (PETSC_USE_COMPLEX)
        ierr = PetscViewerASCIISynchronizedPrintf(viewer," (%12.4e,%12.4e)",
                                 PetscRealPart(idx[n*3+i]),PetscImaginaryPart(idx[n*3+i]));CHKERRQ(ierr);
#else
        ierr = PetscViewerASCIISynchronizedPrintf(viewer," %12.4e",idx[3*n+i]);CHKERRQ(ierr);
#endif
      }
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  } else if (issocket) {
    int         *sizes,rank,size,Ntotal,*displs;
    PetscScalar *array;

    ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
    ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);

    if (size > 1) {
      if (rank) {
        ierr = MPI_Gather(&N,1,MPI_INT,0,0,MPI_INT,0,comm);CHKERRQ(ierr);
        ierr = MPI_Gatherv(idx,N,MPIU_SCALAR,0,0,0,MPIU_SCALAR,0,comm);CHKERRQ(ierr);
      } else {
	ierr   = PetscMalloc(size*sizeof(int),&sizes);CHKERRQ(ierr);
        ierr   = MPI_Gather(&N,1,MPI_INT,sizes,1,MPI_INT,0,comm);CHKERRQ(ierr);
        Ntotal = sizes[0]; 
	ierr   = PetscMalloc(size*sizeof(int),&displs);CHKERRQ(ierr);
        displs[0] = 0;
        for (i=1; i<size; i++) {
          Ntotal    += sizes[i];
          displs[i] =  displs[i-1] + sizes[i-1];
        }
	ierr = PetscMalloc(Ntotal*sizeof(PetscScalar),&array);CHKERRQ(ierr);
        ierr = MPI_Gatherv(idx,N,MPIU_SCALAR,array,sizes,displs,MPIU_SCALAR,0,comm);CHKERRQ(ierr);
        ierr = PetscViewerSocketPutScalar(viewer,Ntotal,1,array);CHKERRQ(ierr);
        ierr = PetscFree(sizes);CHKERRQ(ierr);
        ierr = PetscFree(displs);CHKERRQ(ierr);
        ierr = PetscFree(array);CHKERRQ(ierr);
      }
    } else {
      ierr = PetscViewerSocketPutScalar(viewer,N,1,idx);CHKERRQ(ierr);
    }
  } else {
    SETERRQ(1,"Cannot handle that PetscViewer");
  }
  PetscFunctionReturn(0);
}


/*MC
   SETERRQ - Macro that is called when an error has been detected, 

   Not Collective

   Synopsis:
   void SETERRQ(int errorcode,char *message)


   Input Parameters:
+  errorcode - nonzero error code, see the list of standard error codes in include/petscerror.h
-  message - error message

  Level: beginner

   Notes:
    Once the error handler is called the calling function is then returned from with the given error code.

    See SETERRQ1(), SETERRQ2(), SETERRQ3() for versions that take arguments


   Experienced users can set the error handler with PetscPushErrorHandler().

   Concepts: error^setting condition

.seealso: PetscTraceBackErrorHandler(), PetscPushErrorHandler(), PetscError(), CHKERRQ(), CHKMEMQ(), SETERRQ1(), SETERRQ2(), SETERRQ3()
M*/

/*MC
   SETERRQ1 - Macro that is called when an error has been detected, 

   Not Collective

   Synopsis:
   void SETERRQ1(int errorcode,char *formatmessage,arg)


   Input Parameters:
+  errorcode - nonzero error code, see the list of standard error codes in include/petscerror.h
.  message - error message in the printf format
-  arg - argument (for example an integer, string or double)

  Level: beginner

   Notes:
    Once the error handler is called the calling function is then returned from with the given error code.

   Experienced users can set the error handler with PetscPushErrorHandler().

   Concepts: error^setting condition

.seealso: PetscTraceBackErrorHandler(), PetscPushErrorHandler(), PetscError(), CHKERRQ(), CHKMEMQ(), SETERRQ(), SETERRQ2(), SETERRQ3()
M*/


/*MC
   SETERRQ2 - Macro that is called when an error has been detected, 

   Not Collective

   Synopsis:
   void SETERRQ2(int errorcode,char *formatmessage,arg1,arg2)


   Input Parameters:
+  errorcode - nonzero error code, see the list of standard error codes in include/petscerror.h
.  message - error message in the printf format
.  arg1 - argument (for example an integer, string or double)
-  arg2 - argument (for example an integer, string or double)

  Level: beginner

   Notes:
    Once the error handler is called the calling function is then returned from with the given error code.

   Experienced users can set the error handler with PetscPushErrorHandler().

   Concepts: error^setting condition

.seealso: PetscTraceBackErrorHandler(), PetscPushErrorHandler(), PetscError(), CHKERRQ(), CHKMEMQ(), SETERRQ1(), SETERRQ2(), SETERRQ3()
M*/

/*MC
   SETERRQ3 - Macro that is called when an error has been detected, 

   Not Collective

   Synopsis:
   void SETERRQ3(int errorcode,char *formatmessage,arg1,arg2,arg3)


   Input Parameters:
+  errorcode - nonzero error code, see the list of standard error codes in include/petscerror.h
.  message - error message in the printf format
.  arg1 - argument (for example an integer, string or double)
.  arg2 - argument (for example an integer, string or double)
-  arg3 - argument (for example an integer, string or double)

  Level: beginner

   Notes:
    Once the error handler is called the calling function is then returned from with the given error code.

   Experienced users can set the error handler with PetscPushErrorHandler().

   Concepts: error^setting condition

.seealso: PetscTraceBackErrorHandler(), PetscPushErrorHandler(), PetscError(), CHKERRQ(), CHKMEMQ(), SETERRQ1(), SETERRQ2(), SETERRQ2()
M*/


/*MC
   CHKERRQ - Checks error code, if non-zero it calls the error handler and then returns

   Not Collective

   Synopsis:
   void CHKERRQ(int errorcode)


   Input Parameters:
.  errorcode - nonzero error code, see the list of standard error codes in include/petscerror.h

  Level: beginner

   Notes:
    Once the error handler is called the calling function is then returned from with the given error code.

   Experienced users can set the error handler with PetscPushErrorHandler().

   Concepts: error^setting condition

.seealso: PetscTraceBackErrorHandler(), PetscPushErrorHandler(), PetscError(), SETERRQ(), CHKMEMQ(), SETERRQ1(), SETERRQ2(), SETERRQ2()
M*/

/*MC
   CHKMEMQ - Checks the memory for corruption, calls error handler if any is detected

   Not Collective

   Synopsis:
   void CHKMEMQ(void)

  Level: beginner

   Notes:
    Must run with the option -trdebug to enable this option

    Once the error handler is called the calling function is then returned from with the given error code.

    By defaults prints location where memory that is corrupted was allocated.

   Concepts: memory corruption

.seealso: PetscTraceBackErrorHandler(), PetscPushErrorHandler(), PetscError(), SETERRQ(), CHKMEMQ(), SETERRQ1(), SETERRQ2(), SETERRQ2(), 
          PetscTrValid()
M*/



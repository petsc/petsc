#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: err.c,v 1.96 1999/03/17 23:21:27 bsmith Exp bsmith $";
#endif
/*
      Code that allows one to set the error handlers
*/
#include "petsc.h"           /*I "petsc.h" I*/
#include <stdarg.h>
#if defined(HAVE_STDLIB_H)
#include <stdlib.h>
#endif

typedef struct _EH *EH;
struct _EH {
  int    cookie;
  int    (*handler)(int, char*,char*,char *,int,int,char*,void *);
  void   *ctx;
  EH     previous;
};

static EH eh = 0;

#undef __FUNC__  
#define __FUNC__ "PetscPushErrorHandler"
/*@C
   PetscPushErrorHandler - Sets a routine to be called on detection of errors.

   Not Collective

   Input Parameters:
+  handler - error handler routine
-  ctx - optional handler context that contains information needed by the handler (for 
         example file pointers for error messages etc.)

   Calling sequence of handler:
$    int handler(int line,char *func,char *file,char *dir,int n,int p,char *mess,void *ctx);

+  func - the function where the error occured (indicated by __FUNC__)
.  line - the line number of the error (indicated by __LINE__)
.  file - the file in which the error was detected (indicated by __FILE__)
.  dir - the directory of the file (indicated by __SDIR__)
.  n - the generic error number (see list defined in include/petscerror.h)
.  p - the specific error number
.  mess - an error text string, usually just printed to the screen
-  ctx - the error handler context

   Level: intermediate

   Fortran Note:
   This routine is not supported in Fortran.

.seealso: PetscPopErrorHandler(), PetscAttachDebuggerErrorHandler(), PetscAbortErrorHandler(), PetscTraceBackErrorHandler()

@*/
int PetscPushErrorHandler(int (*handler)(int,char *,char*,char*,int,int,char*,void*),void *ctx )
{
  EH neweh = (EH) PetscMalloc(sizeof(struct _EH)); CHKPTRQ(neweh);

  PetscFunctionBegin;
  if (eh) {neweh->previous = eh;} 
  else    {neweh->previous = 0;}
  neweh->handler = handler;
  neweh->ctx     = ctx;
  eh             = neweh;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscPopErrorHandler"
/*@C
   PetscPopErrorHandler - Removes the latest error handler that was 
   pushed with PetscPushErrorHandler().

   Not Collective

   Level: intermediate

   Fortran Note:
   This routine is not supported in Fortran.

.keywords: pop, error, handler

.seealso: PetscPushErrorHandler()
@*/
int PetscPopErrorHandler(void)
{
  EH tmp;

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

   Not Collective

   Input Parameters:
+  line - the line number of the error (indicated by __LINE__)
.  func - the function where the error occured (indicated by __FUNC__)
.  dir - the directory of file (indicated by __SDIR__)
.  file - the file in which the error was detected (indicated by __FILE__)
.  mess - an error text string, usually just printed to the screen
.  n - the generic error number
-  p - the specific error number

  Level: intermediate

   Notes:
   Most users need not directly use this routine and the error handlers, but
   can instead use the simplified interface SETERRQ, which has the calling 
   sequence
$     SETERRQ(n,p,mess)

   Experienced users can set the error handler with PetscPushErrorHandler().

.keywords: error, SETERRQ, SETERRA

.seealso: PetscTraceBackErrorHandler(), PetscPushErrorHandler()
@*/
int PetscError(int line,char *func,char* file,char *dir,int n,int p,char *mess,...)
{
  va_list     Argp;
  int         ierr;
  char        buf[1024],*lbuf = 0;

  PetscFunctionBegin;
  /* Compose the message evaluating the print format */
  if (mess) {
    va_start( Argp, mess);
    vsprintf(buf,mess,Argp);
    va_end( Argp );
    lbuf = buf;
  }

  if (!eh)     ierr = PetscTraceBackErrorHandler(line,func,file,dir,n,p,lbuf,0);
  else         ierr = (*eh->handler)(line,func,file,dir,n,p,lbuf,eh->ctx);
  PetscFunctionReturn(ierr);
}

/* -------------------------------------------------------------------------*/

#undef __FUNC__  
#define __FUNC__ "PetscIntView"
/*@C
    PetscIntView - Prints an array of integers; useful for debugging.

    Collective on Viewer

    Input Parameters:
+   N - number of integers in array
.   idx - array of integers
-   viewer - location to print array,  VIEWER_STDOUT_WORLD, VIEWER_STDOUT_SELF or 0

  Level: intermediate

.seealso: PetscDoubleView() 
@*/
int PetscIntView(int N,int idx[],Viewer viewer)
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
  if (PetscTypeCompare(vtype,ASCII_VIEWER)) {
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
  } else if (PetscTypeCompare(vtype,SOCKET_VIEWER)) {
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
        ierr = ViewerSocketPutInt_Private(viewer,Ntotal,array);CHKERRQ(ierr);
        PetscFree(sizes);
        PetscFree(displs);
        PetscFree(array);
      }
    } else {
      ierr = ViewerSocketPutInt_Private(viewer,N,idx);CHKERRQ(ierr);
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

    Collective on Viewer

    Input Parameters:
+   N - number of doubles in array
.   idx - array of doubles
-   viewer - location to print array,  VIEWER_STDOUT_WORLD, VIEWER_STDOUT_SELF or 0

  Level: intermediate

.seealso: PetscIntView() 
@*/
int PetscDoubleView(int N,double idx[],Viewer viewer)
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
  if (PetscTypeCompare(vtype,ASCII_VIEWER)) {
    ierr = ViewerASCIIGetPointer(viewer,&file);CHKERRQ(ierr);

    for ( i=0; i<n; i++ ) {
      PetscSynchronizedFPrintf(comm,file,"%2d:",5*i);
      for ( j=0; j<5; j++ ) {
         PetscSynchronizedFPrintf(comm,file," %12.4e",idx[i*5+j]);
      }
      PetscSynchronizedFPrintf(comm,file,"\n");
    }
    if (p) {
      PetscSynchronizedFPrintf(comm,file,"%2d:",5*n);
      for ( i=0; i<p; i++ ) { PetscSynchronizedFPrintf(comm,file," %12.4e",idx[5*n+i]);}
      PetscSynchronizedFPrintf(comm,file,"\n");
    }
    PetscSynchronizedFlush(comm);
  } else if (PetscTypeCompare(vtype,SOCKET_VIEWER)) {
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
        ierr = ViewerSocketPutDouble_Private(viewer,Ntotal,1,array);CHKERRQ(ierr);
        PetscFree(sizes);
        PetscFree(displs);
        PetscFree(array);
      }
    } else {
      ierr = ViewerSocketPutDouble_Private(viewer,N,1,idx);CHKERRQ(ierr);
    }
  } else {
    SETERRQ(1,1,"Cannot handle that viewer");
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscScalarView"
/*@C
    PetscScalarView - Prints an array of scalars; useful for debugging.

    Collective on Viewer

    Input Parameters:
+   N - number of scalars in array
.   idx - array of scalars
-   viewer - location to print array,  VIEWER_STDOUT_WORLD, VIEWER_STDOUT_SELF or 0

  Level: intermediate

.seealso: PetscIntView(), PetscDoubleView()
@*/
int PetscScalarView(int N,Scalar idx[],Viewer viewer)
{
  int        j,i,n = N/3, p = N % 3,ierr;
  MPI_Comm   comm;
  ViewerType vtype;
  FILE       *file;

  PetscFunctionBegin;
  if (!viewer) viewer = VIEWER_STDOUT_SELF;
  PetscValidHeader(viewer);
  PetscValidScalarPointer(idx);
  ierr = PetscObjectGetComm((PetscObject) viewer,&comm); CHKERRQ(ierr);

  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  if (PetscTypeCompare(vtype,ASCII_VIEWER)) {
    ierr = ViewerASCIIGetPointer(viewer,&file);CHKERRQ(ierr);
    for ( i=0; i<n; i++ ) {
      PetscSynchronizedFPrintf(comm,file,"%2d:",3*i);
      for ( j=0; j<3; j++ ) {
#if defined (USE_PETSC_COMPLEX)
        PetscSynchronizedFPrintf(comm,file," (%12.4e,%12.4e)",
                                 PetscReal(idx[i*3+j]),PetscImaginary(idx[i*3+j]));
#else       
         PetscSynchronizedFPrintf(comm,file," %12.4e",idx[i*3+j]);
#endif
      }
      PetscSynchronizedFPrintf(comm,file,"\n");
    }
    if (p) {
      PetscSynchronizedFPrintf(comm,file,"%2d:",3*n);
      for ( i=0; i<p; i++ ) { 
#if defined (USE_PETSC_COMPLEX)
        PetscSynchronizedFPrintf(comm,file," (%12.4e,%12.4e)",
                                 PetscReal(idx[i*3+j]),PetscImaginary(idx[i*3+j]));
#else
        PetscSynchronizedFPrintf(comm,file," %12.4e",idx[3*n+i]);
#endif
      }
      PetscSynchronizedFPrintf(comm,file,"\n");
    }
    PetscSynchronizedFlush(comm);
  } else if (PetscTypeCompare(vtype,SOCKET_VIEWER)) {
    int    *sizes,rank,size,Ntotal,*displs;
    Scalar *array;

    MPI_Comm_rank(comm,&rank);
    MPI_Comm_size(comm,&size);

    if (size > 1) {
      if (rank) {
        MPI_Gather(&N,1,MPI_INT,0,0,MPI_INT,0,comm);
        MPI_Gatherv(idx,N,MPIU_SCALAR,0,0,0,MPIU_SCALAR,0,comm);
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
        array  = (Scalar *) PetscMalloc(Ntotal*sizeof(Scalar));CHKPTRQ(array);
        MPI_Gatherv(idx,N,MPIU_SCALAR,array,sizes,displs,MPIU_SCALAR,0,comm);
        ierr = ViewerSocketPutScalar_Private(viewer,Ntotal,1,array);CHKERRQ(ierr);
        PetscFree(sizes);
        PetscFree(displs);
        PetscFree(array);
      }
    } else {
      ierr = ViewerSocketPutScalar_Private(viewer,N,1,idx);CHKERRQ(ierr);
    }
  } else {
    SETERRQ(1,1,"Cannot handle that viewer");
  }
  PetscFunctionReturn(0);
}












/*
      Code that allows one to set the error handlers
*/
#include <petsc/private/petscimpl.h>           /*I "petscsys.h" I*/
#include <petscviewer.h>

/* A table of Petsc source files containing calls to PETSCABORT. We assume this table will
   stay stable for a while. When things changed, we just need to add new files to the table.
 */
static const char* PetscAbortSourceFiles[] = {
  "Souce code of main",          /* 0 */
  "Not Found",                  /* 1, not found in petsc, but may be in users' code if they called PETSCABORT. */
  "sys/error/adebug.c",
  "src/sys/error/errstop.c",
  "sys/error/fp.c",
  "sys/error/signal.c",           /* 5 */
  "sys/ftn-custom/zutils.c",
  "sys/logging/utils/stagelog.c",
  "sys/mpiuni/mpitime.c",
  "sys/objects/init.c",
  "sys/objects/pinit.c",            /* 10 */
  "vec/vec/interface/dlregisvec.c",
  "vec/vec/utils/comb.c"
};

/* Find index of the soure file where a PETSCABORT was called. */
PetscErrorCode PetscAbortFindSourceFile_Private(const char* filepath, PetscInt *idx)
{
  PetscErrorCode  ierr;
  PetscInt        i,n = sizeof(PetscAbortSourceFiles)/sizeof(PetscAbortSourceFiles[0]);
  PetscBool       match;
  char            subpath[256];

  PetscFunctionBegin;
  PetscValidCharPointer(filepath,1);
  PetscValidIntPointer(idx,2);
  ierr = PetscStackView(stderr);CHKERRQ(ierr);
  *idx = 1;
  for (i=2; i<n; i++) {
    ierr = PetscFixFilename(PetscAbortSourceFiles[i],subpath);CHKERRQ(ierr);
    ierr = PetscStrendswith(filepath,subpath,&match);CHKERRQ(ierr);
    if (match) {*idx = i; break;}
  }
  PetscFunctionReturn(0);
}

typedef struct _EH *EH;
struct _EH {
  PetscErrorCode (*handler)(MPI_Comm,int,const char*,const char*,PetscErrorCode,PetscErrorType,const char*,void*);
  void           *ctx;
  EH             previous;
};

static EH eh = NULL;

/*@C
   PetscEmacsClientErrorHandler - Error handler that uses the emacsclient program to
    load the file where the error occurred. Then calls the "previous" error handler.

   Not Collective

   Input Parameters:
+  comm - communicator over which error occurred
.  line - the line number of the error (indicated by __LINE__)
.  file - the file in which the error was detected (indicated by __FILE__)
.  mess - an error text string, usually just printed to the screen
.  n - the generic error number
.  p - specific error number
-  ctx - error handler context

   Options Database Key:
.   -on_error_emacs <machinename> - will contact machinename to open the Emacs client there

   Level: developer

   Notes:
   You must put (server-start) in your .emacs file for the emacsclient software to work

   Developer Note:
   Since this is an error handler it cannot call CHKERRQ(); thus we just return if an error is detected.

.seealso: PetscError(), PetscPushErrorHandler(), PetscPopErrorHandler(), PetscAttachDebuggerErrorHandler(),
          PetscAbortErrorHandler(), PetscMPIAbortErrorHandler(), PetscTraceBackErrorHandler(), PetscReturnErrorHandler()
 @*/
PetscErrorCode  PetscEmacsClientErrorHandler(MPI_Comm comm,int line,const char *fun,const char *file,PetscErrorCode n,PetscErrorType p,const char *mess,void *ctx)
{
  PetscErrorCode ierr;
  char           command[PETSC_MAX_PATH_LEN];
  const char     *pdir;
  FILE           *fp;

  PetscFunctionBegin;
  ierr = PetscGetPetscDir(&pdir);if (ierr) PetscFunctionReturn(ierr);
  sprintf(command,"cd %s; emacsclient --no-wait +%d %s\n",pdir,line,file);
#if defined(PETSC_HAVE_POPEN)
  ierr = PetscPOpen(MPI_COMM_WORLD,(char*)ctx,command,"r",&fp);if (ierr) PetscFunctionReturn(ierr);
  ierr = PetscPClose(MPI_COMM_WORLD,fp);if (ierr) PetscFunctionReturn(ierr);
#else
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP_SYS,"Cannot run external programs on this machine");
#endif
  ierr = PetscPopErrorHandler();if (ierr) PetscFunctionReturn(ierr); /* remove this handler from the stack of handlers */
  if (!eh) {
    ierr = PetscTraceBackErrorHandler(comm,line,fun,file,n,p,mess,NULL);if (ierr) PetscFunctionReturn(ierr);
  } else {
    ierr = (*eh->handler)(comm,line,fun,file,n,p,mess,eh->ctx);if (ierr) PetscFunctionReturn(ierr);
  }
  PetscFunctionReturn(ierr);
}

/*@C
   PetscPushErrorHandler - Sets a routine to be called on detection of errors.

   Not Collective

   Input Parameters:
+  handler - error handler routine
-  ctx - optional handler context that contains information needed by the handler (for
         example file pointers for error messages etc.)

   Calling sequence of handler:
$    int handler(MPI_Comm comm,int line,char *func,char *file,PetscErrorCode n,int p,char *mess,void *ctx);

+  comm - communicator over which error occurred
.  line - the line number of the error (indicated by __LINE__)
.  file - the file in which the error was detected (indicated by __FILE__)
.  n - the generic error number (see list defined in include/petscerror.h)
.  p - PETSC_ERROR_INITIAL if error just detected, otherwise PETSC_ERROR_REPEAT
.  mess - an error text string, usually just printed to the screen
-  ctx - the error handler context

   Options Database Keys:
+   -on_error_attach_debugger <noxterm,gdb or dbx> - starts up the debugger if an error occurs
-   -on_error_abort - aborts the program if an error occurs

   Level: intermediate

   Notes:
   The currently available PETSc error handlers include PetscTraceBackErrorHandler(),
   PetscAttachDebuggerErrorHandler(), PetscAbortErrorHandler(), and PetscMPIAbortErrorHandler(), PetscReturnErrorHandler().

   Fortran Notes:
    You can only push one error handler from Fortran before poping it.

.seealso: PetscPopErrorHandler(), PetscAttachDebuggerErrorHandler(), PetscAbortErrorHandler(), PetscTraceBackErrorHandler(), PetscPushSignalHandler()

@*/
PetscErrorCode  PetscPushErrorHandler(PetscErrorCode (*handler)(MPI_Comm comm,int,const char*,const char*,PetscErrorCode,PetscErrorType,const char*,void*),void *ctx)
{
  EH             neweh;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNew(&neweh);CHKERRQ(ierr);
  if (eh) neweh->previous = eh;
  else    neweh->previous = NULL;
  neweh->handler = handler;
  neweh->ctx     = ctx;
  eh             = neweh;
  PetscFunctionReturn(0);
}

/*@
   PetscPopErrorHandler - Removes the latest error handler that was
   pushed with PetscPushErrorHandler().

   Not Collective

   Level: intermediate

.seealso: PetscPushErrorHandler()
@*/
PetscErrorCode  PetscPopErrorHandler(void)
{
  EH             tmp;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!eh) PetscFunctionReturn(0);
  tmp  = eh;
  eh   = eh->previous;
  ierr = PetscFree(tmp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscReturnErrorHandler - Error handler that causes a return without printing an error message.

   Not Collective

   Input Parameters:
+  comm - communicator over which error occurred
.  line - the line number of the error (indicated by __LINE__)
.  file - the file in which the error was detected (indicated by __FILE__)
.  mess - an error text string, usually just printed to the screen
.  n - the generic error number
.  p - specific error number
-  ctx - error handler context

   Level: developer

   Notes:
   Most users need not directly employ this routine and the other error
   handlers, but can instead use the simplified interface SETERRQ, which has
   the calling sequence
$     SETERRQ(comm,number,mess)

   PetscIgnoreErrorHandler() does the same thing as this function, but is deprecated, you should use this function.

   Use PetscPushErrorHandler() to set the desired error handler.

.seealso:  PetscPushErrorHandler(), PetscPopErrorHandler(), PetscError(), PetscAbortErrorHandler(), PetscMPIAbortErrorHandler(), PetscTraceBackErrorHandler(),
           PetscAttachDebuggerErrorHandler(), PetscEmacsClientErrorHandler()
 @*/
PetscErrorCode  PetscReturnErrorHandler(MPI_Comm comm,int line,const char *fun,const char *file,PetscErrorCode n,PetscErrorType p,const char *mess,void *ctx)
{
  return n;
}

static char PetscErrorBaseMessage[1024];
/*
       The numerical values for these are defined in include/petscerror.h; any changes
   there must also be made here
*/
static const char *PetscErrorStrings[] = {
  /*55 */ "Out of memory",
          "No support for this operation for this object type",
          "No support for this operation on this system",
  /*58 */ "Operation done in wrong order",
  /*59 */ "Signal received",
  /*60 */ "Nonconforming object sizes",
          "Argument aliasing not permitted",
          "Invalid argument",
  /*63 */ "Argument out of range",
          "Corrupt argument: https://petsc.org/release/faq/#valgrind",
          "Unable to open file",
          "Read from file failed",
          "Write to file failed",
          "Invalid pointer",
  /*69 */ "Arguments must have same type",
  /*70 */ "Attempt to use a pointer that does not point to a valid accessible location",
  /*71 */ "Zero pivot in LU factorization: https://petsc.org/release/faq/#zeropivot",
  /*72 */ "Floating point exception",
  /*73 */ "Object is in wrong state",
          "Corrupted Petsc object",
          "Arguments are incompatible",
          "Error in external library",
  /*77 */ "Petsc has generated inconsistent data",
          "Memory corruption: https://petsc.org/release/faq/#valgrind",
          "Unexpected data in file",
  /*80 */ "Arguments must have same communicators",
  /*81 */ "Zero pivot in Cholesky factorization: https://petsc.org/release/faq/#zeropivot",
          "  ",
          "  ",
          "Overflow in integer operation: https://petsc.org/release/faq/#64-bit-indices",
  /*85 */ "Null argument, when expecting valid pointer",
  /*86 */ "Unknown type. Check for miss-spelling or missing package: https://petsc.org/release/install/install/#external-packages",
  /*87 */ "MPI library at runtime is not compatible with MPI used at compile time",
  /*88 */ "Error in system call",
  /*89 */ "Object Type not set: https://petsc.org/release/faq/#object-type-not-set",
  /*90 */ "  ",
  /*   */ "  ",
  /*92 */ "See https://petsc.org/release/overview/linear_solve_table/ for possible LU and Cholesky solvers",
  /*93 */ "You cannot overwrite this option since that will conflict with other previously set options",
  /*94 */ "Example/application run with number of MPI ranks it does not support",
  /*95 */ "Missing or incorrect user input ",
  /*96 */ "GPU resources unavailable ",
  /*97 */ "GPU error ",
  /*98 */ "General MPI error "
};

/*@C
   PetscErrorMessage - returns the text string associated with a PETSc error code.

   Not Collective

   Input Parameter:
.   errnum - the error code

   Output Parameters:
+  text - the error message (NULL if not desired)
-  specific - the specific error message that was set with SETERRxxx() or PetscError().  (NULL if not desired)

   Level: developer

.seealso:  PetscPushErrorHandler(), PetscAttachDebuggerErrorHandler(), PetscError(), SETERRQ(), CHKERRQ()
          PetscAbortErrorHandler(), PetscTraceBackErrorHandler()
 @*/
PetscErrorCode  PetscErrorMessage(int errnum,const char *text[],char **specific)
{
  PetscFunctionBegin;
  if (text && errnum > PETSC_ERR_MIN_VALUE && errnum < PETSC_ERR_MAX_VALUE) *text = PetscErrorStrings[errnum-PETSC_ERR_MIN_VALUE-1];
  else if (text) *text = NULL;

  if (specific) *specific = PetscErrorBaseMessage;
  PetscFunctionReturn(0);
}

#if defined(PETSC_CLANGUAGE_CXX)
/* C++ exceptions are formally not allowed to propagate through extern "C" code. In practice, far too much software
 * would be broken if implementations did not handle it it some common cases. However, keep in mind
 *
 *   Rule 62. Don't allow exceptions to propagate across module boundaries
 *
 * in "C++ Coding Standards" by Sutter and Alexandrescu. (This accounts for part of the ongoing C++ binary interface
 * instability.) Having PETSc raise errors as C++ exceptions was probably misguided and should eventually be removed.
 *
 * Here is the problem: You have a C++ function call a PETSc function, and you would like to maintain the error message
 * and stack information from the PETSc error. You could make everyone write exactly this code in their C++, but that
 * seems crazy to me.
 */
#include <sstream>
#include <stdexcept>
static void PetscCxxErrorThrow()
{
  const char *str;
  if (eh && eh->ctx) {
    std::ostringstream *msg;
    msg = (std::ostringstream*) eh->ctx;
    str = msg->str().c_str();
  } else str = "Error detected in C PETSc";

  throw std::runtime_error(str);
}
#endif

/*@C
   PetscError - Routine that is called when an error has been detected, usually called through the macro SETERRQ(PETSC_COMM_SELF,).

  Collective on comm

   Input Parameters:
+  comm - communicator over which error occurred.  ALL ranks of this communicator MUST call this routine
.  line - the line number of the error (indicated by __LINE__)
.  func - the function name in which the error was detected
.  file - the file in which the error was detected (indicated by __FILE__)
.  n - the generic error number
.  p - PETSC_ERROR_INITIAL indicates the error was initially detected, PETSC_ERROR_REPEAT indicates this is a traceback from a previously detected error
-  mess - formatted message string - aka printf

  Options Database:
+  -error_output_stdout - output the error messages to stdout instead of the default stderr
-  -error_output_none - do not output the error messages

  Level: intermediate

   Notes:
   PETSc error handling is done with error return codes. A non-zero return indicates an error was detected. Errors are generally not something that the code
   can recover from. Note that numerical errors (potential divide by zero, for example) are not managed by the error return codes; they are managed via, for example,
   KSPGetConvergedReason() that indicates if the solve was successful or not. The option -ksp_error_if_not_converged, for example, turns numerical failures into
   hard errors managed via PetscError().

   PETSc provides a rich supply of error handlers, see the list below, and users can also provide their own error handlers.

   Most users need not directly use this routine and the error handlers, but
   can instead use the simplified interface SETERRQ, which has the calling
   sequence
$     SETERRQ(comm,n,mess)

   Fortran Note:
   This routine is used differently from Fortran
$    PetscError(MPI_Comm comm,PetscErrorCode n,PetscErrorType p,char *message)

   Set the error handler with PetscPushErrorHandler().

   Developer Note: Since this is called after an error condition it should not be calling any error handlers (currently it ignores any error codes)
   BUT this routine does call regular PETSc functions that may call error handlers, this is problematic and could be fixed by never calling other PETSc routines
   but this annoying.

.seealso: PetscErrorCode, PetscPushErrorHandler(), PetscPopErrorHandler(), PetscTraceBackErrorHandler(),  PetscAbortErrorHandler(), PetscMPIAbortErrorHandler(),
          PetscReturnErrorHandler(), PetscAttachDebuggerErrorHandler(), PetscEmacsClientErrorHandler(),
          SETERRQ(), CHKERRQ(), CHKMEMQ, SETERRQ1(), SETERRQ2(), PetscErrorMessage(), PETSCABORT()
@*/
PetscErrorCode PetscError(MPI_Comm comm,int line,const char *func,const char *file,PetscErrorCode n,PetscErrorType p,const char *mess,...)
{
  va_list        Argp;
  size_t         fullLength;
  char           buf[2048],*lbuf = NULL;
  PetscBool      ismain;
  PetscErrorCode ierr;

  if (!func) func = "User provided function";
  if (!file) file = "User file";
  if (comm == MPI_COMM_NULL) comm = PETSC_COMM_SELF;

  /* Compose the message evaluating the print format */
  if (mess) {
    va_start(Argp,mess);
    PetscVSNPrintf(buf,2048,mess,&fullLength,Argp);
    va_end(Argp);
    lbuf = buf;
    if (p == PETSC_ERROR_INITIAL) PetscStrncpy(PetscErrorBaseMessage,lbuf,1023);
  }

  if (p == PETSC_ERROR_INITIAL && n != PETSC_ERR_MEMC) PetscMallocValidate(__LINE__,PETSC_FUNCTION_NAME,__FILE__);

  if (!eh) ierr = PetscTraceBackErrorHandler(comm,line,func,file,n,p,lbuf,NULL);
  else ierr = (*eh->handler)(comm,line,func,file,n,p,lbuf,eh->ctx);
  PetscStackClearTop;

  /*
      If this is called from the main() routine we call MPI_Abort() instead of
    return to allow the parallel program to be properly shutdown.

    Does not call PETSCABORT() since that would provide the wrong source file and line number information
  */
  PetscStrncmp(func,"main",4,&ismain);
  if (ismain) {
    PetscMPIInt errcode;
    errcode = (PetscMPIInt)(0 + 0*line*1000 + ierr);
    if (petscwaitonerrorflg) { PetscSleep(1000); }
    MPI_Abort(MPI_COMM_WORLD,errcode);
  }

#if defined(PETSC_CLANGUAGE_CXX)
  if (p == PETSC_ERROR_IN_CXX) {
    PetscCxxErrorThrow();
  }
#endif
  return ierr;
}

/* -------------------------------------------------------------------------*/

/*@C
    PetscIntView - Prints an array of integers; useful for debugging.

    Collective on PetscViewer

    Input Parameters:
+   N - number of integers in array
.   idx - array of integers
-   viewer - location to print array,  PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_STDOUT_SELF or 0

  Level: intermediate

    Developer Notes:
    idx cannot be const because may be passed to binary viewer where byte swapping is done

.seealso: PetscRealView()
@*/
PetscErrorCode  PetscIntView(PetscInt N,const PetscInt idx[],PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank,size;
  PetscInt       j,i,n = N/20,p = N % 20;
  PetscBool      iascii,isbinary;
  MPI_Comm       comm;

  PetscFunctionBegin;
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_SELF;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,3);
  if (N) PetscValidIntPointer(idx,2);
  ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);

  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPushSynchronized(viewer);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      if (size > 1) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] %D:", rank, 20*i);CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"%D:",20*i);CHKERRQ(ierr);
      }
      for (j=0; j<20; j++) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer," %D",idx[i*20+j]);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"\n");CHKERRQ(ierr);
    }
    if (p) {
      if (size > 1) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] %D:",rank ,20*n);CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"%D:",20*n);CHKERRQ(ierr);
      }
      for (i=0; i<p; i++) { ierr = PetscViewerASCIISynchronizedPrintf(viewer," %D",idx[20*n+i]);CHKERRQ(ierr);}
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopSynchronized(viewer);CHKERRQ(ierr);
  } else if (isbinary) {
    PetscMPIInt *sizes,Ntotal,*displs,NN;
    PetscInt    *array;

    ierr = PetscMPIIntCast(N,&NN);CHKERRQ(ierr);

    if (size > 1) {
      if (rank) {
        ierr = MPI_Gather(&NN,1,MPI_INT,NULL,0,MPI_INT,0,comm);CHKERRMPI(ierr);
        ierr = MPI_Gatherv((void*)idx,NN,MPIU_INT,NULL,NULL,NULL,MPIU_INT,0,comm);CHKERRMPI(ierr);
      } else {
        ierr      = PetscMalloc1(size,&sizes);CHKERRQ(ierr);
        ierr      = MPI_Gather(&NN,1,MPI_INT,sizes,1,MPI_INT,0,comm);CHKERRMPI(ierr);
        Ntotal    = sizes[0];
        ierr      = PetscMalloc1(size,&displs);CHKERRQ(ierr);
        displs[0] = 0;
        for (i=1; i<size; i++) {
          Ntotal   += sizes[i];
          displs[i] =  displs[i-1] + sizes[i-1];
        }
        ierr = PetscMalloc1(Ntotal,&array);CHKERRQ(ierr);
        ierr = MPI_Gatherv((void*)idx,NN,MPIU_INT,array,sizes,displs,MPIU_INT,0,comm);CHKERRMPI(ierr);
        ierr = PetscViewerBinaryWrite(viewer,array,Ntotal,PETSC_INT);CHKERRQ(ierr);
        ierr = PetscFree(sizes);CHKERRQ(ierr);
        ierr = PetscFree(displs);CHKERRQ(ierr);
        ierr = PetscFree(array);CHKERRQ(ierr);
      }
    } else {
      ierr = PetscViewerBinaryWrite(viewer,idx,N,PETSC_INT);CHKERRQ(ierr);
    }
  } else {
    const char *tname;
    ierr = PetscObjectGetName((PetscObject)viewer,&tname);CHKERRQ(ierr);
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot handle that PetscViewer of type %s",tname);
  }
  PetscFunctionReturn(0);
}

/*@C
    PetscRealView - Prints an array of doubles; useful for debugging.

    Collective on PetscViewer

    Input Parameters:
+   N - number of PetscReal in array
.   idx - array of PetscReal
-   viewer - location to print array,  PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_STDOUT_SELF or 0

  Level: intermediate

    Developer Notes:
    idx cannot be const because may be passed to binary viewer where byte swapping is done

.seealso: PetscIntView()
@*/
PetscErrorCode  PetscRealView(PetscInt N,const PetscReal idx[],PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank,size;
  PetscInt       j,i,n = N/5,p = N % 5;
  PetscBool      iascii,isbinary;
  MPI_Comm       comm;

  PetscFunctionBegin;
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_SELF;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,3);
  PetscValidRealPointer(idx,2);
  ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);

  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
  if (iascii) {
    PetscInt tab;

    ierr = PetscViewerASCIIPushSynchronized(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIGetTab(viewer, &tab);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      ierr = PetscViewerASCIISetTab(viewer, tab);CHKERRQ(ierr);
      if (size > 1) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] %2d:",rank ,(int)5*i);CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"%2d:",(int)5*i);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIISetTab(viewer, 0);CHKERRQ(ierr);
      for (j=0; j<5; j++) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer," %12.4e",(double)idx[i*5+j]);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"\n");CHKERRQ(ierr);
    }
    if (p) {
      ierr = PetscViewerASCIISetTab(viewer, tab);CHKERRQ(ierr);
      if (size > 1) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] %2d:",rank ,(int)5*n);CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"%2d:",(int)5*n);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIISetTab(viewer, 0);CHKERRQ(ierr);
      for (i=0; i<p; i++) { PetscViewerASCIISynchronizedPrintf(viewer," %12.4e",(double)idx[5*n+i]);CHKERRQ(ierr);}
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIISetTab(viewer, tab);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopSynchronized(viewer);CHKERRQ(ierr);
  } else if (isbinary) {
    PetscMPIInt *sizes,*displs, Ntotal,NN;
    PetscReal   *array;

    ierr = PetscMPIIntCast(N,&NN);CHKERRQ(ierr);

    if (size > 1) {
      if (rank) {
        ierr = MPI_Gather(&NN,1,MPI_INT,NULL,0,MPI_INT,0,comm);CHKERRMPI(ierr);
        ierr = MPI_Gatherv((PetscReal*)idx,NN,MPIU_REAL,NULL,NULL,NULL,MPIU_REAL,0,comm);CHKERRMPI(ierr);
      } else {
        ierr      = PetscMalloc1(size,&sizes);CHKERRQ(ierr);
        ierr      = MPI_Gather(&NN,1,MPI_INT,sizes,1,MPI_INT,0,comm);CHKERRMPI(ierr);
        Ntotal    = sizes[0];
        ierr      = PetscMalloc1(size,&displs);CHKERRQ(ierr);
        displs[0] = 0;
        for (i=1; i<size; i++) {
          Ntotal   += sizes[i];
          displs[i] =  displs[i-1] + sizes[i-1];
        }
        ierr = PetscMalloc1(Ntotal,&array);CHKERRQ(ierr);
        ierr = MPI_Gatherv((PetscReal*)idx,NN,MPIU_REAL,array,sizes,displs,MPIU_REAL,0,comm);CHKERRMPI(ierr);
        ierr = PetscViewerBinaryWrite(viewer,array,Ntotal,PETSC_REAL);CHKERRQ(ierr);
        ierr = PetscFree(sizes);CHKERRQ(ierr);
        ierr = PetscFree(displs);CHKERRQ(ierr);
        ierr = PetscFree(array);CHKERRQ(ierr);
      }
    } else {
      ierr = PetscViewerBinaryWrite(viewer,(void*) idx,N,PETSC_REAL);CHKERRQ(ierr);
    }
  } else {
    const char *tname;
    ierr = PetscObjectGetName((PetscObject)viewer,&tname);CHKERRQ(ierr);
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot handle that PetscViewer of type %s",tname);
  }
  PetscFunctionReturn(0);
}

/*@C
    PetscScalarView - Prints an array of scalars; useful for debugging.

    Collective on PetscViewer

    Input Parameters:
+   N - number of scalars in array
.   idx - array of scalars
-   viewer - location to print array,  PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_STDOUT_SELF or 0

  Level: intermediate

    Developer Notes:
    idx cannot be const because may be passed to binary viewer where byte swapping is done

.seealso: PetscIntView(), PetscRealView()
@*/
PetscErrorCode  PetscScalarView(PetscInt N,const PetscScalar idx[],PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank,size;
  PetscInt       j,i,n = N/3,p = N % 3;
  PetscBool      iascii,isbinary;
  MPI_Comm       comm;

  PetscFunctionBegin;
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_SELF;
  PetscValidHeader(viewer,3);
  if (N) PetscValidScalarPointer(idx,2);
  ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);

  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPushSynchronized(viewer);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      if (size > 1) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] %2d:",rank ,3*i);CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"%2d:",3*i);CHKERRQ(ierr);
      }
      for (j=0; j<3; j++) {
#if defined(PETSC_USE_COMPLEX)
        ierr = PetscViewerASCIISynchronizedPrintf(viewer," (%12.4e,%12.4e)", (double)PetscRealPart(idx[i*3+j]),(double)PetscImaginaryPart(idx[i*3+j]));CHKERRQ(ierr);
#else
        ierr = PetscViewerASCIISynchronizedPrintf(viewer," %12.4e",(double)idx[i*3+j]);CHKERRQ(ierr);
#endif
      }
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"\n");CHKERRQ(ierr);
    }
    if (p) {
      if (size > 1) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] %2d:",rank ,3*n);CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"%2d:",3*n);CHKERRQ(ierr);
      }
      for (i=0; i<p; i++) {
#if defined(PETSC_USE_COMPLEX)
        ierr = PetscViewerASCIISynchronizedPrintf(viewer," (%12.4e,%12.4e)", (double)PetscRealPart(idx[n*3+i]),(double)PetscImaginaryPart(idx[n*3+i]));CHKERRQ(ierr);
#else
        ierr = PetscViewerASCIISynchronizedPrintf(viewer," %12.4e",(double)idx[3*n+i]);CHKERRQ(ierr);
#endif
      }
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopSynchronized(viewer);CHKERRQ(ierr);
  } else if (isbinary) {
    PetscMPIInt *sizes,Ntotal,*displs,NN;
    PetscScalar *array;

    ierr = PetscMPIIntCast(N,&NN);CHKERRQ(ierr);

    if (size > 1) {
      if (rank) {
        ierr = MPI_Gather(&NN,1,MPI_INT,NULL,0,MPI_INT,0,comm);CHKERRMPI(ierr);
        ierr = MPI_Gatherv((void*)idx,NN,MPIU_SCALAR,NULL,NULL,NULL,MPIU_SCALAR,0,comm);CHKERRMPI(ierr);
      } else {
        ierr      = PetscMalloc1(size,&sizes);CHKERRQ(ierr);
        ierr      = MPI_Gather(&NN,1,MPI_INT,sizes,1,MPI_INT,0,comm);CHKERRMPI(ierr);
        Ntotal    = sizes[0];
        ierr      = PetscMalloc1(size,&displs);CHKERRQ(ierr);
        displs[0] = 0;
        for (i=1; i<size; i++) {
          Ntotal   += sizes[i];
          displs[i] =  displs[i-1] + sizes[i-1];
        }
        ierr = PetscMalloc1(Ntotal,&array);CHKERRQ(ierr);
        ierr = MPI_Gatherv((void*)idx,NN,MPIU_SCALAR,array,sizes,displs,MPIU_SCALAR,0,comm);CHKERRMPI(ierr);
        ierr = PetscViewerBinaryWrite(viewer,array,Ntotal,PETSC_SCALAR);CHKERRQ(ierr);
        ierr = PetscFree(sizes);CHKERRQ(ierr);
        ierr = PetscFree(displs);CHKERRQ(ierr);
        ierr = PetscFree(array);CHKERRQ(ierr);
      }
    } else {
      ierr = PetscViewerBinaryWrite(viewer,(void*) idx,N,PETSC_SCALAR);CHKERRQ(ierr);
    }
  } else {
    const char *tname;
    ierr = PetscObjectGetName((PetscObject)viewer,&tname);CHKERRQ(ierr);
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot handle that PetscViewer of type %s",tname);
  }
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_CUDA)
#include <petscdevice.h>
PETSC_EXTERN const char* PetscCUBLASGetErrorName(cublasStatus_t status)
{
  switch(status) {
#if (CUDART_VERSION >= 8000) /* At least CUDA 8.0 of Sep. 2016 had these */
    case CUBLAS_STATUS_SUCCESS:          return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:  return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:     return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:    return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:    return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:    return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:   return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:    return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:    return "CUBLAS_STATUS_LICENSE_ERROR";
#endif
    default:                             return "unknown error";
  }
}
PETSC_EXTERN const char* PetscCUSolverGetErrorName(cusolverStatus_t status)
{
  switch(status) {
#if (CUDART_VERSION >= 8000) /* At least CUDA 8.0 of Sep. 2016 had these */
    case CUSOLVER_STATUS_SUCCESS:          return "CUSOLVER_STATUS_SUCCESS";
    case CUSOLVER_STATUS_NOT_INITIALIZED:  return "CUSOLVER_STATUS_NOT_INITIALIZED";
    case CUSOLVER_STATUS_INVALID_VALUE:    return "CUSOLVER_STATUS_INVALID_VALUE";
    case CUSOLVER_STATUS_ARCH_MISMATCH:    return "CUSOLVER_STATUS_ARCH_MISMATCH";
    case CUSOLVER_STATUS_INTERNAL_ERROR:   return "CUSOLVER_STATUS_INTERNAL_ERROR";
#if (CUDART_VERSION >= 9000) /* CUDA 9.0 had these defined on June 2021 */
    case CUSOLVER_STATUS_ALLOC_FAILED:     return "CUSOLVER_STATUS_ALLOC_FAILED";
    case CUSOLVER_STATUS_MAPPING_ERROR:    return "CUSOLVER_STATUS_MAPPING_ERROR";
    case CUSOLVER_STATUS_EXECUTION_FAILED: return "CUSOLVER_STATUS_EXECUTION_FAILED";
    case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED: return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    case CUSOLVER_STATUS_NOT_SUPPORTED :  return "CUSOLVER_STATUS_NOT_SUPPORTED ";
    case CUSOLVER_STATUS_ZERO_PIVOT:      return "CUSOLVER_STATUS_ZERO_PIVOT";
    case CUSOLVER_STATUS_INVALID_LICENSE: return "CUSOLVER_STATUS_INVALID_LICENSE";
#endif
#endif
    default:                             return "unknown error";
  }
}
PETSC_EXTERN const char* PetscCUFFTGetErrorName(cufftResult result)
{
 switch (result) {
 case CUFFT_SUCCESS:                   return "CUFFT_SUCCESS";
 case CUFFT_INVALID_PLAN:              return "CUFFT_INVALID_PLAN";
 case CUFFT_ALLOC_FAILED:              return "CUFFT_ALLOC_FAILED";
 case CUFFT_INVALID_TYPE:              return "CUFFT_INVALID_TYPE";
 case CUFFT_INVALID_VALUE:             return "CUFFT_INVALID_VALUE";
 case CUFFT_INTERNAL_ERROR:            return "CUFFT_INTERNAL_ERROR";
 case CUFFT_EXEC_FAILED:               return "CUFFT_EXEC_FAILED";
 case CUFFT_SETUP_FAILED:              return "CUFFT_SETUP_FAILED";
 case CUFFT_INVALID_SIZE:              return "CUFFT_INVALID_SIZE";
 case CUFFT_UNALIGNED_DATA:            return "CUFFT_UNALIGNED_DATA";
 case CUFFT_INCOMPLETE_PARAMETER_LIST: return "CUFFT_INCOMPLETE_PARAMETER_LIST";
 case CUFFT_INVALID_DEVICE:            return "CUFFT_INVALID_DEVICE";
 case CUFFT_PARSE_ERROR:               return "CUFFT_PARSE_ERROR";
 case CUFFT_NO_WORKSPACE:              return "CUFFT_NO_WORKSPACE";
 case CUFFT_NOT_IMPLEMENTED:           return "CUFFT_NOT_IMPLEMENTED";
 case CUFFT_LICENSE_ERROR:             return "CUFFT_LICENSE_ERROR";
 case CUFFT_NOT_SUPPORTED:             return "CUFFT_NOT_SUPPORTED";
 default:                              return "unknown error";
 }
}
#endif

#if defined(PETSC_HAVE_HIP)
#include <petscdevice.h>
PETSC_EXTERN const char* PetscHIPBLASGetErrorName(hipblasStatus_t status)
{
  switch(status) {
    case HIPBLAS_STATUS_SUCCESS:          return "HIPBLAS_STATUS_SUCCESS";
    case HIPBLAS_STATUS_NOT_INITIALIZED:  return "HIPBLAS_STATUS_NOT_INITIALIZED";
    case HIPBLAS_STATUS_ALLOC_FAILED:     return "HIPBLAS_STATUS_ALLOC_FAILED";
    case HIPBLAS_STATUS_INVALID_VALUE:    return "HIPBLAS_STATUS_INVALID_VALUE";
    case HIPBLAS_STATUS_ARCH_MISMATCH:    return "HIPBLAS_STATUS_ARCH_MISMATCH";
    case HIPBLAS_STATUS_MAPPING_ERROR:    return "HIPBLAS_STATUS_MAPPING_ERROR";
    case HIPBLAS_STATUS_EXECUTION_FAILED: return "HIPBLAS_STATUS_EXECUTION_FAILED";
    case HIPBLAS_STATUS_INTERNAL_ERROR:   return "HIPBLAS_STATUS_INTERNAL_ERROR";
    case HIPBLAS_STATUS_NOT_SUPPORTED:    return "HIPBLAS_STATUS_NOT_SUPPORTED";
    default:                              return "unknown error";
  }
}
#endif

/**
*   \page dev-petsc-kernel  The PETSc Kernel


%PETSc provides a variety of basic services for writing scalable, component
based libraries; these are referred to as the %PETSc kernel. The source
code for the kernel is in `src/sys`. It contains systematic support for
  - %PETSc types
  - error handling
  - memory management
  - profiling
  - object management
  - file IO
  - options database

Each of these is discussed in a section below.

\section dev-petsc-kernel-types PETSc Types
For maximum flexibility, the basic data types `int`, `double` etc are
generally not used in source code, rather it has:
  - PetscScalar
  - PetscInt
  - PetscMPIInt
  - PetscBLASInt
  - PetscBool
  - PetscBT  (bit storage of logical true and false)

PetscInt can be set using `./configure` to be either `int` (32 bit) or **long long**
(64 bit)
to allow indexing into very large arrays. PetscMPIInt are for integers passed to MPI
as counts etc, these are always `int` since that is what the MPI standard uses. Similarly
PetscBLASInt are for counts etc passed to BLAS and LAPACK routines. These are almost always
*int* unless one is using a special "64 bit integer" BLAS/LAPACK (this is available, for
example on Solaris systems).

In addition there a special types
  - PetscClassId
  - PetscErrorCode
  - PetscLogEvent

in fact, these are currently always `int` but their use clarifies the code.

\section dev-petsc-kernel-error-handling Implementation of Error Handling

%PETSc uses a *call error handler; then (depending on result) return
error code* model when problems are detected in the running code.

The public include file for error handling is
 [include/petscerror.h](http://www.mcs.anl.gov/petsc/petsc-dev/include/petscerror.h.html), the
source code for the %PETSc error handling is in
`src/sys/error/`.

\subsection dev-petsc-kernel-error-handling-simplified-interface Simplified Interface

The simplified C/C++ macro-based interface consists of the following three calls
  - SETERRQ(comm,error code,''Error message'');
  - CHKERRQ(ierr);

The macro SETERRQ() is given by
\code
return PetscError(comm,__LINE__,__FUNCT__,__FILE__,specific,''Error message'');
\endcode
It calls the error handler with the current function name and location: line number,
file and directory, plus an error codes and an error message.
The macro CHKERRQ() is defined by
\code
  if (ierr) SETERRQ(PETSC_COMM_SELF,ierr,(char *)0);
\endcode

In addition to SETERRQ() are the macros SETERRQ1(), SETERRQ2(), SETERRQ3()
and SETERRQ4() that allow one to include additional arguments that the message
string is formated. For example,
\code
  SETERRQ2(PETSC_ERR,''Iteration overflow: its %d norm %g'',its,norm);
\endcode
The reason for the numbered format is because CPP macros cannot handle variable number
of arguments.

\subsection dev-petsc-kernel-error-handling-error-handlers Error Handlers
The error handling function PetscError() calls the ``current'' error handler
with the code
\code
PetscErrorCode PetscError(MPI_Comm,int line,char *func,char* file,char *dir,PetscErrorCode n,int p,char *mess)
{ 
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!eh)     ierr = PetscTraceBackErrorHandler(line,func,file,dir,n,p,mess,0);
  else         ierr = (*eh-$>$handler)(line,func,file,dir,n,p,mess,eh-$>$ctx);
  PetscFunctionReturn(ierr);
}
\endcode
The variable `eh` is the current error handler context and is defined in
`src/sys/error/err.c` as
\code
typedef struct _EH* EH;
struct _EH {
  int    classid;
  int    (*handler)(int, char*,char*,char *,int,int,char*,void *);
  void   *ctx;
  EH     previous;
};
\endcode

One can set a new error handler with the command
\code
int PetscPushErrorHandler(int (*handler)(int,char *,char*,char*,PetscErrorCode,
                          int,char*,void*),void *ctx )
{
  EH neweh = (EH) PetscMalloc(sizeof(struct _EH)); CHKPTRQ(neweh);

  PetscFunctionBegin;
  if (eh) {neweh-$>$previous = eh;}
  else    {neweh-$>$previous = 0;}
  neweh-$>$handler = handler;
  neweh-$>$ctx     = ctx;
  eh             = neweh;
  PetscFunctionReturn(0);
}
\endcode
which maintains a linked list of error handlers. The most recent error handler is removed
via
\code
int PetscPopErrorHandler(void)
{
  EH tmp;

  PetscFunctionBegin;
  if (!eh) PetscFunctionReturn(0);
  tmp = eh;
  eh  = eh-$>$previous;
  PetscFree(tmp);
  PetscFunctionReturn(0);
}
\endcode

%PETSc provides several default error handlers
  - PetscTraceBackErrorHandler(),
  - PetscAbortErrorHandler(),
  - PetscReturnErrorHandler(),
  - PetscEmacsClientErrorHandler(),
  - PetscMPIAbortErrorHandler(), and
  - PetscAttachDebuggerErrorHandler().


\subsection dev-petsc-kernel-error-handling-error-codes Error Codes

The %PETSc error handler take a generic error code.
The generic error codes are defined in `include/petscerror.h`, the same generic
error code would be used many times in the libraries. For example the
generic error code `PETSC_ERR_MEM` is used whenever requested memory allocation
is not available.

\subsection dev-petsc-kernel-error-handling-error-messages Detailed Error Messages
In a modern parallel component oriented application code it does not make sense
to simply print error messages to the screen (more than likely there is no
"screen", for example with Windows applications).
%PETSc provides the replaceable function pointer
\code
   (*PetscErrorPrintf)(``Format'',...);
\endcode
that, by default prints to standard out. Thus error messages should not
be printed with printf() or fprintf() rather it should be printed with
(*PetscErrorPrintf)(). One can direct all error messages to
`stderr` with the command line options `-error_output_stderr`.


\section dev-petsc-kernel-profiling Implementation of Profiling

This section provides details about the implementation of event
logging and profiling within the %PETSc kernel.
The interface for profiling in %PETSc is contained in the file
`include/petsclog.h`. The source code for the profile logging
is in `src/sys/plog/`.

\subsection dev-petsc-kernel-profiling-create-destruction Profiling Object Create and Destruction

The creation of objects may be profiled with the command `PetscLogObjectCreate()`
\code
   PetscLogObjectCreate(PetscObject h);
\endcode
which logs the creation of any %PETSc object.
Just before an object is destroyed, it should be logged with
with `PetscLogObjectDestroy()`
\code
   PetscLogObjectDestroy(PetscObject h);
\endcode
These are called automatically by PetscHeaderCreate() and
PetscHeaderDestroy() which are used in creating all objects
inherited off the basic object. Thus these logging routines should
never be called directly.

If an object has a clearly defined parent object (for instance, when
a work vector is generated for use in a Krylov solver), this information
is logged with the command, `PetscLogObjectParent()`
\code
   PetscLogObjectParent(PetscObject parent,PetscObject child);
\endcode
It is also useful to log information about the state of an object, as can
be done with the command `PetscLogObjectState()`
\code
   PetscLogObjectState(PetscObject h,char *format,...);
\endcode

For example, for sparse matrices we usually log the matrix
dimensions and number of nonzeros.

\subsection dev-petsc-kernel-profiling-events Profiling Events

Events are logged using the pair `PetscLogEventBegin()`
\code
   PetscLogEventBegin(int event,PetscObject o1,PetscObject o2,PetscObject o3,PetscObject o4);
   PetscLogEventEnd(int event,PetscObject o1,PetscObject o2,PetscObject o3,PetscObject o4);
\endcode
This logging is usually done in the abstract
interface file for the operations, for example, `src/mat/src/matrix.c`.

\subsection dev-petsc-kernel-profiling-control Controling Profiling

Several routines that control the default profiling available in %PETSc are
\code
   PetscLogBegin();
   PetscLogAllBegin();
   PetscLogDump(char *filename);
   PetscLogView(FILE *fd);
\endcode
These routines are normally called by the PetscInitialize()
and PetscFinalize() routines when the option `-log`,
`-log_summary`, or `-log_all` is given.

\subsection dev-petsc-kernel-profiling-details Details of the Logging Design

*/

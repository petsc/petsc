/*
    Contains all error handling interfaces for PETSc.
*/
#if !defined(PETSCERROR_H)
#define PETSCERROR_H

#include <petscmacros.h>
#include <petscsystypes.h>

/* SUBMANSEC = Sys */

/*
     These are the generic error codes. These error codes are used
     many different places in the PETSc source code. The string versions are
     at src/sys/error/err.c any changes here must also be made there
     These are also define in src/sys/f90-mod/petscerror.h any CHANGES here
     must be also made there.

*/
#define PETSC_ERR_MIN_VALUE        54   /* should always be one less then the smallest value */

#define PETSC_ERR_MEM              55   /* unable to allocate requested memory */
#define PETSC_ERR_SUP              56   /* no support for requested operation */
#define PETSC_ERR_SUP_SYS          57   /* no support for requested operation on this computer system */
#define PETSC_ERR_ORDER            58   /* operation done in wrong order */
#define PETSC_ERR_SIG              59   /* signal received */
#define PETSC_ERR_FP               72   /* floating point exception */
#define PETSC_ERR_COR              74   /* corrupted PETSc object */
#define PETSC_ERR_LIB              76   /* error in library called by PETSc */
#define PETSC_ERR_PLIB             77   /* PETSc library generated inconsistent data */
#define PETSC_ERR_MEMC             78   /* memory corruption */
#define PETSC_ERR_CONV_FAILED      82   /* iterative method (KSP or SNES) failed */
#define PETSC_ERR_USER             83   /* user has not provided needed function */
#define PETSC_ERR_SYS              88   /* error in system call */
#define PETSC_ERR_POINTER          70   /* pointer does not point to valid address */
#define PETSC_ERR_MPI_LIB_INCOMP   87   /* MPI library at runtime is not compatible with MPI user compiled with */

#define PETSC_ERR_ARG_SIZ          60   /* nonconforming object sizes used in operation */
#define PETSC_ERR_ARG_IDN          61   /* two arguments not allowed to be the same */
#define PETSC_ERR_ARG_WRONG        62   /* wrong argument (but object probably ok) */
#define PETSC_ERR_ARG_CORRUPT      64   /* null or corrupted PETSc object as argument */
#define PETSC_ERR_ARG_OUTOFRANGE   63   /* input argument, out of range */
#define PETSC_ERR_ARG_BADPTR       68   /* invalid pointer argument */
#define PETSC_ERR_ARG_NOTSAMETYPE  69   /* two args must be same object type */
#define PETSC_ERR_ARG_NOTSAMECOMM  80   /* two args must be same communicators */
#define PETSC_ERR_ARG_WRONGSTATE   73   /* object in argument is in wrong state, e.g. unassembled mat */
#define PETSC_ERR_ARG_TYPENOTSET   89   /* the type of the object has not yet been set */
#define PETSC_ERR_ARG_INCOMP       75   /* two arguments are incompatible */
#define PETSC_ERR_ARG_NULL         85   /* argument is null that should not be */
#define PETSC_ERR_ARG_UNKNOWN_TYPE 86   /* type name doesn't match any registered type */

#define PETSC_ERR_FILE_OPEN        65   /* unable to open file */
#define PETSC_ERR_FILE_READ        66   /* unable to read from file */
#define PETSC_ERR_FILE_WRITE       67   /* unable to write to file */
#define PETSC_ERR_FILE_UNEXPECTED  79   /* unexpected data in file */

#define PETSC_ERR_MAT_LU_ZRPVT     71   /* detected a zero pivot during LU factorization */
#define PETSC_ERR_MAT_CH_ZRPVT     81   /* detected a zero pivot during Cholesky factorization */

#define PETSC_ERR_INT_OVERFLOW     84

#define PETSC_ERR_FLOP_COUNT       90
#define PETSC_ERR_NOT_CONVERGED    91  /* solver did not converge */
#define PETSC_ERR_MISSING_FACTOR   92  /* MatGetFactor() failed */
#define PETSC_ERR_OPT_OVERWRITE    93  /* attempted to over write options which should not be changed */
#define PETSC_ERR_WRONG_MPI_SIZE   94  /* example/application run with number of MPI ranks it does not support */
#define PETSC_ERR_USER_INPUT       95  /* missing or incorrect user input */
#define PETSC_ERR_GPU_RESOURCE     96  /* unable to load a GPU resource, for example cuBLAS */
#define PETSC_ERR_GPU              97  /* An error from a GPU call, this may be due to lack of resources on the GPU or a true error in the call */
#define PETSC_ERR_MPI              98  /* general MPI error */
#define PETSC_ERR_MAX_VALUE        99  /* this is always the one more than the largest error code */

#define SETERRQ1(...) PETSC_DEPRECATED_MACRO("GCC warning \"Use SETERRQ() (since version 3.17)\"") SETERRQ(__VA_ARGS__)
#define SETERRQ2(...) PETSC_DEPRECATED_MACRO("GCC warning \"Use SETERRQ() (since version 3.17)\"") SETERRQ(__VA_ARGS__)
#define SETERRQ3(...) PETSC_DEPRECATED_MACRO("GCC warning \"Use SETERRQ() (since version 3.17)\"") SETERRQ(__VA_ARGS__)
#define SETERRQ4(...) PETSC_DEPRECATED_MACRO("GCC warning \"Use SETERRQ() (since version 3.17)\"") SETERRQ(__VA_ARGS__)
#define SETERRQ5(...) PETSC_DEPRECATED_MACRO("GCC warning \"Use SETERRQ() (since version 3.17)\"") SETERRQ(__VA_ARGS__)
#define SETERRQ6(...) PETSC_DEPRECATED_MACRO("GCC warning \"Use SETERRQ() (since version 3.17)\"") SETERRQ(__VA_ARGS__)
#define SETERRQ7(...) PETSC_DEPRECATED_MACRO("GCC warning \"Use SETERRQ() (since version 3.17)\"") SETERRQ(__VA_ARGS__)
#define SETERRQ8(...) PETSC_DEPRECATED_MACRO("GCC warning \"Use SETERRQ() (since version 3.17)\"") SETERRQ(__VA_ARGS__)
#define SETERRQ9(...) PETSC_DEPRECATED_MACRO("GCC warning \"Use SETERRQ() (since version 3.17)\"") SETERRQ(__VA_ARGS__)

/*MC
   SETERRQ - Macro to be called when an error has been detected,

   Synopsis:
   #include <petscsys.h>
   PetscErrorCode SETERRQ(MPI_Comm comm,PetscErrorCode ierr,char *message,...)

   Collective

   Input Parameters:
+  comm - A communicator, use PETSC_COMM_SELF unless you know all ranks of another communicator will detect the error
.  ierr - nonzero error code, see the list of standard error codes in include/petscerror.h
-  message - error message

  Level: beginner

   Notes:
    Once the error handler is called the calling function is then returned from with the given error code.

    Experienced users can set the error handler with PetscPushErrorHandler().

   Fortran Notes:
      SETERRQ() may be called from Fortran subroutines but SETERRA() must be called from the
      Fortran main program.

.seealso: `PetscCheck()`, `PetscAssert()`, `PetscTraceBackErrorHandler()`, `PetscPushErrorHandler()`,
          `PetscError()`, `PetscCall()`, `CHKMEMQ`, `CHKERRA()`, `PetscCallMPI()`
M*/
#define SETERRQ(comm,ierr,...) return PetscError(comm,__LINE__,PETSC_FUNCTION_NAME,__FILE__,ierr,PETSC_ERROR_INITIAL,__VA_ARGS__)

/*
    Returned from PETSc functions that are called from MPI, such as related to attributes
      Do not confuse PETSC_MPI_ERROR_CODE and PETSC_ERR_MPI, the first is registered with MPI and returned to MPI as
      an error code, the latter is a regular PETSc error code passed within PETSc code indicating an error was detected in an MPI call.
*/
PETSC_EXTERN PetscMPIInt PETSC_MPI_ERROR_CLASS;
PETSC_EXTERN PetscMPIInt PETSC_MPI_ERROR_CODE;

/*MC
   SETERRMPI - Macro to be called when an error has been detected within an MPI callback function

   Synopsis:
   #include <petscsys.h>
   PetscErrorCode SETERRMPI(MPI_Comm comm,PetscErrorCode ierr,char *message,...)

   Collective

   Input Parameters:
+  comm - A communicator, use PETSC_COMM_SELF unless you know all ranks of another communicator will detect the error
.  ierr - nonzero error code, see the list of standard error codes in include/petscerror.h
-  message - error message

  Level: developer

   Notes:
    This macro is FOR USE IN MPI CALLBACK FUNCTIONS ONLY, such as those passed to MPI_Comm_create_keyval(). It always returns the error code PETSC_MPI_ERROR_CODE
    which is registered with MPI_Add_error_code() when PETSc is initialized.

.seealso: `SETERRQ()`, `PetscCall()`, `PetscCallMPI()`, `PetscTraceBackErrorHandler()`, `PetscPushErrorHandler()`, `PetscError()`, `CHKMEMQ`
M*/
#define SETERRMPI(comm,ierr,...) return (PetscError(comm,__LINE__,PETSC_FUNCTION_NAME,__FILE__,ierr,PETSC_ERROR_INITIAL,__VA_ARGS__),PETSC_MPI_ERROR_CODE)

/*MC
   SETERRA - Fortran-only macro that can be called when an error has been detected from the main program

   Synopsis:
   #include <petscsys.h>
   PetscErrorCode SETERRA(MPI_Comm comm,PetscErrorCode ierr,char *message)

   Collective

   Input Parameters:
+  comm - A communicator, so that the error can be collective
.  ierr - nonzero error code, see the list of standard error codes in include/petscerror.h
-  message - error message in the printf format

  Level: beginner

   Notes:
    This should only be used with Fortran. With C/C++, use SETERRQ().

   Fortran Notes:
      SETERRQ() may be called from Fortran subroutines but SETERRA() must be called from the
      Fortran main program.

.seealso: `SETERRQ()`, `SETERRABORT()`, `PetscCall()`, `CHKERRA()`, `PetscCallAbort()`
M*/

/*MC
   SETERRABORT - Macro that can be called when an error has been detected,

   Synopsis:
   #include <petscsys.h>
   PetscErrorCode SETERRABORT(MPI_Comm comm,PetscErrorCode ierr,char *message,...)

   Collective

   Input Parameters:
+  comm - A communicator, so that the error can be collective
.  ierr - nonzero error code, see the list of standard error codes in include/petscerror.h
-  message - error message in the printf format

  Level: beginner

   Notes:
    This function just calls MPI_Abort().

.seealso: `SETERRQ()`, `PetscTraceBackErrorHandler()`, `PetscPushErrorHandler()`, `PetscError()`, `PetscCall()`, `CHKMEMQ`
M*/
#define SETERRABORT(comm,ierr,...) do {                                                        \
    PetscError(comm,__LINE__,PETSC_FUNCTION_NAME,__FILE__,ierr,PETSC_ERROR_INITIAL,__VA_ARGS__); \
    MPI_Abort(comm,ierr);                                                                      \
  } while (0)

/*MC
  PetscCheck - Check that a particular condition is true

  Synopsis:
  #include <petscerror.h>
  void PetscCheck(bool cond, MPI_Comm comm, PetscErrorCode ierr, const char *message, ...)

  Collective

  Input Parameters:
+ cond    - The boolean condition
. comm    - The communicator on which the check can be collective on
. ierr    - A nonzero error code, see include/petscerror.h for the complete list
- message - Error message in printf format

  Notes:
  Enabled in both optimized and debug builds.

  Calls SETERRQ() if the assertion fails, so can only be called from functions returning a
  PetscErrorCode (or equivalent type after conversion).

  Level: beginner

.seealso: `PetscAssert()`, `SETERRQ()`, `PetscError()`, `PetscCall()`, `PetscCheckAbort()`
M*/
#define PetscCheck(cond,comm,ierr,...) do { if (PetscUnlikely(!(cond))) SETERRQ(comm,ierr,__VA_ARGS__); } while (0)

/*MC
  PetscCheckAbort - Check that a particular condition is true, otherwise prints error and aborts

  Synopsis:
  #include <petscerror.h>
  void PetscCheckAbort(bool cond, MPI_Comm comm, PetscErrorCode ierr, const char *message, ...)

  Collective

  Input Parameters:
+ cond    - The boolean condition
. comm    - The communicator on which the check can be collective on
. ierr    - A nonzero error code, see include/petscerror.h for the complete list
- message - Error message in printf format

  Notes:
  Enabled in both optimized and debug builds.

  Calls SETERRABORT() if the assertion fails, can be called from a function that does not return an
  error code. usually `PetscCheck()` should be used.

  Level: developer

.seealso: `PetscAssert()`, `SETERRQ()`, `PetscError()`, `PetscCall()`, `PetscCheck()`, `SETTERRABORT()`
M*/
#define PetscCheckAbort(cond,comm,ierr,...) if (PetscUnlikely(!(cond))) SETERRABORT(comm,ierr,__VA_ARGS__)

/*MC
  PetscAssert - Assert that a particular condition is true

  Synopsis:
  #include <petscerror.h>
  void PetscAssert(bool cond, MPI_Comm comm, PetscErrorCode ierr, const char *message, ...)

  Collective

  Input Parameters:
+ cond    - The boolean condition
. comm    - The communicator on which the check can be collective on
. ierr    - A nonzero error code, see include/petscerror.h for the complete list
- message - Error message in printf format

  Notes:
  Enabled only in debug builds. Note that any arguments to this macros are still visible to the
  compiler optimized builds (so must still contain valid code) but are guaranteed to not be
  executed.

  See PetscCheck() for usage and behaviour.

  Level: beginner

.seealso: `PetscCheck()`, `SETERRQ()`, `PetscError()`
M*/
#define PetscAssert(cond,comm,ierr,...) do { if (PetscUnlikelyDebug(!(cond))) SETERRQ(comm,ierr,__VA_ARGS__); } while (0)

/*MC
  PetscCall - Calls a PETSc function and then checks the resulting error code, if it is non-zero it calls the error
  handler and returns from the current function with the error code.

  Synopsis:
  #include <petscerror.h>
  void PetscCall(PetscFunction(args))

  Not Collective

  Input Parameter:
. PetscFunction - any PETSc function that returns an error code

  Notes:
  Once the error handler is called the calling function is then returned from with the given
  error code. Experienced users can set the error handler with PetscPushErrorHandler().

  PetscCall() cannot be used in functions returning a datatype not convertible to
  PetscErrorCode. For example, PetscCall() may not be used in functions returning void, use
  PetscCallVoid() in this case.

  Example Usage:
.vb
  PetscCall(PetscInitiailize(...)); // OK to call even when PETSc is not yet initialized!

  struct my_struct
  {
    void *data;
  } my_complex_type;

  struct my_struct bar(void)
  {
    PetscCall(foo(15)); // ERROR PetscErrorCode not convertible to struct my_struct!
  }

  PetscCall(bar()) // ERROR input not convertible to PetscErrorCode
.ve

  It is also possible to call this directly on a PetscErrorCode variable
.vb
  PetscCall(ierr);  // check if ierr is nonzero
.ve

  Should not be used to call callback functions provided by users, `PetscCallBack()` should be used in that situation.

  Fortran Notes:
    The Fortran function from which this is used must declare a variable PetscErrorCode ierr and ierr must be
    the final argument to the PetscFunction being called.

    In the main program and in Fortran subroutines that do not have ierr as the final return parameter one
    should use PetscCallA()

  Example Fortran Usage:
.vb
  PetscErrorCode ierr
  Vec v

  ...
  PetscCall(VecShift(v,1.0,ierr))
  PetscCallA(VecShift(v,1.0,ierr))
.ve

  Level: beginner

.seealso: `SETERRQ()`, `PetscCheck()`, `PetscAssert()`, `PetscTraceBackErrorHandler()`, `PetscCallMPI()`
          `PetscPushErrorHandler()`, `PetscError()`, `CHKMEMQ`, `CHKERRA()`, `CHKERRMPI()`, `PetscCallBack()`
M*/

/*MC
  PetscCallBack - Calls a user provided PETSc callback function and then checks the resulting error code, if it is non-zero it calls the error
  handler and returns from the current function with the error code.

  Synopsis:
  #include <petscerror.h>
  void PetscCallBack(const char *functionname,PetscFunction(args))

  Not Collective

  Input Parameters:
+ functionname - the name of the function being called, this can be a string with spaces that describes the meaning of the callback
- PetscFunction - any PETSc function that returns an error code

  Notes:
  Once the error handler is called the calling function is then returned from with the given
  error code. Experienced users can set the error handler with PetscPushErrorHandler().

  PetscCallBack() should only be called in PETSc when a call is being made to a user provided call-back routine.

  Example Usage:
.vb
  PetscCallBack("XXX callback to do something",a->callback(...));
.ve

  Level: developer

.seealso: `SETERRQ()`, `PetscCheck()`, `PetscAssert()`, `PetscTraceBackErrorHandler()`, `PetscCallMPI()`
          `PetscPushErrorHandler()`, `PetscError()`, `CHKMEMQ`, `CHKERRA()`, `CHKERRMPI()`, `PetscCall()`
M*/

#if defined(PETSC_CLANG_STATIC_ANALYZER)
void PetscCall(PetscErrorCode);
void PetscCallBack(const char *,PetscErrorCode);
void PetscCallVoid(PetscErrorCode);
#else
#define PetscCall(...) do {                                             \
    PetscErrorCode ierr_q_;                                             \
    PetscStackUpdateLine;                                               \
    ierr_q_ = __VA_ARGS__;                                              \
    if (PetscUnlikely(ierr_q_)) return PetscError(PETSC_COMM_SELF,__LINE__,PETSC_FUNCTION_NAME,__FILE__,ierr_q_,PETSC_ERROR_REPEAT," "); \
  } while (0)
#define PetscCallBack(function,...) do {                                \
    PetscErrorCode ierr_q_;                                             \
    PetscStackUpdateLine;                                               \
    PetscStackPushExternal(function);                                   \
    ierr_q_ = __VA_ARGS__;                                              \
    PetscStackPop;                                                      \
    if (PetscUnlikely(ierr_q_)) return PetscError(PETSC_COMM_SELF,__LINE__,PETSC_FUNCTION_NAME,__FILE__,ierr_q_,PETSC_ERROR_REPEAT," "); \
  } while (0)
#define PetscCallVoid(...) do {                                         \
    PetscErrorCode ierr_void_;                                          \
    PetscStackUpdateLine;                                               \
    ierr_void_ = __VA_ARGS__;                                           \
    if (PetscUnlikely(ierr_void_)) {(void)PetscError(PETSC_COMM_SELF,__LINE__,PETSC_FUNCTION_NAME,__FILE__,ierr_void_,PETSC_ERROR_REPEAT," "); return;} \
  } while (0)
#endif

/*MC
  CHKERRQ - Checks error code returned from PETSc function

  Synopsis:
  #include <petscsys.h>
  void CHKERRQ(PetscErrorCode ierr)

  Not Collective

  Input Parameters:
. ierr - nonzero error code

  Notes:
  Deprecated in favor of PetscCall(). This routine behaves identically to it.

  Level: deprecated

.seealso: `PetscCall()`
M*/
#define CHKERRQ(...) PetscCall(__VA_ARGS__)
#define CHKERRV(...) PetscCallVoid(__VA_ARGS__)

PETSC_EXTERN void PetscMPIErrorString(PetscMPIInt, char*);

/*MC
  PetscCallMPI - Checks error code returned from MPI calls, if non-zero it calls the error
  handler and then returns

  Synopsis:
  #include <petscerror.h>
  void PetscCallMPI(MPI_Function(args))

  Not Collective

  Input Parameters:
. MPI_Function - an MPI function that returns an MPI error code

  Notes:
  Always returns the error code PETSC_ERR_MPI; the MPI error code and string are embedded in
  the string error message. Do not use this to call any other routines (for example PETSc
  routines), it should only be used for direct MPI calls. Due to limitations of the
  preprocessor this can unfortunately not easily be enforced, so the user should take care to
  check this themselves.

  Example Usage:
.vb
  PetscCallMPI(MPI_Comm_size(...)); // OK, calling MPI function

  PetscCallMPI(PetscFunction(...)); // ERROR, use PetscCall() instead!
.ve

  Fortran Notes:
    The Fortran function from which this is used must declare a variable PetscErrorCode ierr and ierr must be
    the final argument to the MPI function being called.

    In the main program and in Fortran subroutines that do not have ierr as the final return parameter one
    should use PetscCallMPIA()

  Fortran Usage:
.vb
  PetscErrorCode ierr or integer ierr
  ...
  PetscCallMPI(MPI_Comm_size(...,ierr))
  PetscCallMPIA(MPI_Comm_size(...,ierr)) ! Will abort after calling error handler

  PetscCallMPI(MPI_Comm_size(...,eflag)) ! ERROR, final argument must be ierr
.ve

  Level: beginner

.seealso: `SETERRMPI()`, `PetscCall()`, `SETERRQ()`, `SETERRABORT()`, `PetscCallAbort()`,
          `PetscTraceBackErrorHandler()`, `PetscPushErrorHandler()`, `PetscError()`, `CHKMEMQ`
M*/
#if defined(PETSC_CLANG_STATIC_ANALYZER)
void PetscCallMPI(PetscMPIInt);
#else
#define PetscCallMPI(...) do {                                          \
    PetscMPIInt _7_errorcode;                                           \
    char _7_errorstring[2*MPI_MAX_ERROR_STRING];                        \
    PetscStackUpdateLine;                                               \
    PetscStackPushExternal("MPI function");                             \
    {_7_errorcode = __VA_ARGS__;}                                       \
    PetscStackPop;                                                      \
    if (PetscUnlikely(_7_errorcode)) {                                  \
      PetscMPIErrorString(_7_errorcode,(char*)_7_errorstring);          \
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MPI,"MPI error %d %s",(int)_7_errorcode,_7_errorstring); \
    }                                                                   \
  } while (0)
#endif

/*MC
  CHKERRMPI - Checks error code returned from MPI calls, if non-zero it calls the error
  handler and then returns

  Synopsis:
  #include <petscerror.h>
  void CHKERRMPI(PetscErrorCode ierr)

  Not Collective

  Input Parameter:
. ierr - nonzero error code, see the list of standard error codes in include/petscerror.h

  Notes:
  Deprecated in favor of PetscCallMPI(). This routine behaves identically to it.

  Level: deprecated

.seealso: `PetscCallMPI()`
M*/
#define CHKERRMPI(...) PetscCallMPI(__VA_ARGS__)

/*MC
  PetscCallAbort - Checks error code returned from PETSc function, if non-zero it aborts immediately

  Synopsis:
  #include <petscerror.h>
  void PetscCallAbort(MPI_Comm comm, PetscErrorCode ierr)

  Collective on comm

  Input Parameters:
+ comm - the MPI communicator on which to abort
- ierr - nonzero error code, see the list of standard error codes in include/petscerror.h

  Notes:
  This macro has identical type and usage semantics to PetscCall() with the important caveat
  that this macro does not return. Instead, if ierr is nonzero it calls the PETSc error handler
  and then immediately calls MPI_Abort(). It can therefore be used anywhere.

  As per MPI_Abort semantics the communicator passed must be valid, although there is currently
  no attempt made at handling any potential errors from MPI_Abort(). Note that while
  MPI_Abort() is required to terminate only those processes which reside on comm, it is often
  the case that MPI_Abort() terminates *all* processes.

  Example Usage:
.vb
  PetscErrorCode boom(void) { return PETSC_ERR_MEM; }

  void foo(void)
  {
    PetscCallAbort(PETSC_COMM_WORLD,boom()); // OK, does not return a type
  }

  double bar(void)
  {
    PetscCallAbort(PETSC_COMM_WORLD,boom()); // OK, does not return a type
  }

  PetscCallAbort(MPI_COMM_NULL,boom()); // ERROR, communicator should be valid

  struct baz
  {
    baz()
    {
      PetscCallAbort(PETSC_COMM_SELF,boom()); // OK
    }

    ~baz()
    {
      PetscCallAbort(PETSC_COMM_SELF,boom()); // OK (in fact the only way to handle PETSc errors)
    }
  };
.ve

  Level: intermediate

.seealso: `SETERRABORT()`, `PetscTraceBackErrorHandler()`, `PetscPushErrorHandler()`, `PetscError()`,
          `SETERRQ()`, `CHKMEMQ`, `PetscCallMPI()`
M*/
#if defined(PETSC_CLANG_STATIC_ANALYZER)
void PetscCallAbort(MPI_Comm,PetscErrorCode);
void PetscCallContinue(PetscErrorCode);
#else
#define PetscCallAbort(comm,...) do {                                                          \
    PetscErrorCode ierr_abort_ = __VA_ARGS__;                                                  \
    if (PetscUnlikely(ierr_abort_)) {                                                          \
      PetscError(PETSC_COMM_SELF,__LINE__,PETSC_FUNCTION_NAME,__FILE__,ierr_abort_,PETSC_ERROR_REPEAT," "); \
      MPI_Abort(comm,ierr_abort_);                                                             \
    }                                                                                          \
  } while (0)
#define PetscCallContinue(...)   do {                                                          \
    PetscErrorCode ierr_continue_ = __VA_ARGS__;                                               \
    if (PetscUnlikely(ierr_continue_)) PetscError(PETSC_COMM_SELF,__LINE__,PETSC_FUNCTION_NAME,__FILE__,ierr_continue_,PETSC_ERROR_REPEAT," "); \
  } while (0)
#endif

/*MC
  CHKERRABORT - Checks error code returned from PETSc function. If non-zero it aborts immediately.

  Synopsis:
  #include <petscerror.h>
  void CHKERRABORT(MPI_Comm comm, PetscErrorCode ierr)

  Not Collective

  Input Parameters:
+ comm - the MPI communicator
- ierr - nonzero error code, see the list of standard error codes in include/petscerror.h

  Notes:
  Deprecated in favor of PetscCallAbort(). This routine behaves identically to it.

  Level: deprecated

.seealso: `PetscCallAbort()`
M*/
#define CHKERRABORT(comm,...) PetscCallAbort(comm,__VA_ARGS__)
#define CHKERRCONTINUE(...)   PetscCallContinue(__VA_ARGS__)

/*MC
   CHKERRA - Fortran-only replacement for PetscCall in the main program, which aborts immediately

   Synopsis:
   #include <petscsys.h>
   PetscErrorCode CHKERRA(PetscErrorCode ierr)

   Not Collective

   Input Parameters:
.  ierr - nonzero error code, see the list of standard error codes in include/petscerror.h

  Level: beginner

   Notes:
      This should only be used with Fortran. With C/C++, use PetscCall() in normal usage,
      or PetscCallAbort() if wanting to abort immediately on error.

   Fortran Notes:
      PetscCall() may be called from Fortran subroutines but CHKERRA() must be called from the
      Fortran main program.

.seealso: `PetscCall()`, `PetscCallAbort()`, `SETERRA()`, `SETERRQ()`, `SETERRABORT()`
M*/

PETSC_EXTERN PetscBool petscwaitonerrorflg;
PETSC_EXTERN PetscBool petscindebugger;

/*MC
   PETSCABORT - Call MPI_Abort with an informative error code

   Synopsis:
   #include <petscsys.h>
   PETSCABORT(MPI_Comm comm, PetscErrorCode ierr)

   Collective

   Input Parameters:
+  comm - A communicator, so that the error can be collective
-  ierr - nonzero error code, see the list of standard error codes in include/petscerror.h

   Level: advanced

   Notes:
   If the option -start_in_debugger was used then this calls abort() to stop the program in the debugger.

   if PetscCIEnabledPortableErrorOutput it strives to exit cleanly without call `MPI_Abort()`

 M*/
#define PETSCABORT(comm,...) do {                                                              \
    if (petscwaitonerrorflg) PetscSleep(1000);                                                 \
    if (petscindebugger) abort();                                                              \
    else {                                                                                     \
      PetscErrorCode ierr_petsc_abort_ = __VA_ARGS__;                                          \
      PetscMPIInt    size;                                                                     \
      MPI_Comm_size(comm,&size);                                                               \
      if (PetscCIEnabledPortableErrorOutput && size == PetscGlobalSize && ierr_petsc_abort_ != PETSC_ERR_SIG) { \
        MPI_Finalize(); exit(0);                                                               \
      } else if (PetscCIEnabledPortableErrorOutput && PetscGlobalSize == 1) {                  \
        exit(0);                                                        \
      } else {                                                                                 \
        MPI_Abort(comm,(PetscMPIInt)ierr_petsc_abort_);                 \
      }                                                                                        \
    }                                                                                          \
  } while (0)

#ifdef PETSC_CLANGUAGE_CXX
/*MC
  PetscCallThrow - Checks error code, if non-zero it calls the C++ error handler which throws
  an exception

  Synopsis:
  #include <petscerror.h>
  void PetscCallThrow(PetscErrorCode ierr)

  Not Collective

  Input Parameter:
. ierr - nonzero error code, see the list of standard error codes in include/petscerror.h

  Notes:
  Requires PETSc to be configured with clanguage = c++. Throws a std::runtime_error() on error.

  Once the error handler throws the exception you can use PetscCallVoid() which returns without
  an error code (bad idea since the error is ignored) or PetscCallAbort() to have MPI_Abort()
  called immediately.

  Level: beginner

.seealso: `SETERRQ()`, `PetscCall()`, `SETERRABORT()`, `PetscCallAbort()`, `PetscTraceBackErrorHandler()`,
          `PetscPushErrorHandler()`, `PetscError()`, `CHKMEMQ`
M*/
#define PetscCallThrow(...) do {                                                                    \
    PetscErrorCode ierr_cxx_ = __VA_ARGS__;                                                    \
    if (PetscUnlikely(ierr_cxx_)) PetscError(PETSC_COMM_SELF,__LINE__,PETSC_FUNCTION_NAME,__FILE__,ierr_cxx_,PETSC_ERROR_IN_CXX,PETSC_NULLPTR); \
  } while (0)

/*MC
  CHKERRXX - Checks error code, if non-zero it calls the C++ error handler which throws an exception

  Synopsis:
  #include <petscerror.h>
  void CHKERRXX(PetscErrorCode ierr)

  Not Collective

  Input Parameter:
. ierr - nonzero error code, see the list of standard error codes in include/petscerror.h

  Notes:
  Deprecated in favor of PetscCallThrow(). This routine behaves identically to it.

  Level: deprecated

.seealso: `PetscCallThrow()`
M*/
#define CHKERRXX(...) PetscCallThrow(__VA_ARGS__)
#endif

/*MC
  PetscCallCXX - Checks C++ function calls and if they throw an exception, catch it and then
  return a PETSc error code

  Synopsis:
  #include <petscerror.h>
  void PetscCallCXX(expr) noexcept;

  Not Collective

  Input Parameter:
. expr - An arbitrary expression

  Notes:
  PetscCallCXX(expr) is a macro replacement for
.vb
  try {
    expr;
  } catch (const std::exception& e) {
    return ConvertToPetscErrorCode(e);
  }
.ve
  Due to the fact that it catches any (reasonable) exception, it is essentially noexcept.

  Example Usage:
.vb
  void foo(void) { throw std::runtime_error("error"); }

  void bar()
  {
    PetscCallCXX(foo()); // ERROR bar() does not return PetscErrorCode
  }

  PetscErrorCode baz()
  {
    PetscCallCXX(foo()); // OK

    PetscCallCXX(
      bar();
      foo(); // OK mutliple statements allowed
    );
  }

  struct bop
  {
    bop()
    {
      PetscCallCXX(foo()); // ERROR returns PetscErrorCode, cannot be used in constructors
    }
  };

  // ERROR contains do-while, cannot be used as function-try block
  PetscErrorCode qux() PetscCallCXX(
    bar();
    baz();
    foo();
    return 0;
  )
.ve

  Level: beginner

.seealso: `PetscCallThrow()`, `SETERRQ()`, `PetscCall()`, `SETERRABORT()`, `PetscCallAbort()`,
          `PetscTraceBackErrorHandler()`, `PetscPushErrorHandler()`, `PetscError()`, `CHKMEMQ`
M*/
#define PetscCallCXX(...) do {                                  \
    PetscStackUpdateLine;                                       \
    try {                                                       \
      __VA_ARGS__;                                              \
    } catch (const std::exception& e) {                         \
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"%s",e.what());     \
    }                                                           \
  } while (0)

/*MC
  CHKERRCXX - Checks C++ function calls and if they throw an exception, catch it and then
  return a PETSc error code

  Synopsis:
  #include <petscerror.h>
  void CHKERRCXX(func) noexcept;

  Not Collective

  Input Parameter:
. func - C++ function calls

  Notes:
  Deprecated in favor of PetscCallCXX(). This routine behaves identically to it.

  Level: deprecated

.seealso: `PetscCallCXX()`
M*/
#define CHKERRCXX(...) PetscCallCXX(__VA_ARGS__)

/*MC
   CHKMEMQ - Checks the memory for corruption, calls error handler if any is detected

   Synopsis:
   #include <petscsys.h>
   CHKMEMQ;

   Not Collective

  Level: beginner

   Notes:
    We highly recommend using Valgrind https://petsc.org/release/faq/#valgrind or for NVIDIA CUDA systems
    https://docs.nvidia.com/cuda/cuda-memcheck/index.html for finding memory problems. The ``CHKMEMQ`` macro is useful on systems that
    do not have valgrind, but is not as good as valgrind or cuda-memcheck.

    Must run with the option -malloc_debug (-malloc_test in debug mode; or if PetscMallocSetDebug() called) to enable this option

    Once the error handler is called the calling function is then returned from with the given error code.

    By defaults prints location where memory that is corrupted was allocated.

    Use CHKMEMA for functions that return void

.seealso: `PetscTraceBackErrorHandler()`, `PetscPushErrorHandler()`, `PetscError()`, `SETERRQ()`, `PetscMallocValidate()`
M*/
#if defined(PETSC_CLANG_STATIC_ANALYZER)
#define CHKMEMQ
#define CHKMEMA
#else
#define CHKMEMQ PetscCall(PetscMallocValidate(__LINE__,PETSC_FUNCTION_NAME,__FILE__));
#define CHKMEMA PetscMallocValidate(__LINE__,PETSC_FUNCTION_NAME,__FILE__)
#endif

/*E
  PetscErrorType - passed to the PETSc error handling routines indicating if this is the first or a later call to the error handlers

  Level: advanced

  PETSC_ERROR_IN_CXX indicates the error was detected in C++ and an exception should be generated

  Developer Notes:
    This is currently used to decide when to print the detailed information about the run in PetscTraceBackErrorHandler()

.seealso: `PetscError()`, `SETERRXX()`
E*/
typedef enum {PETSC_ERROR_INITIAL=0,PETSC_ERROR_REPEAT=1,PETSC_ERROR_IN_CXX = 2} PetscErrorType;

#if defined(__clang_analyzer__)
__attribute__((analyzer_noreturn))
#endif
PETSC_EXTERN PetscErrorCode PetscError(MPI_Comm,int,const char*,const char*,PetscErrorCode,PetscErrorType,const char*,...) PETSC_ATTRIBUTE_COLD PETSC_ATTRIBUTE_FORMAT(7,8);

PETSC_EXTERN PetscErrorCode PetscErrorPrintfInitialize(void);
PETSC_EXTERN PetscErrorCode PetscErrorMessage(int,const char*[],char **);
PETSC_EXTERN PetscErrorCode PetscTraceBackErrorHandler(MPI_Comm,int,const char*,const char*,PetscErrorCode,PetscErrorType,const char*,void*) PETSC_ATTRIBUTE_COLD;
PETSC_EXTERN PetscErrorCode PetscIgnoreErrorHandler(MPI_Comm,int,const char*,const char*,PetscErrorCode,PetscErrorType,const char*,void*) PETSC_ATTRIBUTE_COLD;
PETSC_EXTERN PetscErrorCode PetscEmacsClientErrorHandler(MPI_Comm,int,const char*,const char*,PetscErrorCode,PetscErrorType,const char*,void*) PETSC_ATTRIBUTE_COLD;
PETSC_EXTERN PetscErrorCode PetscMPIAbortErrorHandler(MPI_Comm,int,const char*,const char*,PetscErrorCode,PetscErrorType,const char*,void*) PETSC_ATTRIBUTE_COLD;
PETSC_EXTERN PetscErrorCode PetscAbortErrorHandler(MPI_Comm,int,const char*,const char*,PetscErrorCode,PetscErrorType,const char*,void*) PETSC_ATTRIBUTE_COLD;
PETSC_EXTERN PetscErrorCode PetscAttachDebuggerErrorHandler(MPI_Comm,int,const char*,const char*,PetscErrorCode,PetscErrorType,const char*,void*) PETSC_ATTRIBUTE_COLD;
PETSC_EXTERN PetscErrorCode PetscReturnErrorHandler(MPI_Comm,int,const char*,const char*,PetscErrorCode,PetscErrorType,const char*,void*) PETSC_ATTRIBUTE_COLD;
PETSC_EXTERN PetscErrorCode PetscPushErrorHandler(PetscErrorCode (*handler)(MPI_Comm,int,const char*,const char*,PetscErrorCode,PetscErrorType,const char*,void*),void*);
PETSC_EXTERN PetscErrorCode PetscPopErrorHandler(void);
PETSC_EXTERN PetscErrorCode PetscSignalHandlerDefault(int,void*);
PETSC_EXTERN PetscErrorCode PetscPushSignalHandler(PetscErrorCode (*)(int,void *),void*);
PETSC_EXTERN PetscErrorCode PetscPopSignalHandler(void);
PETSC_EXTERN PetscErrorCode PetscCheckPointerSetIntensity(PetscInt);
PETSC_EXTERN void PetscSignalSegvCheckPointerOrMpi(void);
PETSC_DEPRECATED_FUNCTION("Use PetscSignalSegvCheckPointerOrMpi() (since version 3.13)") static inline void PetscSignalSegvCheckPointer(void) {PetscSignalSegvCheckPointerOrMpi();}

/*MC
    PetscErrorPrintf - Prints error messages.

   Synopsis:
    #include <petscsys.h>
     PetscErrorCode (*PetscErrorPrintf)(const char format[],...);

    Not Collective

    Input Parameter:
.   format - the usual printf() format string

   Options Database Keys:
+    -error_output_stdout - cause error messages to be printed to stdout instead of the  (default) stderr
-    -error_output_none - to turn off all printing of error messages (does not change the way the error is handled.)

   Notes:
    Use
$     PetscErrorPrintf = PetscErrorPrintfNone; to turn off all printing of error messages (does not change the way the
$                        error is handled.) and
$     PetscErrorPrintf = PetscErrorPrintfDefault; to turn it back on or you can use your own function

          Use
     PETSC_STDERR = FILE* obtained from a file open etc. to have stderr printed to the file.
     PETSC_STDOUT = FILE* obtained from a file open etc. to have stdout printed to the file.

          Use
      PetscPushErrorHandler() to provide your own error handler that determines what kind of messages to print

   Level: developer

    Fortran Note:
    This routine is not supported in Fortran.

.seealso: `PetscFPrintf()`, `PetscSynchronizedPrintf()`, `PetscHelpPrintf()`, `PetscPrintf()`, `PetscPushErrorHandler()`, `PetscVFPrintf()`, `PetscHelpPrintf()`
M*/
PETSC_EXTERN PetscErrorCode (*PetscErrorPrintf)(const char[],...) PETSC_ATTRIBUTE_FORMAT(1,2);

/*E
     PetscFPTrap - types of floating point exceptions that may be trapped

     Currently only PETSC_FP_TRAP_OFF and PETSC_FP_TRAP_ON are handled. All others are treated as PETSC_FP_TRAP_ON.

     Level: intermediate

.seealso: `PetscSetFPTrap()`, `PetscPushFPTrap()`
 E*/
typedef enum {PETSC_FP_TRAP_OFF=0, PETSC_FP_TRAP_INDIV=1, PETSC_FP_TRAP_FLTOPERR=2, PETSC_FP_TRAP_FLTOVF=4, PETSC_FP_TRAP_FLTUND=8, PETSC_FP_TRAP_FLTDIV=16, PETSC_FP_TRAP_FLTINEX=32} PetscFPTrap;
#define  PETSC_FP_TRAP_ON  (PETSC_FP_TRAP_INDIV | PETSC_FP_TRAP_FLTOPERR | PETSC_FP_TRAP_FLTOVF | PETSC_FP_TRAP_FLTDIV | PETSC_FP_TRAP_FLTINEX)
PETSC_EXTERN PetscErrorCode PetscSetFPTrap(PetscFPTrap);
PETSC_EXTERN PetscErrorCode PetscFPTrapPush(PetscFPTrap);
PETSC_EXTERN PetscErrorCode PetscFPTrapPop(void);
PETSC_EXTERN PetscErrorCode PetscDetermineInitialFPTrap(void);

/*
      Allows the code to build a stack frame as it runs
*/

#if defined(PETSC_USE_DEBUG)
#define PETSCSTACKSIZE 64
typedef struct  {
  const char *function[PETSCSTACKSIZE];
  const char *file[PETSCSTACKSIZE];
        int  line[PETSCSTACKSIZE];
        int  petscroutine[PETSCSTACKSIZE]; /* 0 external called from petsc, 1 petsc functions, 2 petsc user functions */
        int  currentsize;
        int  hotdepth;
        PetscBool  check; /* option to check for correct Push/Pop semantics, true for default petscstack but not other stacks */
} PetscStack;
PETSC_EXTERN PetscStack petscstack;
#else
typedef struct {
  char Silence_empty_struct_has_size_0_in_C_size_1_in_Cpp;
} PetscStack;
#endif

#if defined(PETSC_SERIALIZE_FUNCTIONS)
#include <petsc/private/petscfptimpl.h>
/*
   Registers the current function into the global function pointer to function name table

   Have to fix this to handle errors but cannot return error since used in PETSC_VIEWER_DRAW_() etc
*/
#define PetscRegister__FUNCT__() do { \
  static PetscBool __chked = PETSC_FALSE; \
  if (!__chked) {\
  void *ptr; PetscDLSym(NULL,PETSC_FUNCTION_NAME,&ptr);\
  __chked = PETSC_TRUE;\
  }} while (0)
#else
#define PetscRegister__FUNCT__()
#endif

#if defined(PETSC_CLANG_STATIC_ANALYZER)
#define PetscStackPushNoCheck(funct,petsc_routine,hot)
#define PetscStackUpdateLine
#define PetscStackPushExternal(funct)
#define PetscStackPopNoCheck
#define PetscStackClearTop
#define PetscFunctionBegin
#define PetscFunctionBeginUser
#define PetscFunctionBeginHot
#define PetscFunctionReturn(a)    return a
#define PetscFunctionReturnVoid() return
#define PetscStackPop
#define PetscStackPush(f)
#elif defined(PETSC_USE_DEBUG)

#define PetscStackPush_Private(stack__,file__,func__,line__,petsc_routine__,hot__) do { \
    if (stack__.currentsize < PETSCSTACKSIZE) {                                         \
      stack__.function[stack__.currentsize]     = func__;                               \
      if (petsc_routine__) {                                                            \
        stack__.file[stack__.currentsize]         = file__;                             \
        stack__.line[stack__.currentsize]         = line__;                             \
      } else {                                                                          \
        stack__.file[stack__.currentsize]         = PETSC_NULLPTR;                      \
        stack__.line[stack__.currentsize]         = 0;                                  \
      }                                                                                 \
      stack__.petscroutine[stack__.currentsize] = petsc_routine__;                      \
    }                                                                                   \
    ++stack__.currentsize;                                                              \
    stack__.hotdepth += (hot__ || stack__.hotdepth);                                    \
  } while (0)

/* uses PetscCheckAbort() because may be used in a function that does not return an error code */
#define PetscStackPop_Private(stack__,func__) do {                                             \
    PetscCheckAbort(!stack__.check || stack__.currentsize > 0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Invalid stack size %d, pop %s %s:%d.\n",stack__.currentsize,func__,__FILE__,__LINE__);\
    if (--stack__.currentsize < PETSCSTACKSIZE) {\
      PetscCheckAbort(!stack__.check || stack__.petscroutine[stack__.currentsize] != 1 || stack__.function[stack__.currentsize] == (const char*)(func__),PETSC_COMM_SELF,PETSC_ERR_PLIB,"Invalid stack: push from %s %s:%d. Pop from %s %s:%d.\n",stack__.function[stack__.currentsize],stack__.file[stack__.currentsize],stack__.line[stack__.currentsize],func__,__FILE__,__LINE__); \
      stack__.function[stack__.currentsize]     = PETSC_NULLPTR;                             \
      stack__.file[stack__.currentsize]         = PETSC_NULLPTR;                             \
      stack__.line[stack__.currentsize]         = 0;                                         \
      stack__.petscroutine[stack__.currentsize] = 0;                                         \
    }                                                                                        \
    stack__.hotdepth = PetscMax(stack__.hotdepth-1,0);                                       \
  } while (0)

/*MC
   PetscStackPushNoCheck - Pushes a new function name and line number onto the PETSc default stack that tracks where the running program is
   currently in the source code.

   Not Collective

   Synopsis:
   #include <petscsys.h>
   void PetscStackPushNoCheck(char *funct,int petsc_routine,PetscBool hot);

   Input Parameters:
+  funct - the function name
.  petsc_routine - 2 user function, 1 PETSc function, 0 some other function
-  hot - indicates that the function may be called often so expensive error checking should be turned off inside the function

   Level: developer

   Notes:
   In debug mode PETSc maintains a stack of the current function calls that can be used to help to quickly see where a problem has
   occurred, for example, when a signal is received. It is recommended to use the debugger if extensive information is needed to
   help debug the problem.

   This version does not check the memory corruption (an expensive operation), use `PetscStackPush()` to check the memory.

   Use `PetscStackPushExternal()` for a function call that is about to be made to a non-PETSc or user function (such as BLAS etc).

   The default stack is a global variable called `petscstack`.

   In general the line number is at the beginning of the function (where `PetscFunctionBegin` is called) so it is not accurate

.seealso: `PetscAttachDebugger()`, `PetscStackCopy()`, `PetscStackView()`, `PetscStackPopNoCheck()`, `PetscCall()`, `PetscFunctionBegin()`,
          `PetscFunctionReturn()`, `PetscFunctionBeginHot()`, `PetscFunctionBeginUser()`, `PetscStackPush()`, `PetscStackPop`,
          `PetscStackPushExternal()`
M*/
#define PetscStackPushNoCheck(funct,petsc_routine,hot) do {                             \
    PetscStackSAWsTakeAccess();                                                         \
    PetscStackPush_Private(petscstack,__FILE__,funct,__LINE__,petsc_routine,hot);       \
    PetscStackSAWsGrantAccess();                                                        \
  } while (0)

/*MC
   PetscStackUpdateLine - in a function that has a PetscFunctionBegin or PetscFunctionBeginUser updates the stack line number to the
   current line number.

   Not Collective

   Synopsis:
   #include <petscsys.h>
   void PetscStackUpdateLine

   Level: developer

   Notes:
   In debug mode PETSc maintains a stack of the current function calls that can be used to help to quickly see where a problem has
   occurred, for example, when a signal is received. It is recommended to use the debugger if extensive information is needed to
   help debug the problem.

   The default stack is a global variable called petscstack.

   This is used by `PetscCall()` and is otherwise not like to be needed

.seealso: `PetscAttachDebugger()`, `PetscStackCopy()`, `PetscStackView()`, `PetscStackPushNoCheck()`, `PetscStackPop`, `PetscCall()`
M*/
#define PetscStackUpdateLine                                         \
  if (petscstack.currentsize > 0 && petscstack.function[petscstack.currentsize-1] == PETSC_FUNCTION_NAME){ \
    petscstack.line[petscstack.currentsize-1] = __LINE__;              \
  }

/*MC
   PetscStackPushExternal - Pushes a new function name onto the PETSc default stack that tracks where the running program is
   currently in the source code. Does not include the filename or line number since this is called by the calling routine
   for non-PETSc or user functions.

   Not Collective

   Synopsis:
   #include <petscsys.h>
   void PetscStackPushExternal(char *funct);

   Input Parameters:
.  funct - the function name

   Level: developer

   Notes:
   In debug mode PETSc maintains a stack of the current function calls that can be used to help to quickly see where a problem has
   occurred, for example, when a signal is received. It is recommended to use the debugger if extensive information is needed to
   help debug the problem.

   The default stack is a global variable called `petscstack`.

   This is to be used when calling an external package function such as a BLAS function.

   This also updates the stack line number for the current stack function.

.seealso: `PetscAttachDebugger()`, `PetscStackCopy()`, `PetscStackView()`, `PetscStackPopNoCheck()`, `PetscCall()`, `PetscFunctionBegin()`,
          `PetscFunctionReturn()`, `PetscFunctionBeginHot()`, `PetscFunctionBeginUser()`, `PetscStackPushNoCheck()`, `PetscStackPop`
M*/
#define PetscStackPushExternal(funct) do {PetscStackUpdateLine; PetscStackPushNoCheck(funct,0,PETSC_TRUE);} while (0);

/*MC
   PetscStackPopNoCheck - Pops a function name from the PETSc default stack that tracks where the running program is
   currently in the source code.

   Not Collective

   Synopsis:
   #include <petscsys.h>
   void PetscStackPopNoCheck(char *funct);

   Input Parameter:
.   funct - the function name

   Level: developer

   Notes:
   In debug mode PETSc maintains a stack of the current function calls that can be used to help to quickly see where a problem has
   occurred, for example, when a signal is received. It is recommended to use the debugger if extensive information is needed to
   help debug the problem.

   The default stack is a global variable called petscstack.

   Developer Note:
   `PetscStackPopNoCheck()` takes a function argument while  `PetscStackPop` does not, this difference is likely just historical.

.seealso: `PetscAttachDebugger()`, `PetscStackCopy()`, `PetscStackView()`, `PetscStackPushNoCheck()`, `PetscStackPop`
M*/
#define PetscStackPopNoCheck(funct)                    do {     \
    PetscStackSAWsTakeAccess();                                 \
    PetscStackPop_Private(petscstack,funct);                    \
    PetscStackSAWsGrantAccess();                                \
  } while (0)

#define PetscStackClearTop                             do {             \
    PetscStackSAWsTakeAccess();                                         \
    if (petscstack.currentsize > 0 &&                                   \
        --petscstack.currentsize < PETSCSTACKSIZE) {                    \
      petscstack.function[petscstack.currentsize]     = PETSC_NULLPTR;  \
      petscstack.file[petscstack.currentsize]         = PETSC_NULLPTR;  \
      petscstack.line[petscstack.currentsize]         = 0;              \
      petscstack.petscroutine[petscstack.currentsize] = 0;              \
    }                                                                   \
    petscstack.hotdepth = PetscMax(petscstack.hotdepth-1,0);            \
    PetscStackSAWsGrantAccess();                                        \
  } while (0)

/*MC
   PetscFunctionBegin - First executable line of each PETSc function,  used for error handling. Final
      line of PETSc functions should be PetscFunctionReturn(0);

   Synopsis:
   #include <petscsys.h>
   void PetscFunctionBegin;

   Not Collective

   Usage:
.vb
     int something;

     PetscFunctionBegin;
.ve

   Notes:
     Use PetscFunctionBeginUser for application codes.

     Not available in Fortran

   Level: developer

.seealso: `PetscFunctionReturn()`, `PetscFunctionBeginHot()`, `PetscFunctionBeginUser()`, `PetscStackPushNoCheck()`

M*/
#define PetscFunctionBegin do {                               \
    PetscStackPushNoCheck(PETSC_FUNCTION_NAME,1,PETSC_FALSE); \
    PetscRegister__FUNCT__();                                 \
  } while (0)

/*MC
   PetscFunctionBeginHot - Substitute for PetscFunctionBegin to be used in functions that are called in
   performance-critical circumstances.  Use of this function allows for lighter profiling by default.

   Synopsis:
   #include <petscsys.h>
   void PetscFunctionBeginHot;

   Not Collective

   Usage:
.vb
     int something;

     PetscFunctionBeginHot;
.ve

   Notes:
     Not available in Fortran

   Level: developer

.seealso: `PetscFunctionBegin`, `PetscFunctionReturn()`, `PetscStackPushNoCheck()`

M*/
#define PetscFunctionBeginHot do {                           \
    PetscStackPushNoCheck(PETSC_FUNCTION_NAME,1,PETSC_TRUE); \
    PetscRegister__FUNCT__();                                \
  } while (0)

/*MC
   PetscFunctionBeginUser - First executable line of user provided routines

   Synopsis:
   #include <petscsys.h>
   void PetscFunctionBeginUser;

   Not Collective

   Usage:
.vb
     int something;

     PetscFunctionBeginUser;
.ve

   Notes:
      Functions that incorporate this must call `PetscFunctionReturn()` instead of return except for main().

      May be used before `PetscInitialize()`

      Not available in Fortran

      This is identical to `PetscFunctionBegin` except it labels the routine as a user
      routine instead of as a PETSc library routine.

   Level: intermediate

.seealso: `PetscFunctionReturn()`, `PetscFunctionBegin`, `PetscFunctionBeginHot`, `PetscStackPushNoCheck()`

M*/
#define PetscFunctionBeginUser do {                           \
    PetscStackPushNoCheck(PETSC_FUNCTION_NAME,2,PETSC_FALSE); \
    PetscRegister__FUNCT__();                                 \
  } while (0)

/*MC
   PetscStackPush - Pushes a new function name and line number onto the PETSc default stack that tracks where the running program is
   currently in the source code and verifies the memory is not corrupted.

   Not Collective

   Synopsis:
   #include <petscsys.h>
   void PetscStackPush(char *funct)

   Input Parameter:
.  funct - the function name

   Level: developer

   Notes:
   In debug mode PETSc maintains a stack of the current function calls that can be used to help to quickly see where a problem has
   occurred, for example, when a signal is received. It is recommended to use the debugger if extensive information is needed to
   help debug the problem.

   The default stack is a global variable called petscstack.

   In general the line number is at the beginning of the function (where `PetscFunctionBegin` is called) so it is not accurate

.seealso: `PetscAttachDebugger()`, `PetscStackCopy()`, `PetscStackView()`, `PetscStackPopNoCheck()`, `PetscCall()`, `PetscFunctionBegin()`,
          `PetscFunctionReturn()`, `PetscFunctionBeginHot()`, `PetscFunctionBeginUser()`, `PetscStackPushNoCheck()`, `PetscStackPop`
M*/
#define PetscStackPush(n)       do {        \
    PetscStackPushNoCheck(n,0,PETSC_FALSE); \
    CHKMEMQ;                                \
  } while (0)

/*MC
   PetscStackPop - Pops a function name from the PETSc default stack that tracks where the running program is
   currently in the source code and verifies the memory is not corrupted.

   Not Collective

   Synopsis:
   #include <petscsys.h>
   void PetscStackPop

   Level: developer

   Notes:
   In debug mode PETSc maintains a stack of the current function calls that can be used to help to quickly see where a problem has
   occurred, for example, when a signal is received. It is recommended to use the debugger if extensive information is needed to
   help debug the problem.

   The default stack is a global variable called petscstack.

.seealso: `PetscAttachDebugger()`, `PetscStackCopy()`, `PetscStackView()`, `PetscStackPushNoCheck()`, `PetscStackPopNoCheck()`, `PetscStackPush()`
M*/
#define PetscStackPop           do {       \
      CHKMEMQ;                                   \
      PetscStackPopNoCheck(PETSC_FUNCTION_NAME); \
    } while (0)

/*MC
   PetscFunctionReturn - Last executable line of each PETSc function
        used for error handling. Replaces return()

   Synopsis:
   #include <petscsys.h>
   void PetscFunctionReturn(0);

   Not Collective

   Usage:
.vb
    ....
     PetscFunctionReturn(0);
   }
.ve

   Notes:
     Not available in Fortran

   Level: developer

.seealso: `PetscFunctionBegin()`, `PetscStackPopNoCheck()`

M*/
#define PetscFunctionReturn(a)    do {          \
    PetscStackPopNoCheck(PETSC_FUNCTION_NAME);  \
    return a;                                   \
  } while (0)

#define PetscFunctionReturnVoid() do {          \
    PetscStackPopNoCheck(PETSC_FUNCTION_NAME);  \
    return;                                     \
  } while (0)
#else /* PETSC_USE_DEBUG */
#define PetscStackPushNoCheck(funct,petsc_routine,hot)
#define PetscStackUpdateLine
#define PetscStackPushExternal(funct)
#define PetscStackPopNoCheck
#define PetscStackClearTop
#define PetscFunctionBegin
#define PetscFunctionBeginUser
#define PetscFunctionBeginHot
#define PetscFunctionReturn(a)    return a
#define PetscFunctionReturnVoid() return
#define PetscStackPop             CHKMEMQ
#define PetscStackPush(f)         CHKMEMQ
#endif /* PETSC_USE_DEBUG */

#if defined(PETSC_CLANG_STATIC_ANALYZER)
#define PetscStackCallExternalVoid(name,routine)
#define PetscCallExternal(func,...)
#else
/*MC
    PetscStackCallExternalVoid - Calls an external library routine or user function after pushing the name of the routine on the stack.

   Input Parameters:
+   name - string that gives the name of the function being called
-   routine - actual call to the routine, for example, functionname(a,b)

   Level: developer

   Note:
   Often one should use `PetscCallExternal()` instead. This routine is intended for external library routines that DO NOT return error codes

   In debug mode this also checks the memory for corruption at the end of the function call.

   Certain external packages, such as BLAS/LAPACK may have their own macros for managing the call, error checking, etc.

   Developer Note:
   This is so that when a user or external library routine results in a crash or corrupts memory, they get blamed instead of PETSc.

.seealso: `PetscCall()`, `PetscStackPushNoCheck()`, `PetscStackPush()`, `PetscCallExternal()`, `PetscCallBLAS()`
@*/
#define PetscStackCallExternalVoid(name,routine) do { PetscStackPush(name);routine;PetscStackPop; } while (0)

/*MC
    PetscCallExternal - Calls an external library routine that returns an error code after pushing the name of the routine on the stack.

   Input Parameters:
+   func-  name of the routine
-   args - arguments to the routine

   Level: developer

   Notes:
   This is intended for external package routines that return error codes. Use `PetscStackCallExternalVoid()` for those that do not.

   In debug mode this also checks the memory for corruption at the end of the function call.

   Developer Note:
   This is so that when an external packge routine results in a crash or corrupts memory, they get blamed instead of PETSc.

.seealso: `PetscCall()`, `PetscStackPushNoCheck()`, `PetscStackPush()`, `PetscStackCallExternalVoid()`
M*/
#define PetscCallExternal(func,...) do {                                                  \
    PetscStackPush(PetscStringize(func));                                                      \
    PetscErrorCode __ierr = func(__VA_ARGS__);                                                 \
    PetscStackPop;                                                                             \
    PetscCheck(!__ierr,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in %s(): error code %d",PetscStringize(func),__ierr); \
  } while (0)
#endif /* PETSC_CLANG_STATIC_ANALYZER */

#endif

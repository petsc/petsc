/*
    Contains all error handling interfaces for PETSc.
*/
#ifndef PETSCERROR_H
#define PETSCERROR_H

#include <petscmacros.h>
#include <petscsystypes.h>

#if defined(__cplusplus)
  #include <exception> // std::exception
#endif

/* SUBMANSEC = Sys */

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
+  comm - A communicator, use `PETSC_COMM_SELF` unless you know all ranks of another communicator will detect the error
.  ierr - nonzero error code, see the list of standard error codes in include/petscerror.h
-  message - error message

  Level: beginner

   Notes:
    This is rarely needed, one should use `PetscCheck()` and `PetscCall()` and friends to automatically handle error conditions.
    Once the error handler is called the calling function is then returned from with the given error code.

    Experienced users can set the error handler with `PetscPushErrorHandler()`.

   Fortran Note:
   `SETERRQ()` may be called from Fortran subroutines but `SETERRA()` must be called from the
   Fortran main program.

.seealso: `PetscCheck()`, `PetscAssert()`, `PetscTraceBackErrorHandler()`, `PetscPushErrorHandler()`,
          `PetscError()`, `PetscCall()`, `CHKMEMQ`, `CHKERRA()`, `PetscCallMPI()`
M*/
#define SETERRQ(comm, ierr, ...) \
  do { \
    PetscErrorCode ierr_seterrq_petsc_ = PetscError(comm, __LINE__, PETSC_FUNCTION_NAME, __FILE__, ierr, PETSC_ERROR_INITIAL, __VA_ARGS__); \
    return ierr_seterrq_petsc_ ? ierr_seterrq_petsc_ : PETSC_ERR_RETURN; \
  } while (0)

/*
    Returned from PETSc functions that are called from MPI, such as related to attributes
      Do not confuse PETSC_MPI_ERROR_CODE and PETSC_ERR_MPI, the first is registered with MPI and returned to MPI as
      an error code, the latter is a regular PETSc error code passed within PETSc code indicating an error was detected in an MPI call.
*/
PETSC_EXTERN PetscMPIInt PETSC_MPI_ERROR_CLASS;
PETSC_EXTERN PetscMPIInt PETSC_MPI_ERROR_CODE;

/*MC
   SETERRMPI - Macro to be called when an error has been detected within an MPI callback function

   No Fortran Support

   Synopsis:
   #include <petscsys.h>
   PetscErrorCode SETERRMPI(MPI_Comm comm,PetscErrorCode ierr,char *message,...)

   Collective

   Input Parameters:
+  comm - A communicator, use `PETSC_COMM_SELF` unless you know all ranks of another communicator will detect the error
.  ierr - nonzero error code, see the list of standard error codes in include/petscerror.h
-  message - error message

  Level: developer

   Notes:
    This macro is FOR USE IN MPI CALLBACK FUNCTIONS ONLY, such as those passed to `MPI_Comm_create_keyval()`. It always returns the error code `PETSC_MPI_ERROR_CODE`
    which is registered with `MPI_Add_error_code()` when PETSc is initialized.

.seealso: `SETERRQ()`, `PetscCall()`, `PetscCallMPI()`, `PetscTraceBackErrorHandler()`, `PetscPushErrorHandler()`, `PetscError()`, `CHKMEMQ`
M*/
#define SETERRMPI(comm, ierr, ...) return ((void)PetscError(comm, __LINE__, PETSC_FUNCTION_NAME, __FILE__, ierr, PETSC_ERROR_INITIAL, __VA_ARGS__), PETSC_MPI_ERROR_CODE)

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
   This should only be used with Fortran. With C/C++, use `SETERRQ()`.

   `SETERRQ()` may be called from Fortran subroutines but `SETERRA()` must be called from the
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
   This function just calls `MPI_Abort()`.

   This should only be called in routines that cannot return an error code, such as in C++ constructors.

   Fortran Note:
   Use `SETERRA()` in Fortran main program and `SETERRQ()` in Fortran subroutines

   Developer Note:
   In Fortran `SETERRA()` could be called `SETERRABORT()` since they serve the same purpose

.seealso: `SETERRQ()`, `PetscTraceBackErrorHandler()`, `PetscPushErrorHandler()`, `PetscError()`, `PetscCall()`, `CHKMEMQ`
M*/
#define SETERRABORT(comm, ierr, ...) \
  do { \
    (void)PetscError(comm, __LINE__, PETSC_FUNCTION_NAME, __FILE__, ierr, PETSC_ERROR_INITIAL, __VA_ARGS__); \
    MPI_Abort(comm, ierr); \
  } while (0)

/*MC
  PetscCheck - Check that a particular condition is true

  No Fortran Support

  Synopsis:
  #include <petscerror.h>
  void PetscCheck(bool cond, MPI_Comm comm, PetscErrorCode ierr, const char *message, ...)

  Collective

  Input Parameters:
+ cond    - The boolean condition
. comm    - The communicator on which the check can be collective on
. ierr    - A nonzero error code, see include/petscerror.h for the complete list
- message - Error message in printf format

  Level: beginner

  Notes:
  Enabled in both optimized and debug builds.

  Calls `SETERRQ()` if the assertion fails, so can only be called from functions returning a
  `PetscErrorCode` (or equivalent type after conversion).

 .seealso: `PetscAssert()`, `SETERRQ()`, `PetscError()`, `PetscCall()`, `PetscCheckAbort()`
M*/
#define PetscCheck(cond, comm, ierr, ...) \
  do { \
    if (PetscUnlikely(!(cond))) SETERRQ(comm, ierr, __VA_ARGS__); \
  } while (0)

/*MC
  PetscCheckAbort - Check that a particular condition is true, otherwise prints error and aborts

  No Fortran Support

  Synopsis:
  #include <petscerror.h>
  void PetscCheckAbort(bool cond, MPI_Comm comm, PetscErrorCode ierr, const char *message, ...)

  Collective

  Input Parameters:
+ cond    - The boolean condition
. comm    - The communicator on which the check can be collective on
. ierr    - A nonzero error code, see include/petscerror.h for the complete list
- message - Error message in printf format

  Level: developer

  Notes:
  Enabled in both optimized and debug builds.

  Calls `SETERRABORT()` if the assertion fails, can be called from a function that does not return an
  error code, such as a C++ constructor. usually `PetscCheck()` should be used.

.seealso: `PetscAssertAbort()`, `PetscAssert()`, `SETERRQ()`, `PetscError()`, `PetscCall()`, `PetscCheck()`, `SETERRABORT()`
M*/
#define PetscCheckAbort(cond, comm, ierr, ...) \
  do { \
    if (PetscUnlikely(!(cond))) SETERRABORT(comm, ierr, __VA_ARGS__); \
  } while (0)

/*MC
  PetscAssert - Assert that a particular condition is true

  No Fortran Support

  Synopsis:
  #include <petscerror.h>
  void PetscAssert(bool cond, MPI_Comm comm, PetscErrorCode ierr, const char *message, ...)

  Collective

  Input Parameters:
+ cond    - The boolean condition
. comm    - The communicator on which the check can be collective on
. ierr    - A nonzero error code, see include/petscerror.h for the complete list
- message - Error message in printf format

  Level: beginner

  Notes:
  Equivalent to `PetscCheck()` if debugging is enabled, and `PetscAssume(cond)` otherwise.

  See `PetscCheck()` for usage and behaviour.

  This is needed instead of simply using `assert()` because this correctly handles the collective nature of errors under MPI

.seealso: `PetscCheck()`, `SETERRQ()`, `PetscError()`, `PetscAssertAbort()`
M*/
#if PetscDefined(USE_DEBUG)
  #define PetscAssert(cond, comm, ierr, ...) PetscCheck(cond, comm, ierr, __VA_ARGS__)
#else
  #define PetscAssert(cond, ...) PetscAssume(cond)
#endif

/*MC
  PetscAssertAbort - Assert that a particular condition is true, otherwise prints error and aborts

  No Fortran Support

  Synopsis:
  #include <petscerror.h>
  void PetscAssertAbort(bool cond, MPI_Comm comm, PetscErrorCode ierr, const char *message, ...)

  Collective

  Input Parameters:
+ cond    - The boolean condition
. comm    - The communicator on which the check can be collective on
. ierr    - A nonzero error code, see include/petscerror.h for the complete list
- message - Error message in printf format

  Level: beginner

  Notes:
  Enabled only in debug builds. See `PetscCheckAbort()` for usage.

.seealso: `PetscCheckAbort()`, `PetscAssert()`, `PetscCheck()`, `SETERRABORT()`, `PetscError()`
M*/
#if PetscDefined(USE_DEBUG)
  #define PetscAssertAbort(cond, comm, ierr, ...) PetscCheckAbort(cond, comm, ierr, __VA_ARGS__)
#else
  #define PetscAssertAbort(cond, comm, ierr, ...) PetscAssume(cond)
#endif

/*MC
  PetscCall - Calls a PETSc function and then checks the resulting error code, if it is
  non-zero it calls the error handler and returns from the current function with the error
  code.

  Synopsis:
  #include <petscerror.h>
  void PetscCall(PetscFunction(args))

  Not Collective

  Input Parameter:
. PetscFunction - any PETSc function that returns an error code

  Level: beginner

  Notes:
  Once the error handler is called the calling function is then returned from with the given
  error code. Experienced users can set the error handler with `PetscPushErrorHandler()`.

  `PetscCall()` cannot be used in functions returning a datatype not convertible to
  `PetscErrorCode`. For example, `PetscCall()` may not be used in functions returning void, use
  `PetscCallAbort()` or `PetscCallVoid()` in this case.

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

  It is also possible to call this directly on a `PetscErrorCode` variable
.vb
  PetscCall(ierr);  // check if ierr is nonzero
.ve

  Should not be used to call callback functions provided by users, `PetscCallBack()` should be used in that situation.

  `PetscUseTypeMethod()` or `PetscTryTypeMethod()` should be used when calling functions pointers contained in a PETSc object's `ops` array

  Fortran Notes:
    The Fortran function from which this is used must declare a variable PetscErrorCode ierr and ierr must be
    the final argument to the PETSc function being called.

    In the main program and in Fortran subroutines that do not have ierr as the final return parameter one
    should use `PetscCallA()`

  Example Fortran Usage:
.vb
  PetscErrorCode ierr
  Vec v

  ...
  PetscCall(VecShift(v,1.0,ierr))
  PetscCallA(VecShift(v,1.0,ierr))
.ve

.seealso: `SETERRQ()`, `PetscCheck()`, `PetscAssert()`, `PetscTraceBackErrorHandler()`, `PetscCallMPI()`,
          `PetscPushErrorHandler()`, `PetscError()`, `CHKMEMQ`, `CHKERRA()`,
          `CHKERRMPI()`, `PetscCallBack()`, `PetscCallAbort()`, `PetscCallVoid()`
M*/

/*MC
   PetscCallA - Fortran-only macro that should be used in the main program to call PETSc functions instead of using
   PetscCall() which should be used in other Fortran subroutines

   Synopsis:
   #include <petscsys.h>
   PetscErrorCode PetscCallA(PetscFunction(arguments,ierr))

   Collective

   Input Parameter:
.  PetscFunction(arguments,ierr) - the call to the function

  Level: beginner

   Notes:
   This should only be used with Fortran. With C/C++, use `PetscCall()` always.

   Use `SETERRA()` to set an error in a Fortran main program and `SETERRQ()` in Fortran subroutines

.seealso: `SETERRQ()`, `SETERRA()`, `SETERRABORT()`, `PetscCall()`, `CHKERRA()`, `PetscCallAbort()`
M*/

/*MC
  PetscCallBack - Calls a user provided PETSc callback function and then checks the resulting error code, if it is non-zero it calls the error
  handler and returns from the current function with the error code.

  No Fortran Support

  Synopsis:
  #include <petscerror.h>
  void PetscCallBack(const char *functionname,PetscFunction(args))

  Not Collective

  Input Parameters:
+ functionname - the name of the function being called, this can be a string with spaces that describes the meaning of the callback
- PetscFunction - user provided callback function that returns an error code

  Example Usage:
.vb
  PetscCallBack("XXX callback to do something",a->callback(...));
.ve

  Level: developer

  Notes:
  Once the error handler is called the calling function is then returned from with the given
  error code. Experienced users can set the error handler with `PetscPushErrorHandler()`.

  `PetscCallBack()` should only be called in PETSc when a call is being made to a user provided call-back routine.

.seealso: `SETERRQ()`, `PetscCheck()`, `PetscCall()`, `PetscAssert()`, `PetscTraceBackErrorHandler()`, `PetscCallMPI()`
          `PetscPushErrorHandler()`, `PetscError()`, `CHKMEMQ`, `CHKERRA()`, `CHKERRMPI()`, `PetscCall()`
M*/

/*MC
  PetscCallVoid - Like `PetscCall()` but for functions returning `void`

  No Fortran Support

  Synopsis:
  #include <petscerror.h>
  void PetscCall(PetscFunction(args))

  Not Collective

  Input Parameter:
. PetscFunction - any PETSc function that returns an error code

  Example Usage:
.vb
  void foo()
  {
    KSP ksp;

    PetscFunctionBeginUser;
    // OK, properly handles PETSc error codes
    PetscCallVoid(KSPCreate(PETSC_COMM_WORLD, &ksp));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode bar()
  {
    KSP ksp;

    PetscFunctionBeginUser;
    // ERROR, Non-void function 'bar' should return a value
    PetscCallVoid(KSPCreate(PETSC_COMM_WORLD, &ksp));
    // OK, returning PetscErrorCode
    PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
.ve

  Level: beginner

  Notes:
  Has identical usage to `PetscCall()`, except that it returns `void` on error instead of a
  `PetscErrorCode`. See `PetscCall()` for more detailed discussion.

  Note that users should prefer `PetscCallAbort()` to this routine. While this routine does
  "handle" errors by returning from the enclosing function, it effectively gobbles the
  error. Since the enclosing function itself returns `void`, its callers have no way of knowing
  that the routine returned early due to an error. `PetscCallAbort()` at least ensures that the
  program crashes gracefully.

.seealso: `PetscCall()`, `PetscErrorCode`
M*/
#if defined(PETSC_CLANG_STATIC_ANALYZER)
void PetscCall(PetscErrorCode);
void PetscCallBack(const char *, PetscErrorCode);
void PetscCallVoid(PetscErrorCode);
#else
  #define PetscCall(...) \
    do { \
      PetscErrorCode ierr_petsc_call_q_; \
      PetscStackUpdateLine; \
      ierr_petsc_call_q_ = __VA_ARGS__; \
      if (PetscUnlikely(ierr_petsc_call_q_ != PETSC_SUCCESS)) return PetscError(PETSC_COMM_SELF, __LINE__, PETSC_FUNCTION_NAME, __FILE__, ierr_petsc_call_q_, PETSC_ERROR_REPEAT, " "); \
    } while (0)
  #define PetscCallBack(function, ...) \
    do { \
      PetscErrorCode ierr_petsc_call_q_; \
      PetscStackUpdateLine; \
      PetscStackPushExternal(function); \
      ierr_petsc_call_q_ = __VA_ARGS__; \
      PetscStackPop; \
      if (PetscUnlikely(ierr_petsc_call_q_ != PETSC_SUCCESS)) return PetscError(PETSC_COMM_SELF, __LINE__, PETSC_FUNCTION_NAME, __FILE__, ierr_petsc_call_q_, PETSC_ERROR_REPEAT, " "); \
    } while (0)
  #define PetscCallVoid(...) \
    do { \
      PetscErrorCode ierr_petsc_call_void_; \
      PetscStackUpdateLine; \
      ierr_petsc_call_void_ = __VA_ARGS__; \
      if (PetscUnlikely(ierr_petsc_call_void_ != PETSC_SUCCESS)) { \
        ierr_petsc_call_void_ = PetscError(PETSC_COMM_SELF, __LINE__, PETSC_FUNCTION_NAME, __FILE__, ierr_petsc_call_void_, PETSC_ERROR_REPEAT, " "); \
        (void)ierr_petsc_call_void_; \
        return; \
      } \
    } while (0)
#endif

/*MC
  CHKERRQ - Checks error code returned from PETSc function

  Synopsis:
  #include <petscsys.h>
  void CHKERRQ(PetscErrorCode ierr)

  Not Collective

  Input Parameter:
. ierr - nonzero error code

  Level: deprecated

  Note:
  Deprecated in favor of `PetscCall()`. This routine behaves identically to it.

.seealso: `PetscCall()`
M*/
#define CHKERRQ(...) PetscCall(__VA_ARGS__)
#define CHKERRV(...) PetscCallVoid(__VA_ARGS__)

PETSC_EXTERN void PetscMPIErrorString(PetscMPIInt, char *);

/*MC
  PetscCallMPI - Checks error code returned from MPI calls, if non-zero it calls the error
  handler and then returns

  Synopsis:
  #include <petscerror.h>
  void PetscCallMPI(MPI_Function(args))

  Not Collective

  Input Parameter:
. MPI_Function - an MPI function that returns an MPI error code

  Level: beginner

  Notes:
  Always returns the error code `PETSC_ERR_MPI`; the MPI error code and string are embedded in
  the string error message. Do not use this to call any other routines (for example PETSc
  routines), it should only be used for direct MPI calls. The user may configure PETSc with the
  `--with-strict-petscerrorcode` option to check this at compile-time, otherwise they must
  check this themselves.

  This routine can only be used in functions returning `PetscErrorCode` themselves. If the
  calling function returns a different type, use `PetscCallMPIAbort()` instead.

  Example Usage:
.vb
  PetscCallMPI(MPI_Comm_size(...)); // OK, calling MPI function

  PetscCallMPI(PetscFunction(...)); // ERROR, use PetscCall() instead!
.ve

  Fortran Notes:
    The Fortran function from which this is used must declare a variable `PetscErrorCode` ierr and ierr must be
    the final argument to the MPI function being called.

    In the main program and in Fortran subroutines that do not have ierr as the final return parameter one
    should use `PetscCallMPIA()`

  Fortran Usage:
.vb
  PetscErrorCode ierr or integer ierr
  ...
  PetscCallMPI(MPI_Comm_size(...,ierr))
  PetscCallMPIA(MPI_Comm_size(...,ierr)) ! Will abort after calling error handler

  PetscCallMPI(MPI_Comm_size(...,eflag)) ! ERROR, final argument must be ierr
.ve

.seealso: `SETERRMPI()`, `PetscCall()`, `SETERRQ()`, `SETERRABORT()`, `PetscCallAbort()`,
          `PetscCallMPIAbort()`, `PetscTraceBackErrorHandler()`, `PetscPushErrorHandler()`,
          `PetscError()`, `CHKMEMQ`
M*/

/*MC
  PetscCallMPIAbort - Like `PetscCallMPI()` but calls `MPI_Abort()` on error

  Synopsis:
  #include <petscerror.h>
  void PetscCallMPIAbort(MPI_Comm comm, MPI_Function(args))

  Not Collective

  Input Parameters:
+ comm         - the MPI communicator to abort on
- MPI_Function - an MPI function that returns an MPI error code

  Level: beginner

  Notes:
  Usage is identical to `PetscCallMPI()`. See `PetscCallMPI()` for detailed discussion.

  This routine may be used in functions returning `void` or other non-`PetscErrorCode` types.

  Fortran Note:
  In Fortran this is called `PetscCallMPIA()` and is intended to be used in the main program while `PetscCallMPI()` is
  used in Fortran subroutines.

  Developer Note:
  This should have the same name in Fortran.

.seealso: `PetscCallMPI()`, `PetscCallAbort()`, `SETERRABORT()`
M*/
#if defined(PETSC_CLANG_STATIC_ANALYZER)
void PetscCallMPI(PetscMPIInt);
void PetscCallMPIAbort(MPI_Comm, PetscMPIInt);
#else
  #define PetscCallMPI_Private(__PETSC_STACK_POP_FUNC__, __SETERR_FUNC__, __COMM__, ...) \
    do { \
      PetscMPIInt ierr_petsc_call_mpi_; \
      PetscStackUpdateLine; \
      PetscStackPushExternal("MPI function"); \
      { \
        ierr_petsc_call_mpi_ = __VA_ARGS__; \
      } \
      __PETSC_STACK_POP_FUNC__; \
      if (PetscUnlikely(ierr_petsc_call_mpi_ != MPI_SUCCESS)) { \
        char petsc_mpi_7_errorstring[2 * MPI_MAX_ERROR_STRING]; \
        PetscMPIErrorString(ierr_petsc_call_mpi_, (char *)petsc_mpi_7_errorstring); \
        __SETERR_FUNC__(__COMM__, PETSC_ERR_MPI, "MPI error %d %s", (int)ierr_petsc_call_mpi_, petsc_mpi_7_errorstring); \
      } \
    } while (0)

  #define PetscCallMPI(...)            PetscCallMPI_Private(PetscStackPop, SETERRQ, PETSC_COMM_SELF, __VA_ARGS__)
  #define PetscCallMPIAbort(comm, ...) PetscCallMPI_Private(PetscStackPopNoCheck(PETSC_FUNCTION_NAME), SETERRABORT, comm, __VA_ARGS__)
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

  Level: deprecated

  Note:
  Deprecated in favor of `PetscCallMPI()`. This routine behaves identically to it.

.seealso: `PetscCallMPI()`
M*/
#define CHKERRMPI(...) PetscCallMPI(__VA_ARGS__)

/*MC
  PetscCallAbort - Checks error code returned from PETSc function, if non-zero it aborts immediately by calling `MPI_Abort()`

  Synopsis:
  #include <petscerror.h>
  void PetscCallAbort(MPI_Comm comm, PetscErrorCode ierr)

  Collective

  Input Parameters:
+ comm - the MPI communicator on which to abort
- ierr - nonzero error code, see the list of standard error codes in include/petscerror.h

  Level: intermediate

  Notes:
  This macro has identical type and usage semantics to `PetscCall()` with the important caveat
  that this macro does not return. Instead, if ierr is nonzero it calls the PETSc error handler
  and then immediately calls `MPI_Abort()`. It can therefore be used anywhere.

  As per `MPI_Abort()` semantics the communicator passed must be valid, although there is currently
  no attempt made at handling any potential errors from `MPI_Abort()`. Note that while
  `MPI_Abort()` is required to terminate only those processes which reside on comm, it is often
  the case that `MPI_Abort()` terminates *all* processes.

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

  Fortran Note:
  Use `PetscCallA()`.

  Developer Note:
  This should have the same name in Fortran as in C.

.seealso: `SETERRABORT()`, `PetscTraceBackErrorHandler()`, `PetscPushErrorHandler()`, `PetscError()`,
          `SETERRQ()`, `CHKMEMQ`, `PetscCallMPI()`, `PetscCallCXXAbort()`
M*/
#if defined(PETSC_CLANG_STATIC_ANALYZER)
void PetscCallAbort(MPI_Comm, PetscErrorCode);
void PetscCallContinue(PetscErrorCode);
#else
  #define PetscCallAbort(comm, ...) \
    do { \
      PetscErrorCode ierr_petsc_call_abort_; \
      PetscStackUpdateLine; \
      ierr_petsc_call_abort_ = __VA_ARGS__; \
      if (PetscUnlikely(ierr_petsc_call_abort_ != PETSC_SUCCESS)) { \
        ierr_petsc_call_abort_ = PetscError(PETSC_COMM_SELF, __LINE__, PETSC_FUNCTION_NAME, __FILE__, ierr_petsc_call_abort_, PETSC_ERROR_REPEAT, " "); \
        (void)MPI_Abort(comm, (PetscMPIInt)ierr_petsc_call_abort_); \
      } \
    } while (0)
  #define PetscCallContinue(...) \
    do { \
      PetscErrorCode ierr_petsc_call_continue_; \
      PetscStackUpdateLine; \
      ierr_petsc_call_continue_ = __VA_ARGS__; \
      if (PetscUnlikely(ierr_petsc_call_continue_ != PETSC_SUCCESS)) { \
        ierr_petsc_call_continue_ = PetscError(PETSC_COMM_SELF, __LINE__, PETSC_FUNCTION_NAME, __FILE__, ierr_petsc_call_continue_, PETSC_ERROR_REPEAT, " "); \
        (void)ierr_petsc_call_continue_; \
      } \
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

  Level: deprecated

  Note:
  Deprecated in favor of `PetscCallAbort()`. This routine behaves identically to it.

.seealso: `PetscCallAbort()`
M*/
#define CHKERRABORT(comm, ...) PetscCallAbort(comm, __VA_ARGS__)
#define CHKERRCONTINUE(...)    PetscCallContinue(__VA_ARGS__)

/*MC
   CHKERRA - Fortran-only replacement for use of `CHKERRQ()` in the main program, which aborts immediately

   Synopsis:
   #include <petscsys.h>
   PetscErrorCode CHKERRA(PetscErrorCode ierr)

   Not Collective

   Input Parameter:
.  ierr - nonzero error code, see the list of standard error codes in include/petscerror.h

  Level: deprecated

   Note:
   This macro is rarely needed, normal usage is `PetscCallA()` in the main Fortran program.

   Developer Note:
   Why isn't this named `CHKERRABORT()` in Fortran?

.seealso: `PetscCall()`, `PetscCallA()`, `PetscCallAbort()`, `CHKERRQ()`, `SETERRA()`, `SETERRQ()`, `SETERRABORT()`
M*/

PETSC_EXTERN PetscBool petscwaitonerrorflg;
PETSC_EXTERN PetscBool petscindebugger;

/*MC
   PETSCABORT - Call `MPI_Abort()` with an informative error code

   No Fortran Support

   Synopsis:
   #include <petscsys.h>
   PETSCABORT(MPI_Comm comm, PetscErrorCode ierr)

   Collective

   Input Parameters:
+  comm - A communicator, so that the error can be collective
-  ierr - nonzero error code, see the list of standard error codes in include/petscerror.h

   Level: advanced

   Notes:
   If the option `-start_in_debugger` was used then this calls `abort()` to stop the program in the debugger.

   if `PetscCIEnabledPortableErrorOutput` is set, which means the code is running in the PETSc test harness (make test),
   and `comm` is `MPI_COMM_WORLD` it strives to exit cleanly without calling `MPI_Abort()` and instead calling `MPI_Finalize()`.

   This is currently only used when an error propagates up to the C `main()` program and is detected by a `PetscCall()`, `PetscCallMPI()`,
   or is set in `main()` with `SETERRQ()`. Abort calls such as `SETERRABORT()`,
   `PetscCheckAbort()`, `PetscCallMPIAbort()`, and `PetscCallAbort()` always call `MPI_Abort()` and do not have any special
   handling for the test harness.

   Developer Note:
   Should the other abort calls also pass through this call instead of calling `MPI_Abort()` directly?

.seealso: `PetscError()`, `PetscCall()`, `SETERRABORT()`, `PetscCheckAbort()`, `PetscCallMPIAbort()`, `PetscCall()`, `PetscCallMPI()`,
          `PetscCallAbort()`, `MPI_Abort()`
M*/
#if defined(PETSC_CLANG_STATIC_ANALYZER)
void PETSCABORT(MPI_Comm, PetscErrorCode);
#else
  #define PETSCABORT(comm, ...) \
    do { \
      PetscErrorCode ierr_petsc_abort_; \
      if (petscwaitonerrorflg) { ierr_petsc_abort_ = PetscSleep(1000); } \
      if (petscindebugger) { \
        abort(); \
      } else { \
        PetscMPIInt size_; \
        ierr_petsc_abort_ = __VA_ARGS__; \
        MPI_Comm_size(comm, &size_); \
        if (PetscCIEnabledPortableErrorOutput && size_ == PetscGlobalSize && ierr_petsc_abort_ != PETSC_ERR_SIG) { \
          MPI_Finalize(); \
          exit(0); \
        } else if (PetscCIEnabledPortableErrorOutput && PetscGlobalSize == 1) { \
          exit(0); \
        } else { \
          MPI_Abort(comm, (PetscMPIInt)ierr_petsc_abort_); \
        } \
      } \
    } while (0)
#endif

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

  Level: beginner

  Notes:
  Requires PETSc to be configured with clanguage = c++. Throws a std::runtime_error() on error.

  Once the error handler throws the exception you can use `PetscCallVoid()` which returns without
  an error code (bad idea since the error is ignored) or `PetscCallAbort()` to have `MPI_Abort()`
  called immediately.

.seealso: `SETERRQ()`, `PetscCall()`, `SETERRABORT()`, `PetscCallAbort()`, `PetscTraceBackErrorHandler()`,
          `PetscPushErrorHandler()`, `PetscError()`, `CHKMEMQ`
M*/
  #define PetscCallThrow(...) \
    do { \
      PetscStackUpdateLine; \
      PetscErrorCode ierr_petsc_call_throw_ = __VA_ARGS__; \
      if (PetscUnlikely(ierr_petsc_call_throw_ != PETSC_SUCCESS)) PetscError(PETSC_COMM_SELF, __LINE__, PETSC_FUNCTION_NAME, __FILE__, ierr_petsc_call_throw_, PETSC_ERROR_IN_CXX, PETSC_NULLPTR); \
    } while (0)

  /*MC
  CHKERRXX - Checks error code, if non-zero it calls the C++ error handler which throws an exception

  Synopsis:
  #include <petscerror.h>
  void CHKERRXX(PetscErrorCode ierr)

  Not Collective

  Input Parameter:
. ierr - nonzero error code, see the list of standard error codes in include/petscerror.h

  Level: deprecated

  Note:
  Deprecated in favor of `PetscCallThrow()`. This routine behaves identically to it.

.seealso: `PetscCallThrow()`
M*/
  #define CHKERRXX(...) PetscCallThrow(__VA_ARGS__)
#endif

#define PetscCallCXX_Private(__SETERR_FUNC__, __COMM__, ...) \
  do { \
    PetscStackUpdateLine; \
    try { \
      __VA_ARGS__; \
    } catch (const std::exception &e) { \
      __SETERR_FUNC__(__COMM__, PETSC_ERR_LIB, "%s", e.what()); \
    } \
  } while (0)

/*MC
  PetscCallCXX - Checks C++ function calls and if they throw an exception, catch it and then
  return a PETSc error code

  Synopsis:
  #include <petscerror.h>
  void PetscCallCXX(...) noexcept;

  Not Collective

  Input Parameter:
. __VA_ARGS__ - An arbitrary expression

  Level: beginner

  Notes:
  `PetscCallCXX(...)` is a macro replacement for
.vb
  try {
    __VA_ARGS__;
  } catch (const std::exception& e) {
    return ConvertToPetscErrorCode(e);
  }
.ve
  Due to the fact that it catches any (reasonable) exception, it is essentially noexcept.

  If you cannot return a `PetscErrorCode` use `PetscCallCXXAbort()` instead.

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
      foo(); // OK multiple statements allowed
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

.seealso: `PetscCallCXXAbort()`, `PetscCallThrow()`, `SETERRQ()`, `PetscCall()`,
          `SETERRABORT()`, `PetscCallAbort()`, `PetscTraceBackErrorHandler()`, `PetscPushErrorHandler()`,
          `PetscError()`, `CHKMEMQ`
M*/
#define PetscCallCXX(...) PetscCallCXX_Private(SETERRQ, PETSC_COMM_SELF, __VA_ARGS__)

/*MC
  PetscCallCXXAbort - Like `PetscCallCXX()` but calls `MPI_Abort()` instead of returning an
  error-code

  Synopsis:
  #include <petscerror.h>
  void PetscCallCXXAbort(MPI_Comm comm, ...) noexcept;

  Collective; No Fortran Support

  Input Parameters:
+ comm        - The MPI communicator to abort on
- __VA_ARGS__ - An arbitrary expression

  Level: beginner

  Notes:
  This macro may be used to check C++ expressions for exceptions in cases where you cannot
  return an error code. This includes constructors, destructors, copy/move assignment functions
  or constructors among others.

  If an exception is caught, the macro calls `SETERRABORT()` on `comm`. The exception must
  derive from `std::exception` in order to be caught.

  If the routine _can_ return an error-code it is highly advised to use `PetscCallCXX()`
  instead.

  See `PetscCallCXX()` for additional discussion.

  Example Usage:
.vb
  class Foo
  {
    std::vector<int> data_;

  public:
    // normally std::vector::reserve() may raise an exception, but since we handle it with
    // PetscCallCXXAbort() we may mark this routine as noexcept!
    Foo() noexcept
    {
      PetscCallCXXAbort(PETSC_COMM_SELF, data_.reserve(10));
    }
  };

  std::vector<int> bar()
  {
    std::vector<int> v;

    PetscFunctionBegin;
    // OK!
    PetscCallCXXAbort(PETSC_COMM_SELF, v.emplace_back(1));
    PetscFunctionReturn(v);
  }

  PetscErrorCode baz()
  {
    std::vector<int> v;

    PetscFunctionBegin;
    // WRONG! baz() returns a PetscErrorCode, prefer PetscCallCXX() instead
    PetscCallCXXAbort(PETSC_COMM_SELF, v.emplace_back(1));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
.ve

.seealso: `PetscCallCXX()`, `SETERRABORT()`, `PetscCallAbort()`
M*/
#define PetscCallCXXAbort(comm, ...) PetscCallCXX_Private(SETERRABORT, comm, __VA_ARGS__)

/*MC
  CHKERRCXX - Checks C++ function calls and if they throw an exception, catch it and then
  return a PETSc error code

  Synopsis:
  #include <petscerror.h>
  void CHKERRCXX(func) noexcept;

  Not Collective

  Input Parameter:
. func - C++ function calls

  Level: deprecated

  Note:
  Deprecated in favor of `PetscCallCXX()`. This routine behaves identically to it.

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

    Must run with the option `-malloc_debug` (`-malloc_test` in debug mode; or if `PetscMallocSetDebug()` called) to enable this option

    Once the error handler is called the calling function is then returned from with the given error code.

    By defaults prints location where memory that is corrupted was allocated.

    Use `CHKMEMA` for functions that return void

.seealso: `PetscTraceBackErrorHandler()`, `PetscPushErrorHandler()`, `PetscError()`, `SETERRQ()`, `PetscMallocValidate()`
M*/
#if defined(PETSC_CLANG_STATIC_ANALYZER)
  #define CHKMEMQ
  #define CHKMEMA
#else
  #define CHKMEMQ \
    do { \
      PetscErrorCode ierr_petsc_memq_ = PetscMallocValidate(__LINE__, PETSC_FUNCTION_NAME, __FILE__); \
      if (PetscUnlikely(ierr_petsc_memq_ != PETSC_SUCCESS)) return PetscError(PETSC_COMM_SELF, __LINE__, PETSC_FUNCTION_NAME, __FILE__, ierr_petsc_memq_, PETSC_ERROR_REPEAT, " "); \
    } while (0)
  #define CHKMEMA PetscMallocValidate(__LINE__, PETSC_FUNCTION_NAME, __FILE__)
#endif

/*E
  PetscErrorType - passed to the PETSc error handling routines indicating if this is the first or a later call to the error handlers

  Level: advanced

  Note:
  `PETSC_ERROR_IN_CXX` indicates the error was detected in C++ and an exception should be generated

  Developer Note:
    This is currently used to decide when to print the detailed information about the run in `PetscTraceBackErrorHandler()`

.seealso: `PetscError()`, `SETERRQ()`
E*/
typedef enum {
  PETSC_ERROR_INITIAL = 0,
  PETSC_ERROR_REPEAT  = 1,
  PETSC_ERROR_IN_CXX  = 2
} PetscErrorType;

#if defined(__clang_analyzer__)
__attribute__((analyzer_noreturn))
#endif
PETSC_EXTERN PetscErrorCode
PetscError(MPI_Comm, int, const char *, const char *, PetscErrorCode, PetscErrorType, const char *, ...) PETSC_ATTRIBUTE_COLD PETSC_ATTRIBUTE_FORMAT(7, 8);

PETSC_EXTERN PetscErrorCode PetscErrorPrintfInitialize(void);
PETSC_EXTERN PetscErrorCode PetscErrorMessage(PetscErrorCode, const char *[], char **);
PETSC_EXTERN PetscErrorCode PetscTraceBackErrorHandler(MPI_Comm, int, const char *, const char *, PetscErrorCode, PetscErrorType, const char *, void *) PETSC_ATTRIBUTE_COLD;
PETSC_EXTERN PetscErrorCode PetscIgnoreErrorHandler(MPI_Comm, int, const char *, const char *, PetscErrorCode, PetscErrorType, const char *, void *) PETSC_ATTRIBUTE_COLD;
PETSC_EXTERN PetscErrorCode PetscEmacsClientErrorHandler(MPI_Comm, int, const char *, const char *, PetscErrorCode, PetscErrorType, const char *, void *) PETSC_ATTRIBUTE_COLD;
PETSC_EXTERN PetscErrorCode PetscMPIAbortErrorHandler(MPI_Comm, int, const char *, const char *, PetscErrorCode, PetscErrorType, const char *, void *) PETSC_ATTRIBUTE_COLD;
PETSC_EXTERN PetscErrorCode PetscAbortErrorHandler(MPI_Comm, int, const char *, const char *, PetscErrorCode, PetscErrorType, const char *, void *) PETSC_ATTRIBUTE_COLD;
PETSC_EXTERN PetscErrorCode PetscAttachDebuggerErrorHandler(MPI_Comm, int, const char *, const char *, PetscErrorCode, PetscErrorType, const char *, void *) PETSC_ATTRIBUTE_COLD;
PETSC_EXTERN PetscErrorCode PetscReturnErrorHandler(MPI_Comm, int, const char *, const char *, PetscErrorCode, PetscErrorType, const char *, void *) PETSC_ATTRIBUTE_COLD;
PETSC_EXTERN PetscErrorCode PetscPushErrorHandler(PetscErrorCode (*handler)(MPI_Comm, int, const char *, const char *, PetscErrorCode, PetscErrorType, const char *, void *), void *);
PETSC_EXTERN PetscErrorCode PetscPopErrorHandler(void);
PETSC_EXTERN PetscErrorCode PetscSignalHandlerDefault(int, void *);
PETSC_EXTERN PetscErrorCode PetscPushSignalHandler(PetscErrorCode (*)(int, void *), void *);
PETSC_EXTERN PetscErrorCode PetscPopSignalHandler(void);
PETSC_EXTERN PetscErrorCode PetscCheckPointerSetIntensity(PetscInt);
PETSC_EXTERN void           PetscSignalSegvCheckPointerOrMpi(void);
PETSC_DEPRECATED_FUNCTION("Use PetscSignalSegvCheckPointerOrMpi() (since version 3.13)") static inline void PetscSignalSegvCheckPointer(void)
{
  PetscSignalSegvCheckPointerOrMpi();
}

/*MC
    PetscErrorPrintf - Prints error messages.

    Not Collective; No Fortran Support

   Synopsis:
    #include <petscsys.h>
     PetscErrorCode (*PetscErrorPrintf)(const char format[],...);

    Input Parameter:
.   format - the usual `printf()` format string

   Options Database Keys:
+    -error_output_stdout - cause error messages to be printed to stdout instead of the (default) stderr
-    -error_output_none - to turn off all printing of error messages (does not change the way the error is handled.)

   Level: developer

   Notes:
    Use
.vb
     PetscErrorPrintf = PetscErrorPrintfNone; to turn off all printing of error messages (does not change the way the
                        error is handled.) and
     PetscErrorPrintf = PetscErrorPrintfDefault; to turn it back on or you can use your own function
.ve
     Use
.vb
     `PETSC_STDERR` = FILE* obtained from a file open etc. to have stderr printed to the file.
     `PETSC_STDOUT` = FILE* obtained from a file open etc. to have stdout printed to the file.
.ve

       Use
      `PetscPushErrorHandler()` to provide your own error handler that determines what kind of messages to print

.seealso: `PetscFPrintf()`, `PetscSynchronizedPrintf()`, `PetscHelpPrintf()`, `PetscPrintf()`, `PetscPushErrorHandler()`, `PetscVFPrintf()`, `PetscHelpPrintf()`
M*/
PETSC_EXTERN PetscErrorCode (*PetscErrorPrintf)(const char[], ...) PETSC_ATTRIBUTE_FORMAT(1, 2);

/*E
     PetscFPTrap - types of floating point exceptions that may be trapped

     Currently only `PETSC_FP_TRAP_OFF` and `PETSC_FP_TRAP_ON` are handled. All others are treated as `PETSC_FP_TRAP_ON`.

     Level: intermediate

.seealso: `PetscSetFPTrap()`, `PetscPushFPTrap()`
 E*/
typedef enum {
  PETSC_FP_TRAP_OFF      = 0,
  PETSC_FP_TRAP_INDIV    = 1,
  PETSC_FP_TRAP_FLTOPERR = 2,
  PETSC_FP_TRAP_FLTOVF   = 4,
  PETSC_FP_TRAP_FLTUND   = 8,
  PETSC_FP_TRAP_FLTDIV   = 16,
  PETSC_FP_TRAP_FLTINEX  = 32
} PetscFPTrap;
#define PETSC_FP_TRAP_ON (PetscFPTrap)(PETSC_FP_TRAP_INDIV | PETSC_FP_TRAP_FLTOPERR | PETSC_FP_TRAP_FLTOVF | PETSC_FP_TRAP_FLTDIV | PETSC_FP_TRAP_FLTINEX)
PETSC_EXTERN PetscErrorCode PetscSetFPTrap(PetscFPTrap);
PETSC_EXTERN PetscErrorCode PetscFPTrapPush(PetscFPTrap);
PETSC_EXTERN PetscErrorCode PetscFPTrapPop(void);
PETSC_EXTERN PetscErrorCode PetscDetermineInitialFPTrap(void);

/*
      Allows the code to build a stack frame as it runs
*/

#define PETSCSTACKSIZE 64
typedef struct {
  const char *function[PETSCSTACKSIZE];
  const char *file[PETSCSTACKSIZE];
  int         line[PETSCSTACKSIZE];
  int         petscroutine[PETSCSTACKSIZE]; /* 0 external called from petsc, 1 petsc functions, 2 petsc user functions */
  int         currentsize;
  int         hotdepth;
  PetscBool   check; /* option to check for correct Push/Pop semantics, true for default petscstack but not other stacks */
} PetscStack;
#if defined(PETSC_USE_DEBUG) && !defined(PETSC_HAVE_THREADSAFETY)
PETSC_EXTERN PetscStack petscstack;
#endif

#if defined(PETSC_SERIALIZE_FUNCTIONS)
  #include <petsc/private/petscfptimpl.h>
  /*
   Registers the current function into the global function pointer to function name table

   Have to fix this to handle errors but cannot return error since used in PETSC_VIEWER_DRAW_() etc
*/
  #define PetscRegister__FUNCT__() \
    do { \
      static PetscBool __chked = PETSC_FALSE; \
      if (!__chked) { \
        void *ptr; \
        PetscCallAbort(PETSC_COMM_SELF, PetscDLSym(NULL, PETSC_FUNCTION_NAME, &ptr)); \
        __chked = PETSC_TRUE; \
      } \
    } while (0)
#else
  #define PetscRegister__FUNCT__()
#endif

#if defined(PETSC_CLANG_STATIC_ANALYZER) || defined(__clang_analyzer__)
  #define PetscStackPushNoCheck(funct, petsc_routine, hot)
  #define PetscStackUpdateLine
  #define PetscStackPushExternal(funct)
  #define PetscStackPopNoCheck
  #define PetscStackClearTop
  #define PetscFunctionBegin
  #define PetscFunctionBeginUser
  #define PetscFunctionBeginHot
  #define PetscFunctionReturn(...)  return __VA_ARGS__
  #define PetscFunctionReturnVoid() return
  #define PetscStackPop
  #define PetscStackPush(f)
#elif defined(PETSC_USE_DEBUG) && !defined(PETSC_HAVE_THREADSAFETY)

  #define PetscStackPush_Private(stack__, file__, func__, line__, petsc_routine__, hot__) \
    do { \
      if (stack__.currentsize < PETSCSTACKSIZE) { \
        stack__.function[stack__.currentsize] = func__; \
        if (petsc_routine__) { \
          stack__.file[stack__.currentsize] = file__; \
          stack__.line[stack__.currentsize] = line__; \
        } else { \
          stack__.file[stack__.currentsize] = PETSC_NULLPTR; \
          stack__.line[stack__.currentsize] = 0; \
        } \
        stack__.petscroutine[stack__.currentsize] = petsc_routine__; \
      } \
      ++stack__.currentsize; \
      stack__.hotdepth += (hot__ || stack__.hotdepth); \
    } while (0)

  /* uses PetscCheckAbort() because may be used in a function that does not return an error code */
  #define PetscStackPop_Private(stack__, func__) \
    do { \
      PetscCheckAbort(!stack__.check || stack__.currentsize > 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid stack size %d, pop %s %s:%d.\n", stack__.currentsize, func__, __FILE__, __LINE__); \
      if (--stack__.currentsize < PETSCSTACKSIZE) { \
        PetscCheckAbort(!stack__.check || stack__.petscroutine[stack__.currentsize] != 1 || stack__.function[stack__.currentsize] == (const char *)(func__), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid stack: push from %s %s:%d. Pop from %s %s:%d.\n", \
                        stack__.function[stack__.currentsize], stack__.file[stack__.currentsize], stack__.line[stack__.currentsize], func__, __FILE__, __LINE__); \
        stack__.function[stack__.currentsize]     = PETSC_NULLPTR; \
        stack__.file[stack__.currentsize]         = PETSC_NULLPTR; \
        stack__.line[stack__.currentsize]         = 0; \
        stack__.petscroutine[stack__.currentsize] = 0; \
      } \
      stack__.hotdepth = PetscMax(stack__.hotdepth - 1, 0); \
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
   occurred, for example, when a signal is received without running in the debugger. It is recommended to use the debugger if extensive information is needed to
   help debug the problem.

   This version does not check the memory corruption (an expensive operation), use `PetscStackPush()` to check the memory.

   Use `PetscStackPushExternal()` for a function call that is about to be made to a non-PETSc or user function (such as BLAS etc).

   The default stack is a global variable called `petscstack`.

.seealso: `PetscAttachDebugger()`, `PetscStackCopy()`, `PetscStackView()`, `PetscStackPopNoCheck()`, `PetscCall()`, `PetscFunctionBegin()`,
          `PetscFunctionReturn()`, `PetscFunctionBeginHot()`, `PetscFunctionBeginUser()`, `PetscStackPush()`, `PetscStackPop`,
          `PetscStackPushExternal()`
M*/
  #define PetscStackPushNoCheck(funct, petsc_routine, hot) \
    do { \
      PetscStackSAWsTakeAccess(); \
      PetscStackPush_Private(petscstack, __FILE__, funct, __LINE__, petsc_routine, hot); \
      PetscStackSAWsGrantAccess(); \
    } while (0)

  /*MC
   PetscStackUpdateLine - in a function that has a `PetscFunctionBegin` or `PetscFunctionBeginUser` updates the stack line number to the
   current line number.

   Not Collective

   Synopsis:
   #include <petscsys.h>
   void PetscStackUpdateLine

   Level: developer

   Notes:
   Using `PetscCall()` and friends automatically handles this process

   In debug mode PETSc maintains a stack of the current function calls that can be used to help to quickly see where a problem has
   occurred, for example, when a signal is received. It is recommended to use the debugger if extensive information is needed to
   help debug the problem.

   The default stack is a global variable called petscstack.

   This is used by `PetscCall()` and is otherwise not like to be needed

.seealso: `PetscAttachDebugger()`, `PetscStackCopy()`, `PetscStackView()`, `PetscStackPushNoCheck()`, `PetscStackPop`, `PetscCall()`
M*/
  #define PetscStackUpdateLine \
    do { \
      if (petscstack.currentsize > 0 && petscstack.function[petscstack.currentsize - 1] == PETSC_FUNCTION_NAME) { petscstack.line[petscstack.currentsize - 1] = __LINE__; } \
    } while (0)

  /*MC
   PetscStackPushExternal - Pushes a new function name onto the PETSc default stack that tracks where the running program is
   currently in the source code. Does not include the filename or line number since this is called by the calling routine
   for non-PETSc or user functions.

   Not Collective

   Synopsis:
   #include <petscsys.h>
   void PetscStackPushExternal(char *funct);

   Input Parameter:
.  funct - the function name

   Level: developer

   Notes:
   Using `PetscCallExternal()` and friends automatically handles this process

   In debug mode PETSc maintains a stack of the current function calls that can be used to help to quickly see where a problem has
   occurred, for example, when a signal is received. It is recommended to use the debugger if extensive information is needed to
   help debug the problem.

   The default stack is a global variable called `petscstack`.

   This is to be used when calling an external package function such as a BLAS function.

   This also updates the stack line number for the current stack function.

.seealso: `PetscAttachDebugger()`, `PetscStackCopy()`, `PetscStackView()`, `PetscStackPopNoCheck()`, `PetscCall()`, `PetscFunctionBegin()`,
          `PetscFunctionReturn()`, `PetscFunctionBeginHot()`, `PetscFunctionBeginUser()`, `PetscStackPushNoCheck()`, `PetscStackPop`
M*/
  #define PetscStackPushExternal(funct) \
    do { \
      PetscStackUpdateLine; \
      PetscStackPushNoCheck(funct, 0, PETSC_TRUE); \
    } while (0);

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
   Using `PetscCall()`, `PetscCallExternal()`, `PetscCallBack()` and friends negates the need to call this

   In debug mode PETSc maintains a stack of the current function calls that can be used to help to quickly see where a problem has
   occurred, for example, when a signal is received. It is recommended to use the debugger if extensive information is needed to
   help debug the problem.

   The default stack is a global variable called petscstack.

   Developer Note:
   `PetscStackPopNoCheck()` takes a function argument while  `PetscStackPop` does not, this difference is likely just historical.

.seealso: `PetscAttachDebugger()`, `PetscStackCopy()`, `PetscStackView()`, `PetscStackPushNoCheck()`, `PetscStackPop`
M*/
  #define PetscStackPopNoCheck(funct) \
    do { \
      PetscStackSAWsTakeAccess(); \
      PetscStackPop_Private(petscstack, funct); \
      PetscStackSAWsGrantAccess(); \
    } while (0)

  #define PetscStackClearTop \
    do { \
      PetscStackSAWsTakeAccess(); \
      if (petscstack.currentsize > 0 && --petscstack.currentsize < PETSCSTACKSIZE) { \
        petscstack.function[petscstack.currentsize]     = PETSC_NULLPTR; \
        petscstack.file[petscstack.currentsize]         = PETSC_NULLPTR; \
        petscstack.line[petscstack.currentsize]         = 0; \
        petscstack.petscroutine[petscstack.currentsize] = 0; \
      } \
      petscstack.hotdepth = PetscMax(petscstack.hotdepth - 1, 0); \
      PetscStackSAWsGrantAccess(); \
    } while (0)

  /*MC
   PetscFunctionBegin - First executable line of each PETSc function,  used for error handling. Final
      line of PETSc functions should be `PetscFunctionReturn`(0);

   Synopsis:
   #include <petscsys.h>
   void PetscFunctionBegin;

   Not Collective; No Fortran Support

   Usage:
.vb
     int something;

     PetscFunctionBegin;
.ve

   Level: developer

   Note:
     Use `PetscFunctionBeginUser` for application codes.

.seealso: `PetscFunctionReturn()`, `PetscFunctionBeginHot()`, `PetscFunctionBeginUser()`, `PetscStackPushNoCheck()`

M*/
  #define PetscFunctionBegin \
    do { \
      PetscStackPushNoCheck(PETSC_FUNCTION_NAME, 1, PETSC_FALSE); \
      PetscRegister__FUNCT__(); \
    } while (0)

  /*MC
   PetscFunctionBeginHot - Substitute for `PetscFunctionBegin` to be used in functions that are called in
   performance-critical circumstances.  Use of this function allows for lighter profiling by default.

   Synopsis:
   #include <petscsys.h>
   void PetscFunctionBeginHot;

   Not Collective; No Fortran Support

   Usage:
.vb
     int something;

     PetscFunctionBeginHot;
.ve

   Level: developer

.seealso: `PetscFunctionBegin`, `PetscFunctionReturn()`, `PetscStackPushNoCheck()`

M*/
  #define PetscFunctionBeginHot \
    do { \
      PetscStackPushNoCheck(PETSC_FUNCTION_NAME, 1, PETSC_TRUE); \
      PetscRegister__FUNCT__(); \
    } while (0)

  /*MC
   PetscFunctionBeginUser - First executable line of user provided routines

   Synopsis:
   #include <petscsys.h>
   void PetscFunctionBeginUser;

   Not Collective; No Fortran Support

   Usage:
.vb
     int something;

     PetscFunctionBeginUser;
.ve

   Level: intermediate

   Notes:
      Functions that incorporate this must call `PetscFunctionReturn()` instead of return except for main().

      May be used before `PetscInitialize()`

      This is identical to `PetscFunctionBegin` except it labels the routine as a user
      routine instead of as a PETSc library routine.

.seealso: `PetscFunctionReturn()`, `PetscFunctionBegin`, `PetscFunctionBeginHot`, `PetscStackPushNoCheck()`

M*/
  #define PetscFunctionBeginUser \
    do { \
      PetscStackPushNoCheck(PETSC_FUNCTION_NAME, 2, PETSC_FALSE); \
      PetscRegister__FUNCT__(); \
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

.seealso: `PetscAttachDebugger()`, `PetscStackCopy()`, `PetscStackView()`, `PetscStackPopNoCheck()`, `PetscCall()`, `PetscFunctionBegin()`,
          `PetscFunctionReturn()`, `PetscFunctionBeginHot()`, `PetscFunctionBeginUser()`, `PetscStackPushNoCheck()`, `PetscStackPop`
M*/
  #define PetscStackPush(n) \
    do { \
      PetscStackPushNoCheck(n, 0, PETSC_FALSE); \
      CHKMEMQ; \
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
  #define PetscStackPop \
    do { \
      CHKMEMQ; \
      PetscStackPopNoCheck(PETSC_FUNCTION_NAME); \
    } while (0)

  /*MC
   PetscFunctionReturn - Last executable line of each PETSc function used for error
   handling. Replaces `return()`.

   Synopsis:
   #include <petscerror.h>
   void PetscFunctionReturn(...)

   Not Collective; No Fortran Support

   Level: beginner

   Notes:
   This routine is a macro, so while it does not "return" anything itself, it does return from
   the function in the literal sense.

   Usually the return value is the integer literal `0` (for example in any function returning
   `PetscErrorCode`), however it is possible to return any arbitrary type. The arguments of
   this macro are placed before the `return` statement as-is.

   Any routine which returns via `PetscFunctionReturn()` must begin with a corresponding
   `PetscFunctionBegin`.

   For routines which return `void` use `PetscFunctionReturnVoid()` instead.

   Example Usage:
.vb
   PetscErrorCode foo(int *x)
   {
     PetscFunctionBegin; // don't forget the begin!
     *x = 10;
     PetscFunctionReturn(PETSC_SUCCESS);
   }
.ve

   May return any arbitrary type\:
.vb
  struct Foo
  {
    int x;
  };

  struct Foo make_foo(int value)
  {
    struct Foo f;

    PetscFunctionBegin;
    f.x = value;
    PetscFunctionReturn(f);
  }
.ve

.seealso: `PetscFunctionBegin`, `PetscFunctionBeginUser`, `PetscFunctionReturnVoid()`,
          `PetscStackPopNoCheck()`
M*/
  #define PetscFunctionReturn(...) \
    do { \
      PetscStackPopNoCheck(PETSC_FUNCTION_NAME); \
      return __VA_ARGS__; \
    } while (0)

  /*MC
  PetscFunctionReturnVoid - Like `PetscFunctionReturn()` but returns `void`

  Synopsis:
  #include <petscerror.h>
  void PetscFunctionReturnVoid()

  Not Collective

  Level: beginner

  Note:
  Behaves identically to `PetscFunctionReturn()` except that it returns `void`. That is, this
  macro culminates with `return`.

  Example Usage:
.vb
  void foo()
  {
    PetscFunctionBegin; // must start with PetscFunctionBegin!
    bar();
    baz();
    PetscFunctionReturnVoid();
  }
.ve

.seealso: `PetscFunctionReturn()`, `PetscFunctionBegin`, PetscFunctionBeginUser`
M*/
  #define PetscFunctionReturnVoid() \
    do { \
      PetscStackPopNoCheck(PETSC_FUNCTION_NAME); \
      return; \
    } while (0)
#else /* PETSC_USE_DEBUG */
  #define PetscStackPushNoCheck(funct, petsc_routine, hot)
  #define PetscStackUpdateLine
  #define PetscStackPushExternal(funct)
  #define PetscStackPopNoCheck(...)
  #define PetscStackClearTop
  #define PetscFunctionBegin
  #define PetscFunctionBeginUser
  #define PetscFunctionBeginHot
  #define PetscFunctionReturn(...)  return __VA_ARGS__
  #define PetscFunctionReturnVoid() return
  #define PetscStackPop             CHKMEMQ
  #define PetscStackPush(f)         CHKMEMQ
#endif /* PETSC_USE_DEBUG */

#if defined(PETSC_CLANG_STATIC_ANALYZER)
  #define PetscStackCallExternalVoid(...)
template <typename F, typename... Args>
void PetscCallExternal(F, Args...);
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
  #define PetscStackCallExternalVoid(name, ...) \
    do { \
      PetscStackPush(name); \
      __VA_ARGS__; \
      PetscStackPop; \
    } while (0)

  /*MC
    PetscCallExternal - Calls an external library routine that returns an error code after pushing the name of the routine on the stack.

   Input Parameters:
+   func-  name of the routine
-   args - arguments to the routine

   Level: developer

   Notes:
   This is intended for external package routines that return error codes. Use `PetscStackCallExternalVoid()` for those that do not.

   In debug mode this also checks the memory for corruption at the end of the function call.

   Assumes the error return code of the function is an integer and that a value of 0 indicates success

   Developer Note:
   This is so that when an external package routine results in a crash or corrupts memory, they get blamed instead of PETSc.

.seealso: `PetscCall()`, `PetscStackPushNoCheck()`, `PetscStackPush()`, `PetscStackCallExternalVoid()`
M*/
  #define PetscCallExternal(func, ...) \
    do { \
      PetscStackPush(PetscStringize(func)); \
      int ierr_petsc_call_external_ = func(__VA_ARGS__); \
      PetscStackPop; \
      PetscCheck(ierr_petsc_call_external_ == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in %s(): error code %d", PetscStringize(func), ierr_petsc_call_external_); \
    } while (0)
#endif /* PETSC_CLANG_STATIC_ANALYZER */

#endif

/*
    Contains all error handling interfaces for PETSc.
*/
#if !defined(PETSCERROR_H)
#define PETSCERROR_H

#include <petscmacros.h>
#include <petscsystypes.h>

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

.seealso: PetscCheck(), PetscAssert(), PetscTraceBackErrorHandler(), PetscPushErrorHandler(),
PetscError(), CHKERRQ(), CHKMEMQ, CHKERRA(), CHKERRMPI()
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

.seealso: SETERRQ(), CHKERRQ(), CHKERRMPI(), PetscTraceBackErrorHandler(), PetscPushErrorHandler(), PetscError(), CHKMEMQ
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

.seealso: SETERRQ(), SETERRABORT(), CHKERRQ(), CHKERRA(), CHKERRABORT()
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

.seealso: SETERRQ(), PetscTraceBackErrorHandler(), PetscPushErrorHandler(), PetscError(), CHKERRQ(), CHKMEMQ
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

.seealso: PetscAssert(), SETERRQ(), PetscError(), CHKERRQ()
MC*/
#define PetscCheck(cond,comm,ierr,...) if (PetscUnlikely(!(cond))) SETERRQ(comm,ierr,__VA_ARGS__)

/*MC
  PetscCheckFalse - Check that a particular condition is false

  Synopsis:
  #include <petscerror.h>
  void PetscCheckFalse(bool cond, MPI_Comm comm, PetscErrorCode ierr, const char *message, ...)

  Collective

  Input Parameters:
+ cond    - The boolean condition
. comm    - The communicator on which the check can be collective on
. ierr    - A nonzero error code, see include/petscerror.h for the complete list
- message - Error message in printf format

  Notes:
  Invert your boolean condition and use PetscCheck() instead. This macro is a temporary stopgap
  to converting to PetscCheck() and is subject to removal without deprecation in a future
  release.

  Level: deprecated

.seealso: PetscCheck()
MC*/
#define PetscCheckFalse(cond,comm,ierr,...) PetscCheck(!(cond),comm,ierr,__VA_ARGS__)

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

.seealso: PetscCheck(), SETERRQ(), PetscError()
MC*/
#define PetscAssert(cond,comm,ierr,...) if (PetscUnlikelyDebug(!(cond))) SETERRQ(comm,ierr,__VA_ARGS__)

/*MC
  PetscAssertFalse - Assert that a particular condition is false

  Synopsis:
  #include <petscerror.h>
  void PetscAssertFalse(bool cond, MPI_Comm comm, PetscErrorCode ierr, const char *message, ...)

  Collective

  Input Parameters:
+ cond    - The boolean condition
. comm    - The communicator on which the check can be collective on
. ierr    - A nonzero error code, see include/petscerror.h for the complete list
- message - Error message in printf format

  Notes:
  Invert your boolean condition and use PetscAssert() instead. This macro is a temporary
  stopgap to converting to PetscAssert() and is subject to removal without deprecation in a
  future release.

  Level: deprecated

.seealso: PetscAssert()
MC*/
#define PetscAssertFalse(cond,comm,ierr,...) PetscAssert(!(cond),comm,ierr,__VA_ARGS__)

/*MC
   CHKERRQ - Checks error code returned from PETSc function, if non-zero it calls the error handler and then returns. Use CHKERRMPI() for checking errors from MPI calls

   Synopsis:
   #include <petscsys.h>
   PetscErrorCode CHKERRQ(PetscErrorCode ierr)

   Not Collective

   Input Parameters:
.  ierr - nonzero error code, see the list of standard error codes in include/petscerror.h

  Level: beginner

   Notes:
    Once the error handler is called the calling function is then returned from with the given error code.

    Experienced users can set the error handler with PetscPushErrorHandler().

    CHKERRQ(ierr) is fundamentally a macro replacement for
         if (ierr) return(PetscError(...,ierr,...));

    Although typical usage resembles "void CHKERRQ(PetscErrorCode)" as described above, for certain uses it is
    highly inappropriate to use it in this manner as it invokes return(PetscErrorCode). In particular,
    it cannot be used in functions which return(void) or any other datatype.  In these types of functions,
    you can use CHKERRV() which returns without an error code (bad idea since the error is ignored or
         if (ierr) {PetscError(....); return(YourReturnType);}
    where you may pass back a NULL to indicate an error. You can also call CHKERRABORT(comm,n) to have
    MPI_Abort() returned immediately.

   Fortran Notes:
      CHKERRQ() may be called from Fortran subroutines but CHKERRA() must be called from the
      Fortran main program.

.seealso: SETERRQ(), PetscCheck(), PetscAssert(), PetscTraceBackErrorHandler(),
PetscPushErrorHandler(), PetscError(), CHKMEMQ, CHKERRA()
M*/
#if !defined(PETSC_CLANG_STATIC_ANALYZER)
#define CHKERRQ(ierr)          do {PetscErrorCode ierr__ = (ierr); if (PetscUnlikely(ierr__)) return PetscError(PETSC_COMM_SELF,__LINE__,PETSC_FUNCTION_NAME,__FILE__,ierr__,PETSC_ERROR_REPEAT," ");} while (0)
#define CHKERRV(ierr)          do {PetscErrorCode ierr__ = (ierr); if (PetscUnlikely(ierr__)) {PetscError(PETSC_COMM_SELF,__LINE__,PETSC_FUNCTION_NAME,__FILE__,ierr__,PETSC_ERROR_REPEAT," ");return;}} while (0)
#else
#define CHKERRQ(ierr)
#define CHKERRV(ierr)
#endif

/*MC
   CHKERRA - Fortran-only replacement for CHKERRQ in the main program, which aborts immediately

   Synopsis:
   #include <petscsys.h>
   PetscErrorCode CHKERRA(PetscErrorCode ierr)

   Not Collective

   Input Parameters:
.  ierr - nonzero error code, see the list of standard error codes in include/petscerror.h

  Level: beginner

   Notes:
      This should only be used with Fortran. With C/C++, use CHKERRQ() in normal usage,
      or CHKERRABORT() if wanting to abort immediately on error.

   Fortran Notes:
      CHKERRQ() may be called from Fortran subroutines but CHKERRA() must be called from the
      Fortran main program.

.seealso: CHKERRQ(), CHKERRABORT(), SETERRA(), SETERRQ(), SETERRABORT()
M*/

/*MC
   CHKERRABORT - Checks error code returned from PETSc function. If non-zero it aborts immediately.

   Synopsis:
   #include <petscsys.h>
   PetscErrorCode CHKERRABORT(MPI_Comm comm,PetscErrorCode ierr)

   Not Collective

   Input Parameters:
.  ierr - nonzero error code, see the list of standard error codes in include/petscerror.h

  Level: intermediate

.seealso: SETERRABORT(), PetscTraceBackErrorHandler(), PetscPushErrorHandler(), PetscError(), SETERRQ(), CHKMEMQ, CHKERRMPI()
M*/
#if defined(PETSC_CLANG_STATIC_ANALYZER)
#define CHKERRABORT(comm,ierr)
#define CHKERRCONTINUE(ierr)
#else
#define CHKERRABORT(comm,ierr) do {PetscErrorCode ierr__ = (ierr); if (PetscUnlikely(ierr__)) {PetscError(PETSC_COMM_SELF,__LINE__,PETSC_FUNCTION_NAME,__FILE__,ierr__,PETSC_ERROR_REPEAT," ");MPI_Abort(comm,ierr);}} while (0)
#define CHKERRCONTINUE(ierr)   do {PetscErrorCode ierr__ = (ierr); if (PetscUnlikely(ierr__)) {PetscError(PETSC_COMM_SELF,__LINE__,PETSC_FUNCTION_NAME,__FILE__,ierr__,PETSC_ERROR_REPEAT," ");}} while (0)
#endif

PETSC_EXTERN PetscErrorCode PetscAbortFindSourceFile_Private(const char*,PetscInt*);
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
   We pass MPI_Abort() an error code of format XX_YYYY_ZZZ, where XX, YYYY are an index and line number of the file
   where PETSCABORT is called, respectively. ZZZ is the PETSc error code.

   If XX is zero, this means that the call was made in the routine main().
   If XX is one, that means 1) the file is not in PETSc (it may be in users code); OR 2) the file is in PETSc but PetscAbortSourceFiles[]
     is out of date. PETSc developers have to update it.
   Otherwise, look up the value of XX in the table PetscAbortSourceFiles[] in src/sys/error/err.c to map XX back to the source file where the PETSCABORT() was called.

   If the option -start_in_debugger was used then this calls abort() to stop the program in the debugger.

M*/
#define PETSCABORT(comm,ierr)  \
   do {                                                               \
      PetscInt       idx = 0;                                         \
      PetscMPIInt    errcode;                                         \
      PetscAbortFindSourceFile_Private(__FILE__,&idx);                \
      errcode = (PetscMPIInt)(0*idx*10000000 + 0*__LINE__*1000 + ierr);   \
      if (petscwaitonerrorflg) PetscSleep(1000);                      \
      if (petscindebugger) abort();                                   \
      else MPI_Abort(comm,errcode);                                   \
   } while (0)

/*MC
   CHKERRMPI - Checks error code returned from MPI calls, if non-zero it calls the error handler and then returns

   Synopsis:
   #include <petscsys.h>
   PetscErrorCode CHKERRMPI(PetscErrorCode ierr)

   Not Collective

   Input Parameters:
.  ierr - nonzero error code, see the list of standard error codes in include/petscerror.h

  Level: intermediate

   Notes:
    Always returns the error code PETSC_ERR_MPI; the MPI error code and string are embedded in the string error message

.seealso: SETERRMPI(), CHKERRQ(), SETERRQ(), SETERRABORT(), CHKERRABORT(), PetscTraceBackErrorHandler(), PetscPushErrorHandler(), PetscError(), CHKMEMQ
M*/
#if defined(PETSC_CLANG_STATIC_ANALYZER)
#define CHKERRMPI(ierr)
#else
#define CHKERRMPI(ierr) \
do { \
  PetscErrorCode _7_errorcode = (ierr); \
  if (PetscUnlikely(_7_errorcode)) { \
    char _7_errorstring[MPI_MAX_ERROR_STRING]; \
    PetscMPIInt _7_resultlen; \
    MPI_Error_string(_7_errorcode,(char*)_7_errorstring,&_7_resultlen); (void)_7_resultlen; \
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MPI,"MPI error %d %s",(int)_7_errorcode,_7_errorstring); \
  } \
} while (0)
#endif

#ifdef PETSC_CLANGUAGE_CXX

/*MC
   CHKERRXX - Checks error code, if non-zero it calls the C++ error handler which throws an exception

   Synopsis:
   #include <petscsys.h>
   void CHKERRXX(PetscErrorCode ierr)

   Not Collective

   Input Parameters:
.  ierr - nonzero error code, see the list of standard error codes in include/petscerror.h

  Level: beginner

   Notes:
    Once the error handler throws a ??? exception.

    You can use CHKERRV() which returns without an error code (bad idea since the error is ignored)
    or CHKERRABORT(comm,n) to have MPI_Abort() returned immediately.

.seealso: SETERRQ(), CHKERRQ(), SETERRABORT(), CHKERRABORT(), PetscTraceBackErrorHandler(), PetscPushErrorHandler(), PetscError(), CHKMEMQ
M*/
#define CHKERRXX(ierr)  do {if (PetscUnlikely(ierr)) {PetscError(PETSC_COMM_SELF,__LINE__,PETSC_FUNCTION_NAME,__FILE__,ierr,PETSC_ERROR_IN_CXX,0);}} while (0)
#endif

/*MC
   CHKERRCXX - Checks C++ function calls and if they throw an exception, catch it and then return a PETSc error code

   Synopsis:
   #include <petscsys.h>
   CHKERRCXX(func);

   Not Collective

   Input Parameters:
.  func - C++ function calls

  Level: beginner

  Notes:
   For example,

$     void foo(int x) {throw std::runtime_error("error");}
$     CHKERRCXX(foo(1));

.seealso: CHKERRXX(), SETERRQ(), CHKERRQ(), SETERRABORT(), CHKERRABORT(), PetscTraceBackErrorHandler(), PetscPushErrorHandler(), PetscError(), CHKMEMQ
M*/
#define CHKERRCXX(func) do {try {func;} catch (const std::exception& e) { SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"%s", e.what()); }} while (0)

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

.seealso: PetscTraceBackErrorHandler(), PetscPushErrorHandler(), PetscError(), SETERRQ(), PetscMallocValidate()
M*/
#if defined(PETSC_CLANG_STATIC_ANALYZER)
#define CHKMEMQ
#define CHKMEMA
#else
#define CHKMEMQ do {PetscErrorCode _7_ierr = PetscMallocValidate(__LINE__,PETSC_FUNCTION_NAME,__FILE__);CHKERRQ(_7_ierr);} while (0)
#define CHKMEMA PetscMallocValidate(__LINE__,PETSC_FUNCTION_NAME,__FILE__)
#endif
/*E
  PetscErrorType - passed to the PETSc error handling routines indicating if this is the first or a later call to the error handlers

  Level: advanced

  PETSC_ERROR_IN_CXX indicates the error was detected in C++ and an exception should be generated

  Developer Notes:
    This is currently used to decide when to print the detailed information about the run in PetscTraceBackErrorHandler()

.seealso: PetscError(), SETERRXX()
E*/
typedef enum {PETSC_ERROR_INITIAL=0,PETSC_ERROR_REPEAT=1,PETSC_ERROR_IN_CXX = 2} PetscErrorType;

#if defined(__clang_analyzer__)
__attribute__((analyzer_noreturn))
#endif
PETSC_EXTERN PetscErrorCode PetscError(MPI_Comm,int,const char*,const char*,PetscErrorCode,PetscErrorType,const char*,...) PETSC_ATTRIBUTE_FORMAT(7,8);

PETSC_EXTERN PetscErrorCode PetscErrorPrintfInitialize(void);
PETSC_EXTERN PetscErrorCode PetscErrorMessage(int,const char*[],char **);
PETSC_EXTERN PetscErrorCode PetscTraceBackErrorHandler(MPI_Comm,int,const char*,const char*,PetscErrorCode,PetscErrorType,const char*,void*);
PETSC_EXTERN PetscErrorCode PetscIgnoreErrorHandler(MPI_Comm,int,const char*,const char*,PetscErrorCode,PetscErrorType,const char*,void*);
PETSC_EXTERN PetscErrorCode PetscEmacsClientErrorHandler(MPI_Comm,int,const char*,const char*,PetscErrorCode,PetscErrorType,const char*,void*);
PETSC_EXTERN PetscErrorCode PetscMPIAbortErrorHandler(MPI_Comm,int,const char*,const char*,PetscErrorCode,PetscErrorType,const char*,void*);
PETSC_EXTERN PetscErrorCode PetscAbortErrorHandler(MPI_Comm,int,const char*,const char*,PetscErrorCode,PetscErrorType,const char*,void*);
PETSC_EXTERN PetscErrorCode PetscAttachDebuggerErrorHandler(MPI_Comm,int,const char*,const char*,PetscErrorCode,PetscErrorType,const char*,void*);
PETSC_EXTERN PetscErrorCode PetscReturnErrorHandler(MPI_Comm,int,const char*,const char*,PetscErrorCode,PetscErrorType,const char*,void*);
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

.seealso: PetscFPrintf(), PetscSynchronizedPrintf(), PetscHelpPrintf(), PetscPrintf(), PetscPushErrorHandler(), PetscVFPrintf(), PetscHelpPrintf()
M*/
PETSC_EXTERN PetscErrorCode (*PetscErrorPrintf)(const char[],...) PETSC_ATTRIBUTE_FORMAT(1,2);

typedef enum {PETSC_FP_TRAP_OFF=0,PETSC_FP_TRAP_ON=1} PetscFPTrap;
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
  PetscBool  check; /* runtime option to check for correct Push/Pop semantics at runtime */
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
/* Stack handling is based on the following two "NoCheck" macros.  These should only be called directly by other error
 * handling macros.  We record the line of the call, which may or may not be the location of the definition.  But is at
 * least more useful than "unknown" because it can distinguish multiple calls from the same function.
 */
#define PetscStackPushNoCheck(funct,petsc_routine,hot) do {             \
    PetscStackSAWsTakeAccess();                                         \
    if (petscstack.currentsize < PETSCSTACKSIZE) {                      \
      petscstack.function[petscstack.currentsize]     = funct;          \
      petscstack.file[petscstack.currentsize]         = __FILE__;       \
      petscstack.line[petscstack.currentsize]         = __LINE__;       \
      petscstack.petscroutine[petscstack.currentsize] = petsc_routine;  \
    }                                                                   \
    ++petscstack.currentsize;                                           \
    petscstack.hotdepth += (hot || petscstack.hotdepth);                \
    PetscStackSAWsGrantAccess();                                        \
  } while (0)

#define PetscStackPopNoCheck(funct)                    do {             \
    PetscStackSAWsTakeAccess();                                         \
    if (PetscUnlikely(petscstack.currentsize <= 0)) {                   \
      if (PetscUnlikely(petscstack.check)) {                            \
        printf("Invalid stack size %d, pop %s\n",                       \
               petscstack.currentsize,funct);                           \
      }                                                                 \
    } else {                                                            \
      if (--petscstack.currentsize < PETSCSTACKSIZE) {                  \
        if (PetscUnlikely(                                              \
              petscstack.check                                &&        \
              petscstack.petscroutine[petscstack.currentsize] &&        \
              (petscstack.function[petscstack.currentsize]    !=        \
               (const char*)funct))) {                                  \
          /* We need this string comparison because "unknown" can be defined in different static strings: */ \
          PetscBool _cmpflg;                                            \
          const char *_funct = petscstack.function[petscstack.currentsize]; \
          PetscStrcmp(_funct,funct,&_cmpflg);                           \
          if (!_cmpflg)                                                 \
            printf("Invalid stack: push from %s, pop from %s\n", _funct,funct); \
        }                                                               \
        petscstack.function[petscstack.currentsize] = PETSC_NULLPTR;    \
        petscstack.file[petscstack.currentsize]     = PETSC_NULLPTR;    \
        petscstack.line[petscstack.currentsize]     = 0;                \
        petscstack.petscroutine[petscstack.currentsize] = 0;            \
      }                                                                 \
      petscstack.hotdepth = PetscMax(petscstack.hotdepth-1,0);          \
    }                                                                   \
    PetscStackSAWsGrantAccess();                                        \
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

.seealso: PetscFunctionReturn(), PetscFunctionBeginHot(), PetscFunctionBeginUser()

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

.seealso: PetscFunctionBegin, PetscFunctionReturn()

M*/
#define PetscFunctionBeginHot do {                           \
    PetscStackPushNoCheck(PETSC_FUNCTION_NAME,1,PETSC_TRUE); \
    PetscRegister__FUNCT__();                                \
  } while (0)

/*MC
   PetscFunctionBeginUser - First executable line of user provided PETSc routine

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
      Final line of PETSc functions should be PetscFunctionReturn(0) except for main().

      Not available in Fortran

      This is identical to PetscFunctionBegin except it labels the routine as a user
      routine instead of as a PETSc library routine.

   Level: intermediate

.seealso: PetscFunctionReturn(), PetscFunctionBegin, PetscFunctionBeginHot

M*/
#define PetscFunctionBeginUser do {                           \
    PetscStackPushNoCheck(PETSC_FUNCTION_NAME,2,PETSC_FALSE); \
    PetscRegister__FUNCT__();                                 \
  } while (0)

#define PetscStackPush(n)       do {        \
    PetscStackPushNoCheck(n,0,PETSC_FALSE); \
    CHKMEMQ;                                \
  } while (0)

#define PetscStackPop           do {             \
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

.seealso: PetscFunctionBegin()

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
#define PetscStackCall(name,routine)
#define PetscStackCallStandard(name,...)
#else
/*
    PetscStackCall - Calls an external library routine or user function after pushing the name of the routine on the stack.

   Input Parameters:
+   name - string that gives the name of the function being called
-   routine - actual call to the routine, including ierr = and CHKERRQ(ierr);

   Note: Often one should use PetscStackCallStandard() instead. This routine is intended for external library routines that DO NOT return error codes

   Developer Note: this is so that when a user or external library routine results in a crash or corrupts memory, they get blamed instead of PETSc.

*/
#define PetscStackCall(name,routine) do { PetscStackPush(name);routine;PetscStackPop; } while (0)

/*
    PetscStackCallStandard - Calls an external library routine after pushing the name of the routine on the stack.

   Input Parameters:
+   func-  name of the routine
-   args - arguments to the routine surrounded by ()

   Notes:
    This is intended for external package routines that return error codes. Use PetscStackCall() for those that do not.

   Developer Note: this is so that when an external packge routine results in a crash or corrupts memory, they get blamed instead of PETSc.

*/
#define PetscStackCallStandard(func,...) do {                                                  \
    PetscStackPush(PetscStringize(func));                                                      \
    PetscErrorCode __ierr = func(__VA_ARGS__);                                                 \
    PetscStackPop;                                                                             \
    PetscCheck(!__ierr,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in %s(): error code %d",PetscStringize(func),__ierr); \
  } while (0)
#endif /* PETSC_CLANG_STATIC_ANALYZER */

#endif

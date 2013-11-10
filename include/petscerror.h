/*
    Contains all error handling interfaces for PETSc.
*/
#if !defined(__PETSCERROR_H)
#define __PETSCERROR_H

/*
   Defines the function where the compiled source is located; used
   in printing error messages. This is defined here in case the user
   does not declare it.
*/
#if !defined(__FUNCT__)
#define __FUNCT__ "User provided function"
#endif

/*
     These are the generic error codes. These error codes are used
     many different places in the PETSc source code. The string versions are
     at src/sys/error/err.c any changes here must also be made there
     These are also define in include/finclude/petscerror.h any CHANGES here
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

#define PETSC_ERR_INT_OVERFLOW     84   /* should always be one less then the smallest value */

#define PETSC_ERR_FLOP_COUNT       90
#define PETSC_ERR_NOT_CONVERGED    91  /* solver did not converge */
#define PETSC_ERR_MAX_VALUE        92  /* this is always the one more than the largest error code */

#define PetscStringizeArg(a) #a
#define PetscStringize(a) PetscStringizeArg(a)

#if defined(PETSC_USE_ERRORCHECKING)

/*MC
   SETERRQ - Macro that is called when an error has been detected,

   Synopsis:
   #include "petscsys.h"
   PetscErrorCode SETERRQ(MPI_Comm comm,PetscErrorCode errorcode,char *message)

   Not Collective

   Input Parameters:
+  errorcode - nonzero error code, see the list of standard error codes in include/petscerror.h
-  message - error message

  Level: beginner

   Notes:
    Once the error handler is called the calling function is then returned from with the given error code.

    See SETERRQ1(), SETERRQ2(), SETERRQ3() for versions that take arguments

    In Fortran MPI_Abort() is always called

    Experienced users can set the error handler with PetscPushErrorHandler().

   Concepts: error^setting condition

.seealso: PetscTraceBackErrorHandler(), PetscPushErrorHandler(), PetscError(), CHKERRQ(), CHKMEMQ, SETERRQ1(), SETERRQ2(), SETERRQ3()
M*/
#define SETERRQ(comm,n,s)              return PetscError(comm,__LINE__,PETSC_FUNCTION_NAME,__FILE__,n,PETSC_ERROR_INITIAL,s)

/*MC
   SETERRQ1 - Macro that is called when an error has been detected,

   Synopsis:
   #include "petscsys.h"
   PetscErrorCode SETERRQ1(MPI_Comm comm,PetscErrorCode errorcode,char *formatmessage,arg)

   Not Collective

   Input Parameters:
+  errorcode - nonzero error code, see the list of standard error codes in include/petscerror.h
.  message - error message in the printf format
-  arg - argument (for example an integer, string or double)

  Level: beginner

   Notes:
    Once the error handler is called the calling function is then returned from with the given error code.

   Experienced users can set the error handler with PetscPushErrorHandler().

   Concepts: error^setting condition

.seealso: PetscTraceBackErrorHandler(), PetscPushErrorHandler(), PetscError(), CHKERRQ(), CHKMEMQ, SETERRQ(), SETERRQ2(), SETERRQ3()
M*/
#define SETERRQ1(comm,n,s,a1)          return PetscError(comm,__LINE__,PETSC_FUNCTION_NAME,__FILE__,n,PETSC_ERROR_INITIAL,s,a1)

/*MC
   SETERRQ2 - Macro that is called when an error has been detected,

   Synopsis:
   #include "petscsys.h"
   PetscErrorCode SETERRQ2(PetscErrorCode errorcode,char *formatmessage,arg1,arg2)

   Not Collective

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

.seealso: PetscTraceBackErrorHandler(), PetscPushErrorHandler(), PetscError(), CHKERRQ(), CHKMEMQ, SETERRQ1(), SETERRQ3()
M*/
#define SETERRQ2(comm,n,s,a1,a2)       return PetscError(comm,__LINE__,PETSC_FUNCTION_NAME,__FILE__,n,PETSC_ERROR_INITIAL,s,a1,a2)

/*MC
   SETERRQ3 - Macro that is called when an error has been detected,

   Synopsis:
   #include "petscsys.h"
   PetscErrorCode SETERRQ3(PetscErrorCode errorcode,char *formatmessage,arg1,arg2,arg3)

   Not Collective

   Input Parameters:
+  errorcode - nonzero error code, see the list of standard error codes in include/petscerror.h
.  message - error message in the printf format
.  arg1 - argument (for example an integer, string or double)
.  arg2 - argument (for example an integer, string or double)
-  arg3 - argument (for example an integer, string or double)

  Level: beginner

   Notes:
    Once the error handler is called the calling function is then returned from with the given error code.

    There are also versions for 4, 5, 6 and 7 arguments.

   Experienced users can set the error handler with PetscPushErrorHandler().

   Concepts: error^setting condition

.seealso: PetscTraceBackErrorHandler(), PetscPushErrorHandler(), PetscError(), CHKERRQ(), CHKMEMQ, SETERRQ1(), SETERRQ2()
M*/
#define SETERRQ3(comm,n,s,a1,a2,a3)    return PetscError(comm,__LINE__,PETSC_FUNCTION_NAME,__FILE__,n,PETSC_ERROR_INITIAL,s,a1,a2,a3)

#define SETERRQ4(comm,n,s,a1,a2,a3,a4) return PetscError(comm,__LINE__,PETSC_FUNCTION_NAME,__FILE__,n,PETSC_ERROR_INITIAL,s,a1,a2,a3,a4)
#define SETERRQ5(comm,n,s,a1,a2,a3,a4,a5)       return PetscError(comm,__LINE__,PETSC_FUNCTION_NAME,__FILE__,n,PETSC_ERROR_INITIAL,s,a1,a2,a3,a4,a5)
#define SETERRQ6(comm,n,s,a1,a2,a3,a4,a5,a6)    return PetscError(comm,__LINE__,PETSC_FUNCTION_NAME,__FILE__,n,PETSC_ERROR_INITIAL,s,a1,a2,a3,a4,a5,a6)
#define SETERRQ7(comm,n,s,a1,a2,a3,a4,a5,a6,a7) return PetscError(comm,__LINE__,PETSC_FUNCTION_NAME,__FILE__,n,PETSC_ERROR_INITIAL,s,a1,a2,a3,a4,a5,a6,a7)
#define SETERRQ8(comm,n,s,a1,a2,a3,a4,a5,a6,a7,a8) return PetscError(comm,__LINE__,PETSC_FUNCTION_NAME,__FILE__,n,PETSC_ERROR_INITIAL,s,a1,a2,a3,a4,a5,a6,a7,a8)
#define SETERRABORT(comm,n,s)     do {PetscError(comm,__LINE__,PETSC_FUNCTION_NAME,__FILE__,n,PETSC_ERROR_INITIAL,s);MPI_Abort(comm,n);} while (0)

/*MC
   CHKERRQ - Checks error code, if non-zero it calls the error handler and then returns

   Synopsis:
   #include "petscsys.h"
   PetscErrorCode CHKERRQ(PetscErrorCode errorcode)

   Not Collective

   Input Parameters:
.  errorcode - nonzero error code, see the list of standard error codes in include/petscerror.h

  Level: beginner

   Notes:
    Once the error handler is called the calling function is then returned from with the given error code.

    Experienced users can set the error handler with PetscPushErrorHandler().

    CHKERRQ(n) is fundamentally a macro replacement for
         if (n) return(PetscError(...,n,...));

    Although typical usage resembles "void CHKERRQ(PetscErrorCode)" as described above, for certain uses it is
    highly inappropriate to use it in this manner as it invokes return(PetscErrorCode). In particular,
    it cannot be used in functions which return(void) or any other datatype.  In these types of functions,
    you can use CHKERRV() which returns without an error code (bad idea since the error is ignored or
         if (n) {PetscError(....); return(YourReturnType);}
    where you may pass back a NULL to indicate an error. You can also call CHKERRABORT(comm,n) to have
    MPI_Abort() returned immediately.

    In Fortran MPI_Abort() is always called

   Concepts: error^setting condition

.seealso: PetscTraceBackErrorHandler(), PetscPushErrorHandler(), PetscError(), SETERRQ(), CHKMEMQ, SETERRQ1(), SETERRQ2(), SETERRQ2()
M*/
#define CHKERRQ(n)             do {if (PetscUnlikely(n)) return PetscError(PETSC_COMM_SELF,__LINE__,PETSC_FUNCTION_NAME,__FILE__,n,PETSC_ERROR_REPEAT," ");} while (0)

#define CHKERRV(n)             do {if (PetscUnlikely(n)) {n = PetscError(PETSC_COMM_SELF,__LINE__,PETSC_FUNCTION_NAME,__FILE__,n,PETSC_ERROR_REPEAT," ");return;}} while(0)
#define CHKERRABORT(comm,n)    do {if (PetscUnlikely(n)) {PetscError(PETSC_COMM_SELF,__LINE__,PETSC_FUNCTION_NAME,__FILE__,n,PETSC_ERROR_REPEAT," ");MPI_Abort(comm,n);}} while (0)
#define CHKERRCONTINUE(n)      do {if (PetscUnlikely(n)) {PetscError(PETSC_COMM_SELF,__LINE__,PETSC_FUNCTION_NAME,__FILE__,n,PETSC_ERROR_REPEAT," ");}} while (0)

#ifdef PETSC_CLANGUAGE_CXX

/*MC
   CHKERRXX - Checks error code, if non-zero it calls the C++ error handler which throws an exception

   Synopsis:
   #include "petscsys.h"
   void CHKERRXX(PetscErrorCode errorcode)

   Not Collective

   Input Parameters:
.  errorcode - nonzero error code, see the list of standard error codes in include/petscerror.h

  Level: beginner

   Notes:
    Once the error handler throws a ??? exception.

    You can use CHKERRV() which returns without an error code (bad idea since the error is ignored)
    or CHKERRABORT(comm,n) to have MPI_Abort() returned immediately.

   Concepts: error^setting condition

.seealso: PetscTraceBackErrorHandler(), PetscPushErrorHandler(), PetscError(), SETERRQ(), CHKERRQ(), CHKMEMQ
M*/
#define CHKERRXX(n)            do {if (PetscUnlikely(n)) {PetscError(PETSC_COMM_SELF,__LINE__,PETSC_FUNCTION_NAME,__FILE__,n,PETSC_ERROR_IN_CXX,0);}} while(0)

#endif

/*MC
   CHKMEMQ - Checks the memory for corruption, calls error handler if any is detected

   Synopsis:
   #include "petscsys.h"
   CHKMEMQ;

   Not Collective

  Level: beginner

   Notes:
    Must run with the option -malloc_debug to enable this option

    Once the error handler is called the calling function is then returned from with the given error code.

    By defaults prints location where memory that is corrupted was allocated.

    Use CHKMEMA for functions that return void

   Concepts: memory corruption

.seealso: PetscTraceBackErrorHandler(), PetscPushErrorHandler(), PetscError(), SETERRQ(), CHKMEMQ, SETERRQ1(), SETERRQ2(), SETERRQ3(),
          PetscMallocValidate()
M*/
#define CHKMEMQ do {PetscErrorCode _7_ierr = PetscMallocValidate(__LINE__,PETSC_FUNCTION_NAME,__FILE__);CHKERRQ(_7_ierr);} while(0)

#define CHKMEMA PetscMallocValidate(__LINE__,PETSC_FUNCTION_NAME,__FILE__)

#else /* PETSC_USE_ERRORCHECKING */

/*
    These are defined to be empty for when error checking is turned off, with ./configure --with-errorchecking=0
*/

#define SETERRQ(c,n,s)
#define SETERRQ1(c,n,s,a1)
#define SETERRQ2(c,n,s,a1,a2)
#define SETERRQ3(c,n,s,a1,a2,a3)
#define SETERRQ4(c,n,s,a1,a2,a3,a4)
#define SETERRQ5(c,n,s,a1,a2,a3,a4,a5)
#define SETERRQ6(c,n,s,a1,a2,a3,a4,a5,a6)
#define SETERRQ7(c,n,s,a1,a2,a3,a4,a5,a6,a7)
#define SETERRQ8(c,n,s,a1,a2,a3,a4,a5,a6,a7,a8)
#define SETERRABORT(comm,n,s)

#define CHKERRQ(n)     ;
#define CHKERRABORT(comm,n) ;
#define CHKERRCONTINUE(n) ;
#define CHKMEMQ        ;

#ifdef PETSC_CLANGUAGE_CXX
#define CHKERRXX(n) ;
#endif

#endif /* PETSC_USE_ERRORCHECKING */

/*E
  PetscErrorType - passed to the PETSc error handling routines indicating if this is the first or a later call to the error handlers

  Level: advanced

  PETSC_ERROR_IN_CXX indicates the error was detected in C++ and an exception should be generated

  Developer Notes: This is currently used to decide when to print the detailed information about the run in PetscTraceBackErrorHandling()

.seealso: PetscError(), SETERRXX()
E*/
typedef enum {PETSC_ERROR_INITIAL=0,PETSC_ERROR_REPEAT=1,PETSC_ERROR_IN_CXX = 2} PetscErrorType;

PETSC_EXTERN PetscErrorCode PetscErrorPrintfInitialize(void);
PETSC_EXTERN PetscErrorCode PetscErrorMessage(int,const char*[],char **);
PETSC_EXTERN PetscErrorCode PetscTraceBackErrorHandler(MPI_Comm,int,const char*,const char*,PetscErrorCode,PetscErrorType,const char*,void*);
PETSC_EXTERN PetscErrorCode PetscIgnoreErrorHandler(MPI_Comm,int,const char*,const char*,PetscErrorCode,PetscErrorType,const char*,void*);
PETSC_EXTERN PetscErrorCode PetscEmacsClientErrorHandler(MPI_Comm,int,const char*,const char*,PetscErrorCode,PetscErrorType,const char*,void*);
PETSC_EXTERN PetscErrorCode PetscMPIAbortErrorHandler(MPI_Comm,int,const char*,const char*,PetscErrorCode,PetscErrorType,const char*,void*);
PETSC_EXTERN PetscErrorCode PetscAbortErrorHandler(MPI_Comm,int,const char*,const char*,PetscErrorCode,PetscErrorType,const char*,void*);
PETSC_EXTERN PetscErrorCode PetscAttachDebuggerErrorHandler(MPI_Comm,int,const char*,const char*,PetscErrorCode,PetscErrorType,const char*,void*);
PETSC_EXTERN PetscErrorCode PetscReturnErrorHandler(MPI_Comm,int,const char*,const char*,PetscErrorCode,PetscErrorType,const char*,void*);
PETSC_EXTERN PetscErrorCode PetscError(MPI_Comm,int,const char*,const char*,PetscErrorCode,PetscErrorType,const char*,...);
PETSC_EXTERN PetscErrorCode PetscPushErrorHandler(PetscErrorCode (*handler)(MPI_Comm,int,const char*,const char*,PetscErrorCode,PetscErrorType,const char*,void*),void*);
PETSC_EXTERN PetscErrorCode PetscPopErrorHandler(void);
PETSC_EXTERN PetscErrorCode PetscSignalHandlerDefault(int,void*);
PETSC_EXTERN PetscErrorCode PetscPushSignalHandler(PetscErrorCode (*)(int,void *),void*);
PETSC_EXTERN PetscErrorCode PetscPopSignalHandler(void);
PETSC_EXTERN PetscErrorCode PetscCheckPointerSetIntensity(PetscInt);

/*MC
    PetscErrorPrintf - Prints error messages.

   Synopsis:
    #include "petscsys.h"
     PetscErrorCode (*PetscErrorPrintf)(const char format[],...);

    Not Collective

    Input Parameters:
.   format - the usual printf() format string

   Options Database Keys:
+    -error_output_stdout - cause error messages to be printed to stdout instead of the
         (default) stderr
-    -error_output_none to turn off all printing of error messages (does not change the way the
          error is handled.)

   Notes: Use
$     PetscErrorPrintf = PetscErrorPrintfNone; to turn off all printing of error messages (does not change the way the
$                        error is handled.) and
$     PetscErrorPrintf = PetscErrorPrintfDefault; to turn it back on
$        of you can use your own function

          Use
     PETSC_STDERR = FILE* obtained from a file open etc. to have stderr printed to the file.
     PETSC_STDOUT = FILE* obtained from a file open etc. to have stdout printed to the file.

          Use
      PetscPushErrorHandler() to provide your own error handler that determines what kind of messages to print

   Level: developer

    Fortran Note:
    This routine is not supported in Fortran.

    Concepts: error messages^printing
    Concepts: printing^error messages

.seealso: PetscFPrintf(), PetscSynchronizedPrintf(), PetscHelpPrintf(), PetscPrintf(), PetscErrorHandlerPush(), PetscVFPrintf(), PetscHelpPrintf()
M*/
PETSC_EXTERN PetscErrorCode (*PetscErrorPrintf)(const char[],...);

typedef enum {PETSC_FP_TRAP_OFF=0,PETSC_FP_TRAP_ON=1} PetscFPTrap;
PETSC_EXTERN PetscErrorCode PetscSetFPTrap(PetscFPTrap);
PETSC_EXTERN PetscErrorCode PetscFPTrapPush(PetscFPTrap);
PETSC_EXTERN PetscErrorCode PetscFPTrapPop(void);

/*  Linux functions CPU_SET and others don't work if sched.h is not included before
    including pthread.h. Also, these functions are active only if either _GNU_SOURCE
    or __USE_GNU is not set (see /usr/include/sched.h and /usr/include/features.h), hence
    set these first.
*/
#if defined(PETSC_HAVE_PTHREADCLASSES) || defined (PETSC_HAVE_OPENMP)
#if defined(PETSC_HAVE_SCHED_H)
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <sched.h>
#endif
#include <pthread.h>
#endif

/*
     This code is for managing thread local global variables. Each of Linux, Microsoft WINDOWS, OpenMP, and Apple OS X have
   different ways to indicate this. On OS X each thread local global is accessed by using a pthread_key_t for that variable.
   Thus we have functions for creating destroying and using the keys. Except for OS X these access functions merely directly 
   acess the thread local variable.
*/

#if defined(PETSC_HAVE_PTHREADCLASSES) && !defined(PETSC_PTHREAD_LOCAL)
typedef pthread_key_t PetscThreadKey;
/* Get the value associated with key */
PETSC_STATIC_INLINE void* PetscThreadLocalGetValue(PetscThreadKey key)
{
  return pthread_getspecific(key);
}

/* Set the value for key */
PETSC_STATIC_INLINE void PetscThreadLocalSetValue(PetscThreadKey *key,void* value)
{
  pthread_setspecific(*key,(void*)value);
}

/* Create pthread thread local key */
PETSC_STATIC_INLINE void PetscThreadLocalRegister(PetscThreadKey *key)
{
  pthread_key_create(key,NULL);
}

/* Delete pthread thread local key */
PETSC_STATIC_INLINE void PetscThreadLocalDestroy(PetscThreadKey key)
{
  pthread_key_delete(key);
}
#else
typedef void* PetscThreadKey;
PETSC_STATIC_INLINE void* PetscThreadLocalGetValue(PetscThreadKey key)
{
  return key;
}

PETSC_STATIC_INLINE void PetscThreadLocalSetValue(PetscThreadKey *key,void* value)
{
  *key = value;
}

PETSC_STATIC_INLINE void PetscThreadLocalRegister(PETSC_UNUSED PetscThreadKey *key)
{
}

PETSC_STATIC_INLINE void PetscThreadLocalDestroy(PETSC_UNUSED PetscThreadKey key)
{
}
#endif

/*
      Allows the code to build a stack frame as it runs
*/

#define PETSCSTACKSIZE 64

typedef struct  {
  const char      *function[PETSCSTACKSIZE];
  const char      *file[PETSCSTACKSIZE];
        int       line[PETSCSTACKSIZE];
        PetscBool petscroutine[PETSCSTACKSIZE];
        int       currentsize;
        int       hotdepth;
} PetscStack;

#if defined(PETSC_HAVE_PTHREADCLASSES)
#if defined(PETSC_PTHREAD_LOCAL)
PETSC_EXTERN PETSC_PTHREAD_LOCAL PetscStack *petscstack;
#else
PETSC_EXTERN PetscThreadKey petscstack;
#endif
#elif defined(PETSC_HAVE_OPENMP)
PETSC_EXTERN PetscStack *petscstack;
#pragma omp threadprivate(petscstack)
#else
PETSC_EXTERN PetscStack *petscstack;
#endif

PETSC_EXTERN PetscErrorCode PetscStackCopy(PetscStack*,PetscStack*);
PETSC_EXTERN PetscErrorCode PetscStackPrint(PetscStack*,FILE* fp);

#if defined(PETSC_USE_DEBUG)
PETSC_STATIC_INLINE PetscBool PetscStackActive(void)
{
  return(PetscThreadLocalGetValue(petscstack) ? PETSC_TRUE : PETSC_FALSE);
}

/* Stack handling is based on the following two "NoCheck" macros.  These should only be called directly by other error
 * handling macros.  We record the line of the call, which may or may not be the location of the definition.  But is at
 * least more useful than "unknown" because it can distinguish multiple calls from the same function.
 */

#define PetscStackPushNoCheck(funct,petsc_routine,hot)                        \
  do {                                                                        \
    PetscStack* petscstackp;                                                  \
    PetscStackAMSTakeAccess();                                                \
    petscstackp = (PetscStack*)PetscThreadLocalGetValue(petscstack);          \
    if (petscstackp && (petscstackp->currentsize < PETSCSTACKSIZE)) {         \
      petscstackp->function[petscstackp->currentsize]  = funct;               \
      petscstackp->file[petscstackp->currentsize]      = __FILE__;            \
      petscstackp->line[petscstackp->currentsize]      = __LINE__;            \
      petscstackp->petscroutine[petscstackp->currentsize] = petsc_routine;    \
      petscstackp->currentsize++;                                             \
    }                                                                         \
    if (petscstackp) {                                                        \
      petscstackp->hotdepth += (hot || petscstackp->hotdepth);                \
    }                                                                         \
    PetscStackAMSGrantAccess();                                               \
  } while (0)

#define PetscStackPopNoCheck                                            \
  do {PetscStack* petscstackp;                                          \
    PetscStackAMSTakeAccess();                                          \
    petscstackp = (PetscStack*)PetscThreadLocalGetValue(petscstack);    \
    if (petscstackp && petscstackp->currentsize > 0) {                  \
      petscstackp->currentsize--;                                       \
      petscstackp->function[petscstackp->currentsize]  = 0;             \
      petscstackp->file[petscstackp->currentsize]      = 0;             \
      petscstackp->line[petscstackp->currentsize]      = 0;             \
      petscstackp->petscroutine[petscstackp->currentsize] = PETSC_FALSE;\
    }                                                                   \
    if (petscstackp) {                                                  \
      petscstackp->hotdepth = PetscMax(petscstackp->hotdepth-1,0);      \
    }                                                                   \
    PetscStackAMSGrantAccess();                                         \
  } while (0)

/*MC
   PetscFunctionBegin - First executable line of each PETSc function
        used for error handling.

   Synopsis:
   #include "petscsys.h"
   void PetscFunctionBegin;

   Not Collective

   Usage:
.vb
     int something;

     PetscFunctionBegin;
.ve

   Notes:
     Not available in Fortran

   Level: developer

.seealso: PetscFunctionReturn()

.keywords: traceback, error handling
M*/
#define PetscFunctionBegin do {                                        \
    PetscStackPushNoCheck(PETSC_FUNCTION_NAME,PETSC_TRUE,PETSC_FALSE); \
    PetscCheck__FUNCT__();                                             \
    PetscRegister__FUNCT__();                                          \
  } while (0)

/*MC
   PetscFunctionBeginHot - Substitute for PetscFunctionBegin to be used in functions that are called in
   performance-critical circumstances.  Use of this function allows for lighter profiling by default.

   Synopsis:
   #include "petscsys.h"
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

.keywords: traceback, error handling
M*/
#define PetscFunctionBeginHot do {                                     \
    PetscStackPushNoCheck(PETSC_FUNCTION_NAME,PETSC_TRUE,PETSC_TRUE);  \
    PetscCheck__FUNCT__();                                             \
    PetscRegister__FUNCT__();                                          \
  } while (0)

/*MC
   PetscFunctionBeginUser - First executable line of user provided PETSc routine

   Synopsis:
   #include "petscsys.h"
   void PetscFunctionBeginUser;

   Not Collective

   Usage:
.vb
     int something;

     PetscFunctionBegin;
.ve

   Notes:
     Not available in Fortran

   Level: intermediate

.seealso: PetscFunctionReturn(), PetscFunctionBegin, PetscFunctionBeginHot

.keywords: traceback, error handling
M*/
#define PetscFunctionBeginUser                                          \
  do {                                                                  \
    PetscStackPushNoCheck(PETSC_FUNCTION_NAME,PETSC_FALSE,PETSC_FALSE); \
    PetscCheck__FUNCT__();                                              \
    PetscRegister__FUNCT__();                                           \
  } while (0)


#if defined(PETSC_SERIALIZE_FUNCTIONS)
#include <petsc-private/petscfptimpl.h>
/*
   Registers the current function into the global function pointer to function name table

   Have to fix this to handle errors but cannot return error since used in PETSC_VIEWER_DRAW_() etc
*/
#define PetscRegister__FUNCT__() do { \
  static PetscBool __chked = PETSC_FALSE; \
  if (!__chked) {\
  void *ptr; PetscDLSym(NULL,__FUNCT__,&ptr);\
  __chked = PETSC_TRUE;\
  }} while (0)
#else
#define PetscRegister__FUNCT__()
#endif

#define PetscCheck__FUNCT__() do { PetscBool _sc1,_sc2;                  \
    PetscStrcmpNoError(PETSC_FUNCTION_NAME,__FUNCT__,&_sc1);\
    PetscStrcmpNoError(__FUNCT__,"User provided function",&_sc2);\
    if (!_sc1 && !_sc2) { \
      printf("%s:%d: __FUNCT__=\"%s\" does not agree with %s=\"%s\"\n",__FILE__,__LINE__,__FUNCT__,PetscStringize(PETSC_FUNCTION_NAME),PETSC_FUNCTION_NAME); \
    }                                                                   \
  } while (0)

#define PetscStackPush(n) \
  do {                                                                  \
    PetscStackPushNoCheck(n,PETSC_FALSE,PETSC_FALSE);                   \
    CHKMEMQ;                                                            \
  } while (0)

#define PetscStackPop                           \
    do {                                        \
      CHKMEMQ;                                  \
      PetscStackPopNoCheck;                     \
    } while (0)

/*MC
   PetscFunctionReturn - Last executable line of each PETSc function
        used for error handling. Replaces return()

   Synopsis:
   #include "petscsys.h"
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

.keywords: traceback, error handling
M*/
#define PetscFunctionReturn(a) \
  do {                                                                \
    PetscStackPopNoCheck;                                             \
    return(a);} while (0)

#define PetscFunctionReturnVoid() \
  do {                                                                \
    PetscStackPopNoCheck;                                             \
    return;} while (0)

#else

PETSC_STATIC_INLINE PetscBool PetscStackActive(void) {return PETSC_FALSE;}
#define PetscStackPushNoCheck(funct,petsc_routine,hot) do {} while (0)
#define PetscStackPopNoCheck                           do {} while (0)
#define PetscFunctionBegin
#define PetscFunctionBeginUser
#define PetscFunctionBeginHot
#define PetscFunctionReturn(a)    return(a)
#define PetscFunctionReturnVoid() return
#define PetscStackPop             CHKMEMQ
#define PetscStackPush(f)         CHKMEMQ

#endif

/*
    PetscStackCall - Calls an external library routine or user function after pushing the name of the routine on the stack.

   Input Parameters:
+   name - string that gives the name of the function being called
-   routine - actual call to the routine, including ierr = and CHKERRQ(ierr);

   Note: Often one should use PetscStackCallStandard() instead. This routine is intended for external library routines that DO NOT return error codes

   Developer Note: this is so that when a user or external library routine results in a crash or corrupts memory, they get blamed instead of PETSc.



*/
#define PetscStackCall(name,routine) do { PetscStackPush(name);routine;PetscStackPop; } while(0)

/*
    PetscStackCallStandard - Calls an external library routine after pushing the name of the routine on the stack.

   Input Parameters:
+   func-  name of the routine
-   args - arguments to the routine surrounded by ()

   Notes: This is intended for external package routines that return error codes. Use PetscStackCall() for those that do not.

   Developer Note: this is so that when an external packge routine results in a crash or corrupts memory, they get blamed instead of PETSc.

*/
#define PetscStackCallStandard(func,args) do {                        \
    PetscStackPush(#func);ierr = func args;PetscStackPop; if (ierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in %s()",#func); \
  } while (0)

PETSC_EXTERN PetscErrorCode PetscStackCreate(void);
PETSC_EXTERN PetscErrorCode PetscStackView(FILE*);
PETSC_EXTERN PetscErrorCode PetscStackDestroy(void);

#endif

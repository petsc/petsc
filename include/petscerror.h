/*
    Contains all error handling interfaces for PETSc.
*/
#if !defined(__PETSCERROR_H)
#define __PETSCERROR_H

PETSC_EXTERN_CXX_BEGIN

/*
   Defines the directory where the compiled source is located; used
   in printing error messages. Each makefile has an entry 
   LOCDIR	  =  thedirectory
   and bmake/common_variables includes in CCPPFLAGS -D__SDIR__=${LOCDIR}
   which is a flag passed to the C/C++ compilers. This declaration below
   is only needed if some code is compiled without the -D__SDIR__
*/
#if !defined(__INSDIR__)
#define __INSDIR__ "unknowndirectory/"
#endif

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

#define PETSC_ERR_FLOP_COUNT       90
#define PETSC_ERR_MAX_VALUE        91  /* this is always the one more than the largest error code */

#define PetscStringizeArg(a) #a
#define PetscStringize(a) PetscStringizeArg(a)
#define __SDIR__ PetscStringize(__INSDIR__)

#if defined(PETSC_USE_ERRORCHECKING)

/*MC
   SETERRQ - Macro that is called when an error has been detected, 

   Synopsis:
   PetscErrorCode SETERRQ(PetscErrorCode errorcode,char *message)

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
#define SETERRQ(n,s)              {return PetscError(__LINE__,__FUNCT__,__FILE__,__SDIR__,n,1,s);}

/*MC
   SETERRQ1 - Macro that is called when an error has been detected, 

   Synopsis:
   PetscErrorCode SETERRQ1(PetscErrorCode errorcode,char *formatmessage,arg)

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
#define SETERRQ1(n,s,a1)          {return PetscError(__LINE__,__FUNCT__,__FILE__,__SDIR__,n,1,s,a1);}

/*MC
   SETERRQ2 - Macro that is called when an error has been detected, 

   Synopsis:
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
#define SETERRQ2(n,s,a1,a2)       {return PetscError(__LINE__,__FUNCT__,__FILE__,__SDIR__,n,1,s,a1,a2);}

/*MC
   SETERRQ3 - Macro that is called when an error has been detected, 

   Synopsis:
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
#define SETERRQ3(n,s,a1,a2,a3)    {return PetscError(__LINE__,__FUNCT__,__FILE__,__SDIR__,n,1,s,a1,a2,a3);}

#define SETERRQ4(n,s,a1,a2,a3,a4) {return PetscError(__LINE__,__FUNCT__,__FILE__,__SDIR__,n,1,s,a1,a2,a3,a4);}
#define SETERRQ5(n,s,a1,a2,a3,a4,a5)       {return PetscError(__LINE__,__FUNCT__,__FILE__,__SDIR__,n,1,s,a1,a2,a3,a4,a5);}
#define SETERRQ6(n,s,a1,a2,a3,a4,a5,a6)    {return PetscError(__LINE__,__FUNCT__,__FILE__,__SDIR__,n,1,s,a1,a2,a3,a4,a5,a6);}
#define SETERRQ7(n,s,a1,a2,a3,a4,a5,a6,a7) {return PetscError(__LINE__,__FUNCT__,__FILE__,__SDIR__,n,1,s,a1,a2,a3,a4,a5,a6,a7);}
#define SETERRABORT(comm,n,s)     {PetscError(__LINE__,__FUNCT__,__FILE__,__SDIR__,n,1,s);MPI_Abort(comm,n);}

/*MC
   CHKERRQ - Checks error code, if non-zero it calls the error handler and then returns

   Synopsis:
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
    where you may pass back a PETSC_NULL to indicate an error. You can also call CHKERRABORT(comm,n) to have
    MPI_Abort() returned immediately.

    In Fortran MPI_Abort() is always called

   Concepts: error^setting condition

.seealso: PetscTraceBackErrorHandler(), PetscPushErrorHandler(), PetscError(), SETERRQ(), CHKMEMQ, SETERRQ1(), SETERRQ2(), SETERRQ2()
M*/
#define CHKERRQ(n)             if (PetscUnlikely(n)) {return PetscError(__LINE__,__FUNCT__,__FILE__,__SDIR__,n,0," ");}

#define CHKERRV(n)             if (PetscUnlikely(n)) {n = PetscError(__LINE__,__FUNCT__,__FILE__,__SDIR__,n,0," ");return;}
#define CHKERRABORT(comm,n)    if (PetscUnlikely(n)) {PetscError(__LINE__,__FUNCT__,__FILE__,__SDIR__,n,0," ");MPI_Abort(comm,n);}
#define CHKERRCONTINUE(n)      if (PetscUnlikely(n)) {PetscError(__LINE__,__FUNCT__,__FILE__,__SDIR__,n,0," ");}

#ifdef PETSC_CLANGUAGE_CXX

/*MC
   CHKERRXX - Checks error code, if non-zero it calls the C++ error handler which throws an exception

   Synopsis:
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
#define CHKERRXX(n)            if (PetscUnlikely(n)) {PetscErrorCxx(__LINE__,__FUNCT__,__FILE__,__SDIR__,n,0);}

#endif

/*MC
   CHKMEMQ - Checks the memory for corruption, calls error handler if any is detected

   Synopsis:
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
#define CHKMEMQ {PetscErrorCode _7_ierr = PetscMallocValidate(__LINE__,__FUNCT__,__FILE__,__SDIR__);CHKERRQ(_7_ierr);}

#define CHKMEMA {PetscMallocValidate(__LINE__,__FUNCT__,__FILE__,__SDIR__);}

#if defined(PETSC_UNDERSCORE_CHKERR)
extern  PetscErrorCode __gierr;
#define _   __gierr = 
#define ___  CHKERRQ(__gierr);
#endif

#define               PETSC_EXCEPTIONS_MAX  256
extern PetscErrorCode PetscErrorUncatchable[PETSC_EXCEPTIONS_MAX];
extern PetscInt       PetscErrorUncatchableCount;
extern PetscErrorCode PetscExceptions[PETSC_EXCEPTIONS_MAX];
extern PetscInt       PetscExceptionsCount;

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscExceptionPush(PetscErrorCode);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscExceptionPop(PetscErrorCode);

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscErrorSetCatchable(PetscErrorCode,PetscTruth);
EXTERN PetscTruth PETSC_DLLEXPORT PetscErrorIsCatchable(PetscErrorCode);
/*MC
   PetscExceptionCaught - Indicates if a specific exception zierr was caught.

   Synopsis:
     PetscTruth PetscExceptionCaught(PetscErrorCode xierr,PetscErrorCode zierr);

   Not Collective

  Input Parameters:
  + xierr - error code returned from PetscExceptionTry1() or other PETSc routine
  - zierr - error code you want it to be

  Level: advanced

   Notes:
    PETSc must not be configured using the option --with-errorchecking=0 for this to work

    Use PetscExceptionValue() to see if an error code is being "tried"

  Concepts: exceptions, exception handling

.seealso: PetscTraceBackErrorHandler(), PetscPushErrorHandler(), PetscError(), SETERRQ(), CHKMEMQ, SETERRQ1(), SETERRQ2(), SETERRQ3(), 
          CHKERRQ(), PetscExceptionTry1(), PetscExceptionValue()
M*/
PETSC_STATIC_INLINE PetscTruth PetscExceptionCaught(PetscErrorCode xierr,PetscErrorCode zierr) 
{
  PetscInt i;
  if (xierr != zierr) return PETSC_FALSE;
  for (i=0; i<PetscErrorUncatchableCount; i++) {
    if (PetscErrorUncatchable[i] == zierr) {
      return PETSC_FALSE;
    }
  }
  return PETSC_TRUE;
}          

/*MC
   PetscExceptionValue - Indicates if the error code is one that is currently being tried

   Synopsis:
     PetscTruth PetscExceptionValue(PetscErrorCode xierr);

   Not Collective

  Input Parameters:
  . xierr - error code 

  Level: developer

   Notes:
    PETSc must not be configured using the option --with-errorchecking=0 for this to work

    Use PetscExceptionCaught() to see if the current error code is EXACTLY the one you want

  Concepts: exceptions, exception hanlding

.seealso: PetscTraceBackErrorHandler(), PetscPushErrorHandler(), PetscError(), SETERRQ(), CHKMEMQ, SETERRQ1(), SETERRQ2(), SETERRQ3(), 
          CHKERRQ(), PetscExceptionTry1(), PetscExceptionCaught()
M*/
PETSC_STATIC_INLINE PetscTruth PetscExceptionValue(PetscErrorCode zierr) 
{
  PetscInt i;
  for (i=0; i<PetscExceptionsCount; i++) {
    if (PetscExceptions[i] == zierr) {
      return PETSC_TRUE;
    }
  }
  return PETSC_FALSE;
}

/*MC
   PetscExceptionTry1 - Runs the routine, causing a particular error code to be treated as an exception,
         rather than an error. That is if that error code is treated the program returns to this level,
         but does not call the error handlers

   Synopsis:
     PetscErrorCode PetscExceptionTry1(PetscErrorCode routine(....),PetscErrorCode);

   Not Collective

  Level: advanced

   No Fortran Equivalent (see PetscExceptionPush() for Fortran)

   Notes:
    PETSc must not be configured using the option --with-errorchecking=0 for this to work

  Note: In general, the outer most try on an exception is the one that will be caught (that is trys down in 
        PETSc code will not usually handle an exception that was issued above). See SNESSolve() for an example
        of how the local try is ignored if a higher (in the stack) one is also in effect.

  Concepts: exceptions, exception hanlding

.seealso: PetscTraceBackErrorHandler(), PetscPushErrorHandler(), PetscError(), SETERRQ(), CHKMEMQ, SETERRQ1(), SETERRQ2(), SETERRQ3(), 
          CHKERRQ(), PetscExceptionCaught(), PetscExceptionPush(), PetscExceptionPop()
M*/
extern PetscErrorCode PetscExceptionTmp,PetscExceptionTmp1;
#define PetscExceptionTry1(a,b) (PetscExceptionTmp1 = PetscExceptionPush(b)) ? PetscExceptionTmp1 : (PetscExceptionTmp1 = a, (PetscExceptionTmp = PetscExceptionPop(b)) ? PetscExceptionTmp : PetscExceptionTmp1)

/*
   Used by PetscExceptionTrySync(). Returns zierr on ALL processes in comm iff xierr is zierr on at least one process and zero on all others.
*/
PETSC_STATIC_INLINE PetscErrorCode PetscExceptionTrySync_Private(MPI_Comm comm,PetscErrorCode xierr,PetscErrorCode zierr) 
{
  PetscReal      in[2],out[2];
  PetscErrorCode ierr;

  if (xierr != zierr) return xierr;

  in[0] = xierr;
  in[1] = 0.0;   /* dummy value */

  ierr = MPI_Allreduce(in,out,2,MPIU_REAL,0,comm); if (ierr) {;}
  return xierr;
}          

/*MC
   PetscExceptionTrySyncNorm - Runs the routine, causing a particular error code to be treated as an exception,
         rather than an error. That is if that error code is treated the program returns to this level,
         but does not call the error handlers

   Synopsis:
     PetscExceptionTrySyncNorm(MPI_Comm comm,PetscErrorCode routine(....),PetscErrorCode);

     Collective on Comm

  Level: advanced

   Notes: This synchronizes the error code across all processes in the communicator IF the code matches PetscErrorCode. The next
     call with an MPI_Reduce()/MPI_Allreduce() MUST be VecNorm() [We can added VecDot() and maybe others as needed].

    PETSc must not be configured using the option --with-errorchecking=0 for this to work

  Note: In general, the outer most try on an exception is the one that will be caught (that is trys down in 
        PETSc code will not usually handle an exception that was issued above). See SNESSolve() for an example
        of how the local try is ignored if a higher (in the stack) one is also in effect.

  Concepts: exceptions, exception hanlding

.seealso: PetscTraceBackErrorHandler(), PetscPushErrorHandler(), PetscError(), SETERRQ(), CHKMEMQ, SETERRQ1(), SETERRQ2(), SETERRQ3(), 
          CHKERRQ(), PetscExceptionCaught(), PetscExceptionPush(), PetscExceptionPop(), PetscExceptionTry1()
M*/
#define PetscExceptionTrySyncNorm(comm,a,b) (PetscExceptionTmp = PetscExceptionPush(b)) ? PetscExceptionTmp : \
                                        (PetscExceptionTmp = a , PetscExceptionPop(b),PetscExceptionTrySyncNorm_Private(comm,PetscExceptionTmp,b))

#else

/* 
    These are defined to be empty for when error checking is turned off, with config/configure.py --with-errorchecking=0
*/

#define SETERRQ(n,s) ;
#define SETERRQ1(n,s,a1) ;
#define SETERRQ2(n,s,a1,a2) ;
#define SETERRQ3(n,s,a1,a2,a3) ;
#define SETERRQ4(n,s,a1,a2,a3,a4) ;
#define SETERRQ5(n,s,a1,a2,a3,a4,a5) ;
#define SETERRQ6(n,s,a1,a2,a3,a4,a5,a6) ;
#define SETERRABORT(comm,n,s) ;

#define CHKERRQ(n)     ;
#define CHKERRABORT(comm,n) ;
#define CHKERRCONTINUE(n) ;
#define CHKMEMQ        ;

#ifdef PETSC_CLANGUAGE_CXX
#define CHKERRXX(n) ;
#endif

#if !defined(PETSC_SKIP_UNDERSCORE_CHKERR)
#define _   
#define ___  
#endif 

#define PetscExceptionPush(a)                0
#define PetscExceptionPop(a)                 0
#define PetscErrorSetCatchable(a,b)          0
#define PetscErrorIsCatchable(a)             PETSC_FALSE

#define PetscExceptionCaught(a,b)            PETSC_FALSE
#define PetscExceptionValue(a)               PETSC_FALSE
#define PetscExceptionTry1(a,b)              a
#define PetscExceptionTrySyncNorm(comm,a,b)  a

#endif

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscErrorPrintfInitialize(void);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscErrorMessage(int,const char*[],char **);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscTraceBackErrorHandler(int,const char*,const char*,const char*,PetscErrorCode,int,const char*,void*);
#if defined(PETSC_CLANGUAGE_CXX) && !defined(PETSC_USE_EXTERN_CXX)
#include <sstream>
EXTERN void           PETSC_DLLEXPORT PetscTraceBackErrorHandlerCxx(int,const char *,const char *,const char *,PetscErrorCode,int, std::ostringstream&);
#endif
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscIgnoreErrorHandler(int,const char*,const char*,const char*,PetscErrorCode,int,const char*,void*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscEmacsClientErrorHandler(int,const char*,const char*,const char*,PetscErrorCode,int,const char*,void*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscMPIAbortErrorHandler(int,const char*,const char*,const char*,PetscErrorCode,int,const char*,void*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscAbortErrorHandler(int,const char*,const char*,const char*,PetscErrorCode,int,const char*,void*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscAttachDebuggerErrorHandler(int,const char*,const char*,const char*,PetscErrorCode,int,const char*,void*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscReturnErrorHandler(int,const char*,const char*,const char*,PetscErrorCode,int,const char*,void*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscError(int,const char*,const char*,const char*,PetscErrorCode,int,const char*,...) PETSC_PRINTF_FORMAT_CHECK(7,8);
EXTERN void           PETSC_DLLEXPORT PetscErrorCxx(int,const char*,const char*,const char*,PetscErrorCode,int);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscPushErrorHandler(PetscErrorCode (*handler)(int,const char*,const char*,const char*,PetscErrorCode,int,const char*,void*),void*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscPopErrorHandler(void);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDefaultSignalHandler(int,void*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscPushSignalHandler(PetscErrorCode (*)(int,void *),void*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscPopSignalHandler(void);

typedef enum {PETSC_FP_TRAP_OFF=0,PETSC_FP_TRAP_ON=1} PetscFPTrap;
EXTERN PetscErrorCode PETSC_DLLEXPORT  PetscSetFPTrap(PetscFPTrap);

/*
      Allows the code to build a stack frame as it runs
*/
#if defined(PETSC_USE_DEBUG)

#define PETSCSTACKSIZE 15

typedef struct  {
  const char *function[PETSCSTACKSIZE];
  const char *file[PETSCSTACKSIZE];
  const char *directory[PETSCSTACKSIZE];
        int  line[PETSCSTACKSIZE];
        int currentsize;
} PetscStack;

extern PETSC_DLLEXPORT PetscStack *petscstack;
EXTERN PetscErrorCode PETSC_DLLEXPORT  PetscStackCopy(PetscStack*,PetscStack*);
EXTERN PetscErrorCode PETSC_DLLEXPORT  PetscStackPrint(PetscStack*,FILE* fp);

#define PetscStackActive (petscstack != 0)


/*MC
   PetscFunctionBegin - First executable line of each PETSc function
        used for error handling.

   Synopsis:
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
#define PetscFunctionBegin \
  {\
   if (petscstack && (petscstack->currentsize < PETSCSTACKSIZE)) {    \
    petscstack->function[petscstack->currentsize]  = __FUNCT__; \
    petscstack->file[petscstack->currentsize]      = __FILE__; \
    petscstack->directory[petscstack->currentsize] = __SDIR__; \
    petscstack->line[petscstack->currentsize]      = __LINE__; \
    petscstack->currentsize++; \
  }}

#define PetscStackPush(n) \
  {if (petscstack && (petscstack->currentsize < PETSCSTACKSIZE)) {    \
    petscstack->function[petscstack->currentsize]  = n; \
    petscstack->file[petscstack->currentsize]      = "unknown"; \
    petscstack->directory[petscstack->currentsize] = "unknown"; \
    petscstack->line[petscstack->currentsize]      = 0; \
    petscstack->currentsize++; \
  }}

#define PetscStackPop \
  {if (petscstack && petscstack->currentsize > 0) {     \
    petscstack->currentsize--; \
    petscstack->function[petscstack->currentsize]  = 0; \
    petscstack->file[petscstack->currentsize]      = 0; \
    petscstack->directory[petscstack->currentsize] = 0; \
    petscstack->line[petscstack->currentsize]      = 0; \
  }};

/*MC
   PetscFunctionReturn - Last executable line of each PETSc function
        used for error handling. Replaces return()

   Synopsis:
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
  {\
  PetscStackPop; \
  return(a);}

#define PetscFunctionReturnVoid() \
  {\
  PetscStackPop; \
  return;}


#else

#define PetscFunctionBegin 
#define PetscFunctionReturn(a)  return(a)
#define PetscFunctionReturnVoid() return
#define PetscStackPop 
#define PetscStackPush(f) 
#define PetscStackActive        0

#endif

EXTERN PetscErrorCode PETSC_DLLEXPORT  PetscStackCreate(void);
EXTERN PetscErrorCode PETSC_DLLEXPORT  PetscStackView(PetscViewer);
EXTERN PetscErrorCode PETSC_DLLEXPORT  PetscStackDestroy(void);
EXTERN PetscErrorCode PETSC_DLLEXPORT  PetscStackPublish(void);
EXTERN PetscErrorCode PETSC_DLLEXPORT  PetscStackDepublish(void);


PETSC_EXTERN_CXX_END
#endif

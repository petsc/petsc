/* $Id: petscerror.h,v 1.59 2001/09/07 20:13:16 bsmith Exp $ */
/*
    Contains all error handling code for PETSc.
*/
#if !defined(__PETSCERROR_H)
#define __PETSCERROR_H

#include "petsc.h"

#if defined(PETSC_HAVE_AMS)
#include "ams.h"
#endif

/*
   Defines the directory where the compiled source is located; used
   in printing error messages. Each makefile has an entry 
   LOCDIR	  =  thedirectory
   and bmake/common_variables includes in CCPPFLAGS -D__SDIR__='"${LOCDIR}"'
   which is a flag passed to the C/C++ compilers.
*/
#if !defined(__SDIR__)
#define __SDIR__ "unknowndirectory/"
#endif

/*
   Defines the function where the compiled source is located; used 
   in printing error messages.
*/
#if !defined(__FUNCT__)
#define __FUNCT__ "User provided function"
#endif

/* 
     These are the generic error codes. These error codes are used
     many different places in the PETSc source code.

*/
#define PETSC_ERR_MEM             55   /* unable to allocate requested memory */
#define PETSC_ERR_MEM_MALLOC_0    85   /* cannot malloc zero size */
#define PETSC_ERR_SUP             56   /* no support for requested operation */
#define PETSC_ERR_SIG             59   /* signal received */
#define PETSC_ERR_FP              72   /* floating point exception */
#define PETSC_ERR_COR             74   /* corrupted PETSc object */
#define PETSC_ERR_LIB             76   /* error in library called by PETSc */
#define PETSC_ERR_PLIB            77   /* PETSc library generated inconsistent data */
#define PETSC_ERR_MEMC            78   /* memory corruption */
#define PETSC_ERR_MAX_ITER        82   /* Maximum iterations reached */

#define PETSC_ERR_ARG_SIZ         60   /* nonconforming object sizes used in operation */
#define PETSC_ERR_ARG_IDN         61   /* two arguments not allowed to be the same */
#define PETSC_ERR_ARG_WRONG       62   /* wrong argument (but object probably ok) */
#define PETSC_ERR_ARG_CORRUPT     64   /* null or corrupted PETSc object as argument */
#define PETSC_ERR_ARG_OUTOFRANGE  63   /* input argument, out of range */
#define PETSC_ERR_ARG_BADPTR      68   /* invalid pointer argument */
#define PETSC_ERR_ARG_NOTSAMETYPE 69   /* two args must be same object type */
#define PETSC_ERR_ARG_NOTSAMECOMM 80   /* two args must be same communicators */
#define PETSC_ERR_ARG_WRONGSTATE  73   /* object in argument is in wrong state, e.g. unassembled mat */
#define PETSC_ERR_ARG_INCOMP      75   /* two arguments are incompatible */

#define PETSC_ERR_FILE_OPEN       65   /* unable to open file */
#define PETSC_ERR_FILE_READ       66   /* unable to read from file */
#define PETSC_ERR_FILE_WRITE      67   /* unable to write to file */
#define PETSC_ERR_FILE_UNEXPECTED 79   /* unexpected data in file */

#define PETSC_ERR_KSP_BRKDWN      70   /* break down in a Krylov method */

#define PETSC_ERR_MAT_LU_ZRPVT    71   /* detected a zero pivot during LU factorization */
#define PETSC_ERR_MAT_CH_ZRPVT    81   /* detected a zero pivot during Cholesky factorization */

#define PETSC_ERR_MESH_NULL_ELEM  84   /* Element had no interior */

#define PETSC_ERR_DISC_SING_JAC   83   /* Singular element Jacobian */

#if defined(PETSC_USE_DEBUG)

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
#define SETERRQ(n,s)              {return PetscError(__LINE__,__FUNCT__,__FILE__,__SDIR__,n,1,s);}

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
#define SETERRQ1(n,s,a1)          {return PetscError(__LINE__,__FUNCT__,__FILE__,__SDIR__,n,1,s,a1);}

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
#define SETERRQ2(n,s,a1,a2)       {return PetscError(__LINE__,__FUNCT__,__FILE__,__SDIR__,n,1,s,a1,a2);}

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
#define SETERRQ3(n,s,a1,a2,a3)    {return PetscError(__LINE__,__FUNCT__,__FILE__,__SDIR__,n,1,s,a1,a2,a3);}

#define SETERRQ4(n,s,a1,a2,a3,a4) {return PetscError(__LINE__,__FUNCT__,__FILE__,__SDIR__,n,1,s,a1,a2,a3,a4);}
#define SETERRQ5(n,s,a1,a2,a3,a4,a5)       {return PetscError(__LINE__,__FUNCT__,__FILE__,__SDIR__,n,1,s,a1,a2,a3,a4,a5);}
#define SETERRQ6(n,s,a1,a2,a3,a4,a5,a6)    {return PetscError(__LINE__,__FUNCT__,__FILE__,__SDIR__,n,1,s,a1,a2,a3,a4,a5,a6);}
#define SETERRQ7(n,s,a1,a2,a3,a4,a5,a6,a7) {return PetscError(__LINE__,__FUNCT__,__FILE__,__SDIR__,n,1,s,a1,a2,a3,a4,a5,a6,a7);}
#define SETERRABORT(comm,n,s)     {PetscError(__LINE__,__FUNCT__,__FILE__,__SDIR__,n,1,s);MPI_Abort(comm,n);}

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
#define CHKERRQ(n)             if (n) {return PetscError(__LINE__,__FUNCT__,__FILE__,__SDIR__,n,0," ");}

#define CHKERRABORT(comm,n)    if (n) {PetscError(__LINE__,__FUNCT__,__FILE__,__SDIR__,n,0," ");MPI_Abort(comm,n);}
#define CHKERRCONTINUE(n)      if (n) {PetscError(__LINE__,__FUNCT__,__FILE__,__SDIR__,n,0," ");}

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
#define CHKMEMQ {int _7_ierr = PetscTrValid(__LINE__,__FUNCT__,__FILE__,__SDIR__);CHKERRQ(_7_ierr);}

#if !defined(PETSC_SKIP_UNDERSCORE_CHKERR)
extern  int __gierr;
#define _   __gierr = 
#define ___  CHKERRQ(__gierr);
#endif

#else
#define SETERRQ(n,s) ;
#define SETERRQ1(n,s,a1) ;
#define SETERRQ2(n,s,a1,a2) ;
#define SETERRQ3(n,s,a1,a2,a3) ;
#define SETERRQ4(n,s,a1,a2,a3,a4) ;
#define SETERRABORT(comm,n,s) ;

#define CHKERRQ(n)     ;
#define CHKERRABORT(comm,n) ;
#define CHKERRCONTINUE(n) ;

#define CHKMEMQ        ;

#if !defined(PETSC_SKIP_UNDERSCORE_CHKERR)
#define _   
#define ___  
#endif 

#endif

EXTERN int PetscErrorMessage(int,char**,char **);
EXTERN int PetscTraceBackErrorHandler(int,char*,char*,char*,int,int,char*,void*);
EXTERN int PetscIgnoreErrorHandler(int,char*,char*,char*,int,int,char*,void*);
EXTERN int PetscEmacsClientErrorHandler(int,char*,char*,char*,int,int,char*,void*);
EXTERN int PetscStopErrorHandler(int,char*,char*,char*,int,int,char*,void*);
EXTERN int PetscAbortErrorHandler(int,char*,char*,char*,int,int,char*,void*);
EXTERN int PetscAttachDebuggerErrorHandler(int,char*,char*,char*,int,int,char*,void*); 
EXTERN int PetscError(int,char*,char*,char*,int,int,char*,...) PETSC_PRINTF_FORMAT_CHECK(7,8);
EXTERN int PetscPushErrorHandler(int (*handler)(int,char*,char*,char*,int,int,char*,void*),void*);
EXTERN int PetscPopErrorHandler(void);
EXTERN int PetscDefaultSignalHandler(int,void*);
EXTERN int PetscPushSignalHandler(int (*)(int,void *),void*);
EXTERN int PetscPopSignalHandler(void);

typedef enum {PETSC_FP_TRAP_OFF=0,PETSC_FP_TRAP_ON=1} PetscFPTrap;
EXTERN int PetscSetFPTrap(PetscFPTrap);

/*
      Allows the code to build a stack frame as it runs
*/
#if defined(PETSC_USE_STACK)

#define PETSCSTACKSIZE 15

typedef struct  {
  char *function[PETSCSTACKSIZE];
  char *file[PETSCSTACKSIZE];
  char *directory[PETSCSTACKSIZE];
  int  line[PETSCSTACKSIZE];
  int  currentsize;
} PetscStack;

extern PetscStack *petscstack;
EXTERN int PetscStackCopy(PetscStack*,PetscStack*);
EXTERN int PetscStackPrint(PetscStack*,FILE* fp);

#define PetscStackActive (petscstack != 0)

#if !defined(PETSC_HAVE_AMS)

/*MC
   PetscFunctionBegin - First executable line of each PETSc function
        used for error handling.

   Synopsis:
   void PetscFunctionBegin;

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

/*
    Duplicate Code for when the ALICE Memory Snooper (AMS)
  is being used. When PETSC_HAVE_AMS is defined.

     stack_mem is the AMS memory that contains fields for the 
               number of stack frames and names of the stack frames
*/

extern AMS_Memory stack_mem;
extern int        stack_err;

#define PetscFunctionBegin \
  {\
   if (petscstack && (petscstack->currentsize < PETSCSTACKSIZE)) {    \
    if (!(stack_mem < 0)) stack_err = AMS_Memory_take_access(stack_mem);\
    petscstack->function[petscstack->currentsize]  = __FUNCT__; \
    petscstack->file[petscstack->currentsize]      = __FILE__; \
    petscstack->directory[petscstack->currentsize] = __SDIR__; \
    petscstack->line[petscstack->currentsize]      = __LINE__; \
    petscstack->currentsize++; \
    if (!(stack_mem < 0)) stack_err = AMS_Memory_grant_access(stack_mem);\
  }}

#define PetscStackPush(n) \
  {if (petscstack && (petscstack->currentsize < PETSCSTACKSIZE)) {    \
    if (!(stack_mem < 0)) stack_err = AMS_Memory_take_access(stack_mem);\
    petscstack->function[petscstack->currentsize]  = n; \
    petscstack->file[petscstack->currentsize]      = "unknown"; \
    petscstack->directory[petscstack->currentsize] = "unknown"; \
    petscstack->line[petscstack->currentsize]      = 0; \
    petscstack->currentsize++; \
    if (!(stack_mem < 0)) stack_err = AMS_Memory_grant_access(stack_mem);\
  }}

#define PetscStackPop \
  {if (petscstack && petscstack->currentsize > 0) {     \
    if (!(stack_mem < 0)) stack_err = AMS_Memory_take_access(stack_mem);\
    petscstack->currentsize--; \
    petscstack->function[petscstack->currentsize]  = 0; \
    petscstack->file[petscstack->currentsize]      = 0; \
    petscstack->directory[petscstack->currentsize] = 0; \
    petscstack->line[petscstack->currentsize]      = 0; \
    if (!(stack_mem < 0)) stack_err = AMS_Memory_grant_access(stack_mem);\
  }};

#define PetscFunctionReturn(a) \
  {\
  PetscStackPop; \
  return(a);}

#define PetscFunctionReturnVoid() \
  {\
  PetscStackPop; \
  return;}


#endif

#else

#define PetscFunctionBegin 
#define PetscFunctionReturn(a)  return(a)
#define PetscFunctionReturnVoid() return()
#define PetscStackPop 
#define PetscStackPush(f) 
#define PetscStackActive        0

#endif

EXTERN int PetscStackCreate(void);
EXTERN int PetscStackView(PetscViewer);
EXTERN int PetscStackDestroy(void);
EXTERN int PetscStackPublish(void);
EXTERN int PetscStackDepublish(void);


#endif










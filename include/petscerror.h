/* $Id: petscerror.h,v 1.13 1997/05/20 03:00:06 bsmith Exp bsmith $ */
/*
    Contains all error handling code for PETSc.
*/
#if !defined(__PETSCERROR_H)
#define __PETSCERROR_H

#include "petsc.h"

/*
   Defines the directory where the compiled source is located; used
   in printing error messages. __SDIR__ is usually defined in the makefile.
*/
#if !defined(__SDIR__)
#define __SDIR__ "unknowndirectory/"
#endif

/*
   Defines the function where the compiled source is located; used 
   in printing error messages.
*/
#if !defined(__FUNC__)
#define __FUNC__ "unknownfunction"
#endif

/* 
     These are the generic error codes. The same error code is used in
     many different places in the code.

     In addition, each specific error in the code has an error
     message: a specific, unique error code.  (The specific error
     code is not yet in use; these will be generated automatically and
     embed an integer into the PetscError() calls. For non-English
     error messages that integer will be extracted and used to look up the
     appropriate error message in the local language from a file.)

*/
#define PETSC_ERR_MEM             55   /* unable to allocate requested memory */
#define PETSC_ERR_SUP             56   /* no support for requested operation */
#define PETSC_ERR_SIG             59   /* signal received */
#define PETSC_ERR_FP              72   /* floating point exception */

#define PETSC_ERR_ARG_SIZ         60   /* nonconforming object sizes used in operation */
#define PETSC_ERR_ARG_IDN         61   /* two arguments not allowed to be the same */
#define PETSC_ERR_ARG_WRONG       62   /* wrong object (but object probably ok) */
#define PETSC_ERR_ARG_CORRUPT     64   /* null or corrupted PETSc object as argument */
#define PETSC_ERR_ARG_OUTOFRANGE  63   /* input argument, out of range */
#define PETSC_ERR_ARG_BADPTR      68   /* invalid pointer argument */
#define PETSC_ERR_ARG_NOTSAMETYPE 69   /* two args must be same object type */
#define PETSC_ERR_ARG_WRONGSTATE  73   /* object in argument is in wrong state, e.g. unassembled mat */

#define PETSC_ERR_FILE_OPEN       65   /* unable to open file */
#define PETSC_ERR_FILE_READ       66   /* unable to read from file */
#define PETSC_ERR_FILE_WRITE      67   /* unable to write to file */

#define PETSC_ERR_KSP_BRKDWN      70   /* Break down in a Krylov method */

#define PETSC_ERR_MAT_LU_ZRPVT    71   /* Detected a zero pivot during LU factorization */
#define PETSC_ERR_MAT_CH_ZRPVT    71   /* Detected a zero pivot during Cholesky factorization */

#if defined(USE_PETSC_DEBUG)
#define SETERRQ(n,p,s) {return PetscError(__LINE__,__FUNC__,__FILE__,__SDIR__,n,p,s);}
#define SETERRA(n,p,s) {int _ierr = PetscError(__LINE__,__FUNC__,__FILE__,__SDIR__,n,p,s);\
                          MPI_Abort(PETSC_COMM_WORLD,_ierr);}
#define CHKERRQ(n)     {if (n) SETERRQ(n,0,(char *)0);}
#define CHKERRA(n)     {if (n) SETERRA(n,0,(char *)0);}
#define CHKPTRQ(p)     if (!p) SETERRQ(PETSC_ERR_MEM,0,(char*)0);
#define CHKPTRA(p)     if (!p) SETERRA(PETSC_ERR_MEM,0,(char*)0);
#else
#define SETERRQ(n,p,s) ;
#define SETERRA(n,p,s) ;
#define CHKERRQ(n)     ;
#define CHKERRA(n)     ;
#define CHKPTRQ(p)     ;
#define CHKPTRA(p)     ;
#endif

extern int PetscTraceBackErrorHandler(int,char*,char*,char*,int,int,char*,void*);
extern int PetscStopErrorHandler(int,char*,char*,char*,int,int,char*,void*);
extern int PetscAbortErrorHandler(int,char*,char*,char*,int,int,char*,void* );
extern int PetscAttachDebuggerErrorHandler(int,char*,char*,char*,int,int,char*,void*); 
extern int PetscError(int,char*,char*,char*,int,int,char*);
extern int PetscPushErrorHandler(int (*handler)(int,char*,char*,char*,int,int,char*,void*),void*);
extern int PetscPopErrorHandler();

extern int PetscDefaultSignalHandler(int,void*);
extern int PetscPushSignalHandler(int (*)(int,void *),void*);
extern int PetscPopSignalHandler();
#define PETSC_FP_TRAP_OFF    0
#define PETSC_FP_TRAP_ON     1
extern int PetscSetFPTrap(int);
extern int PetscInitializeNans(Scalar*,int);
extern int PetscInitializeLargeInts(int *,int);

/*
      Allows the code to build a stack frame as it runs
*/
#if defined(USE_PETSC_STACK)

typedef struct  {
  char *function;
  char *file;
  char *directory;
  int  line;
} PetscStack;

extern int        petscstacksize_max;
extern int        petscstacksize;
extern PetscStack *petscstack;

#define PetscFunctionBegin \
  {if (petscstack && (petscstacksize < petscstacksize_max)) {    \
    petscstack[petscstacksize].function  = __FUNC__; \
    petscstack[petscstacksize].file      = __FILE__; \
    petscstack[petscstacksize].directory = __SDIR__;  \
    petscstack[petscstacksize].line      = __LINE__; \
    petscstacksize++; \
  }}

#define PetscFunctionReturn(a) \
  {if (petscstack && petscstacksize > 0) {    \
    petscstacksize--; \
  } \
  return(a);}

#else

#define PetscFunctionBegin 
#define PetscFunctionReturn(a)  return(a)

#endif

extern int PetscStackCreate(int);
extern int PetscStackView(Viewer);
extern int PetscStackDestroy();
extern int PetscStackPop();

#endif

/* $Id: petscerror.h,v 1.48 2000/05/10 16:44:25 bsmith Exp bsmith $ */
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
#if !defined(__FUNC__)
#define __FUNC__ "unknownfunction"
#endif

/* 
     These are the generic error codes. These error codes are used
     many different places in the PETSc source code.

     In addition, each specific error in the code has an error
     message: a specific, unique error code.  (The specific error
     code is not yet in use; these will be generated automatically and
     embed an integer into the PetscError() calls. For non-English
     error messages, that integer will be extracted and used to look up the
     appropriate error message in the local language from a file.)

*/
#define PETSC_ERR_MEM             55   /* unable to allocate requested memory */
#define PETSC_ERR_SUP             56   /* no support for requested operation */
#define PETSC_ERR_SIG             59   /* signal received */
#define PETSC_ERR_FP              72   /* floating point exception */
#define PETSC_ERR_COR             74   /* corrupted PETSc object */
#define PETSC_ERR_LIB             76   /* error in library called by PETSc */
#define PETSC_ERR_PLIB            77   /* PETSc library generated inconsistent data */
#define PETSC_ERR_MEMC            78   /* memory corruption */

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

#if defined(PETSC_USE_DEBUG)
#define SETERRA(n,p,s)     {int _ierr = PetscError(__LINE__,__FUNC__,__FILE__,__SDIR__,n,p,s);\
                           MPI_Abort(PETSC_COMM_WORLD,_ierr);}
#define SETERRA1(n,p,s,a1) {int _ierr = PetscError(__LINE__,__FUNC__,__FILE__,__SDIR__,n,p,s,a1);\
                           MPI_Abort(PETSC_COMM_WORLD,_ierr);}
#define SETERRQ(n,p,s)              {return PetscError(__LINE__,__FUNC__,__FILE__,__SDIR__,n,p,s);}
#define SETERRQ1(n,p,s,a1)          {return PetscError(__LINE__,__FUNC__,__FILE__,__SDIR__,n,p,s,a1);}
#define SETERRQ2(n,p,s,a1,a2)       {return PetscError(__LINE__,__FUNC__,__FILE__,__SDIR__,n,p,s,a1,a2);}
#define SETERRQ3(n,p,s,a1,a2,a3)    {return PetscError(__LINE__,__FUNC__,__FILE__,__SDIR__,n,p,s,a1,a2,a3);}
#define SETERRQ4(n,p,s,a1,a2,a3,a4) {return PetscError(__LINE__,__FUNC__,__FILE__,__SDIR__,n,p,s,a1,a2,a3,a4);}

#define CHKERRQ(n)     {if (n) SETERRQ(n,0,(char *)0);}
#define CHKERRA(n)     {if (n) SETERRA(n,0,(char *)0);}
#define CHKPTRQ(p)     if (!p) SETERRQ(PETSC_ERR_MEM,0,(char*)0);
#define CHKPTRA(p)     if (!p) SETERRA(PETSC_ERR_MEM,0,(char*)0);

#define CHKMEMQ {int __ierr = PetscTrValid(__LINE__,__FUNC__,__FILE__,__SDIR__);CHKERRQ(__ierr);}
#define CHKMEMA {int __ierr = PetscTrValid(__LINE__,__FUNC__,__FILE__,__SDIR__);CHKERRA(__ierr);}

#if !defined(PETSC_SKIP_UNDERSCORE_CHKERR)
extern  int __gierr;
#define _   __gierr = 
#define ___  CHKERRA(__gierr);
#define ____ CHKERRQ(__gierr);
#endif

#else
#define SETERRA(n,p,s) ;
#define SETERRA1(n,p,s,b) ;
#define SETERRQ(n,p,s) ;
#define SETERRQ1(n,p,s,a1) ;
#define SETERRQ2(n,p,s,a1,a2) ;
#define SETERRQ3(n,p,s,a1,a2,a3) ;
#define SETERRQ4(n,p,s,a1,a2,a3,a4) ;

#define CHKERRQ(n)     ;
#define CHKERRA(n)     ;
#define CHKPTRQ(p)     ;
#define CHKPTRA(p)     ;

#define CHKMEMQ        ;
#define CHKMEMA        ;

#if !defined(PETSC_SKIP_UNDERSCORE_CHKERR)
#define _   
#define ___  
#define ____
#endif 

#endif

EXTERN int PetscErrorMessage(int,char**);
EXTERN int PetscTraceBackErrorHandler(int,char*,char*,char*,int,int,char*,void*);
EXTERN int PetscEmacsClientErrorHandler(int,char*,char*,char*,int,int,char*,void*);
EXTERN int PetscStopErrorHandler(int,char*,char*,char*,int,int,char*,void*);
EXTERN int PetscAbortErrorHandler(int,char*,char*,char*,int,int,char*,void*);
EXTERN int PetscAttachDebuggerErrorHandler(int,char*,char*,char*,int,int,char*,void*); 
EXTERN int PetscError(int,char*,char*,char*,int,int,char*,...);
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

#define PetscFunctionBegin \
  {\
   if (petscstack && (petscstack->currentsize < PETSCSTACKSIZE)) {    \
    petscstack->function[petscstack->currentsize]  = __FUNC__; \
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

#define PetscFunctionReturn(a) \
  {\
  PetscStackPop; \
  return(a);}

#define PetscFunctionReturnVoid() \
  {\
  PetscStackPop;}

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
    petscstack->function[petscstack->currentsize]  = __FUNC__; \
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
  PetscStackPop;}


#endif

#else

#define PetscFunctionBegin 
#define PetscFunctionReturn(a)  return(a)
#define PetscFunctionReturnVoid()
#define PetscStackPop 
#define PetscStackPush(f) 
#define PetscStackActive        0

#endif

EXTERN int PetscStackCreate(void);
EXTERN int PetscStackView(Viewer);
EXTERN int PetscStackDestroy(void);
EXTERN int PetscStackPublish(void);
EXTERN int PetscStackDepublish(void);


#endif



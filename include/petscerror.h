/* $Id: petscerror.h,v 1.1 1996/12/07 20:22:53 bsmith Exp bsmith $ */
/*
    Contains all the error handling code for PETSc.
*/
#if !defined(__PETSCERROR_H)
#define __PETSCERROR_H

/*
   Defines the directory where the compiled source is located; used
   in print error messages. __DIR__ is usually defined in the makefile.
*/
#if !defined(__DIR__)
#define __DIR__ 0
#endif

/* 
       These are the generic error codes. The same error code is used in
     many different places in the code.

       In addition, each specific error in the code has an error
     message: a text string imbedded in the code in English; and a
     specific eroror code.  (The specific error code is not yet in
     use, those will be generated automatically and embed a integer
     charactor string at the begin of the error message string. For
     non-English error messages that integer will be extracted from
     the English error message and used to look up the appropriate
     error message in the local language from a file.)

*/
/*
   The next four error flags are being replaced
*/

#define PETSC_ERR_ARG          57   /* bad input argument */
#define PETSC_ERR_OBJ          58   /* wrong, null or corrupted PETSc object */
#define PETSC_ERR_IDN          61   /* two arguments not allowed to be the same */
#define PETSC_ERR_SIZ          60   /* nonconforming object sizes used in operation */


#define PETSC_ERR_MEM             55   /* unable to allocate requested memory */
#define PETSC_ERR_SUP             56   /* no support yet for requested operation */
#define PETSC_ERR_SIG             59   /* signal received */

#define PETSC_ERR_ARG_SIZ         60   /* nonconforming object sizes used in operation */
#define PETSC_ERR_ARG_IDN         61   /* two arguments not allowed to be the same */
#define PETSC_ERR_ARG_WRONG       62   /* wrong object (but object probably ok) */
#define PETSC_ERR_ARG_CORRUPT     64   /* null or corrupted PETSc object as argument */
#define PETSC_ERR_ARG_OUTOFRANGE  63   /* input argument, out of range */

#define PETSC_ERR_KSP_BRKDWN      70   /* Break down in a Krylov method */
#define PETSC_ERR_MAT_LU_ZRPVT    71   /* Detected a zero pivot during LU factorization */
#define PETSC_ERR_MAT_CH_ZRPVT    71   /* Detected a zero pivot during Cholesky factorization */

#if defined(PETSC_DEBUG)
#define SETERRQ(n,s)   {return PetscError(__LINE__,__DIR__,__FILE__,n,s);}
#define SETERRA(n,s)   {int _ierr = PetscError(__LINE__,__DIR__,__FILE__,n,s);\
                          MPI_Abort(PETSC_COMM_WORLD,_ierr);}
#define CHKERRQ(n)     {if (n) SETERRQ(n,(char *)0);}
#define CHKERRA(n)     {if (n) SETERRA(n,(char *)0);}
#define CHKPTRQ(p)     if (!p) SETERRQ(PETSC_ERR_MEM,(char*)0);
#define CHKPTRA(p)     if (!p) SETERRA(PETSC_ERR_MEM,(char*)0);
#else
#define SETERRQ(n,s)   {return PetscError(__LINE__,__DIR__,__FILE__,n,s);}
#define SETERRA(n,s)   {int _ierr = PetscError(__LINE__,__DIR__,__FILE__,n,s);\
                          MPI_Abort(PETSC_COMM_WORLD,_ierr);}
#define CHKERRQ(n)     {if (n) SETERRQ(n,(char *)0);}
#define CHKERRA(n)     {if (n) SETERRA(n,(char *)0);}
#define CHKPTRQ(p)     if (!p) SETERRQ(PETSC_ERR_MEM,(char*)0);
#define CHKPTRA(p)     if (!p) SETERRA(PETSC_ERR_MEM,(char*)0);
#endif

extern int PetscTraceBackErrorHandler(int,char*,char*,int,char*,void*);
extern int PetscStopErrorHandler(int,char*,char*,int,char*,void*);
extern int PetscAbortErrorHandler(int,char*,char*,int,char*,void* );
extern int PetscAttachDebuggerErrorHandler(int,char*,char*,int,char*,void*); 
extern int PetscError(int,char*,char*,int,char*);
extern int PetscPushErrorHandler(int (*handler)(int,char*,char*,int,char*,void*),void*);
extern int PetscPopErrorHandler();

extern int PetscDefaultSignalHandler(int,void*);
extern int PetscPushSignalHandler(int (*)(int,void *),void*);
extern int PetscPopSignalHandler();
#define PETSC_FP_TRAP_OFF    0
#define PETSC_FP_TRAP_ON     1
extern int PetscSetFPTrap(int);

#endif

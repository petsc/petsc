/* $Id: petsc.h,v 1.37 1995/07/12 04:02:40 curfman Exp curfman $ */

#if !defined(__PETSC_PACKAGE)
#define __PETSC_PACKAGE

#define PETSC_VERSION_NUMBER "PETSc Version 2.0.Beta.5 Released ?, 1995."

#include <stdio.h>
#if defined(PARCH_sun4)
int fprintf(FILE*,char*,...);
int printf(char*,...);
int fflush(FILE*);
int fclose(FILE*);
#endif

/* MPI interface */
#include "mpi.h"
#include "mpiu.h"

#if defined(PETSC_COMPLEX)
/* work around for bug in alpha g++ compiler */
#if defined(PARCH_alpha) 
#define hypot(a,b) (double) sqrt((a)*(a)+(b)*(b)) 
/* extern double hypot(double,double); */
#endif
#include <complex.h>
#define PETSCREAL(a) real(a)
#define Scalar       complex
#else
#define PETSCREAL(a) a
#define Scalar       double
#endif

extern void *(*PetscMalloc)(unsigned int,int,char*);
extern int  (*PetscFree)(void *,int,char*);
#define PETSCMALLOC(a)       (*PetscMalloc)(a,__LINE__,__FILE__)
#define PETSCFREE(a)         (*PetscFree)(a,__LINE__,__FILE__)
extern int  PetscSetMalloc(void *(*)(unsigned int,int,char*),
                           int (*)(void *,int,char*));
extern int  Trdump(FILE *);
extern int  TrGetMaximumAllocated(double*);

#define PETSCNEW(A)         (A*) PETSCMALLOC(sizeof(A))
#define PETSCMEMCPY(a,b,n)   memcpy((char*)(a),(char*)(b),n)
#define PETSCMEMSET(a,b,n)   memset((char*)(a),(int)(b),n)
#include <memory.h>

/*  Macros for error checking */
#if !defined(__DIR__)
#define __DIR__ 0
#endif
#if defined(PETSC_DEBUG)
#define SETERRQ(n,s)     {return PetscError(__LINE__,__DIR__,__FILE__,s,n);}
#define SETERRA(n,s)    \
                {int _ierr = PetscError(__LINE__,__DIR__,__FILE__,s,n);\
                 MPI_Abort(MPI_COMM_WORLD,_ierr);}
#define CHKERRQ(n)       {if (n) SETERRQ(n,(char *)0);}
#define CHKERRA(n)      {if (n) SETERRA(n,(char *)0);}
#define CHKPTRQ(p)       if (!p) SETERRQ(1,"No memory");
#define CHKPTRA(p)      if (!p) SETERRA(1,"No memory");
#else
#define SETERRQ(n,s)     {return PetscError(__LINE__,__DIR__,__FILE__,s,n);}
#define SETERRA(n,s)    \
                {int _ierr = PetscError(__LINE__,__DIR__,__FILE__,s,n);\
                 MPI_Abort(MPI_COMM_WORLD,_ierr);}
#define CHKERRQ(n)       {if (n) SETERRQ(n,(char *)0);}
#define CHKERRA(n)      {if (n) SETERRA(n,(char *)0);}
#define CHKPTRQ(p)       if (!p) SETERRQ(1,"No memory");
#define CHKPTRA(p)      if (!p) SETERRA(1,"No memory");
#endif

typedef struct _PetscObject* PetscObject;
#define PETSC_COOKIE         0x12121212
#define PETSC_DECIDE         -1

typedef enum { PETSC_FALSE, PETSC_TRUE } PetscTruth;

#include "viewer.h"
#include "options.h"

extern int PetscInitialize(int*,char***,char*,char*);
extern int PetscFinalize();

extern int PetscObjectDestroy(PetscObject);
extern int PetscObjectGetComm(PetscObject,MPI_Comm *comm);
extern int PetscObjectSetName(PetscObject,char*);
extern int PetscObjectGetName(PetscObject,char**);

extern int PetscDefaultErrorHandler(int,char*,char*,char*,int,void*);
extern int PetscAbortErrorHandler(int,char*,char*,char*,int,void* );
extern int PetscAttachDebuggerErrorHandler(int,char*,char*,char*,int,void*); 
extern int PetscError(int,char*,char*,char*,int);
extern int PetscPushErrorHandler(int 
                         (*handler)(int,char*,char*,char*,int,void*),void* );
extern int PetscPopErrorHandler();

extern int PetscSetDebugger(char *,int,char *);
extern int PetscAttachDebugger();

extern int PetscDefaultSignalHandler(int,void*);
extern int PetscPushSignalHandler(int (*)(int,void *),void*);
extern int PetscPopSignalHandler();
extern int PetscSetFPTrap(int);
#define FP_TRAP_OFF    0
#define FP_TRAP_ON     1
#define FP_TRAP_ALWAYS 2


#if defined(PARCH_cray) || defined(PARCH_NCUBE) || defined(PARCH_t3d)
#define FORTRANCAPS
#elif !defined(PARCH_rs6000) && !defined(PARCH_NeXT) && !defined(PARCH_hpux)
#define FORTRANUNDERSCORE
#endif

/* Global flop counter */
extern double _TotalFlops;
#if defined(PETSC_LOG)
#define PLogFlops(n) {_TotalFlops += n;}
#else
#define PLogFlops(n)
#endif 

/*M
   PLogFlops - Adds floating point operations to the global counter.

   Input Parameter:
.  f - flop counter

   Synopsis:
   PLogFlops(int f)

   Notes:
   A global counter logs all PETSc flop counts.  The user can use
   PLogFlops() to increment this counter to include flops for the 
   application code.  

   PETSc automatically logs library events if the code has been
   compiled with -DPETSC_LOG (which is the default), and -log,
   -log_summary, or -log_all are specified.  PLogFlops() is
   intended for logging user flops to supplement this PETSc
   information.

    Example of Usage:
$     #define USER_EVENT 75
$     PLogEventRegister(USER_EVENT,"User event");
$     PLogEventBegin(USER_EVENT,0,0,0,0);
$     [code segment to monitor]
$     PLogFlops(user_flops)
$     PLogEventEnd(USER_EVENT,0,0,0,0);

.seealso:  PLogEventRegister(), PLogEventBegin(), PLogEventEnd()

.keywords:  Petsc, log, flops, floating point operations
M*/

extern int PLogPrint(MPI_Comm,FILE *);
extern int PLogBegin();
extern int PLogAllBegin();
extern int PLogDump(char*);

#endif

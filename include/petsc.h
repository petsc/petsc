/* $Id: petsc.h,v 1.114 1996/04/18 14:47:13 bsmith Exp bsmith $ */
/*
   PETSc header file, included in all PETSc programs.
*/
#if !defined(__PETSC_PACKAGE)
#define __PETSC_PACKAGE

#define PETSC_VERSION_NUMBER "PETSc Version 2.0.13, Released April 18, 1996."

#include <stdio.h>

#include "mpi.h"

#if defined(PETSC_COMPLEX)
#include <complex.h>
extern  MPI_Datatype      MPIU_COMPLEX;
#define MPIU_SCALAR       MPIU_COMPLEX
#define PetscReal(a)      real(a)
#define PetscAbsScalar(a) abs(a)
#define Scalar            complex
#else
#define MPIU_SCALAR       MPI_DOUBLE
#define PetscReal(a)      a
#define PetscAbsScalar(a) ( ((a)<0.0)   ? -(a) : (a) )
#define Scalar            double
#endif

extern void *(*PetscMalloc)(unsigned int,int,char*);
extern int  (*PetscFree)(void *,int,char*);
extern int  PetscSetMalloc(void *(*)(unsigned int,int,char*),int (*)(void *,int,char*));
#define PetscMalloc(a)       (*PetscMalloc)(a,__LINE__,__FILE__)
#define PetscNew(A)          (A*) PetscMalloc(sizeof(A))
#define PetscFree(a)         (*PetscFree)(a,__LINE__,__FILE__)

extern int  PetscTrDump(FILE *);
extern int  PetscTrSpace( double *, double *,double *);
extern int  PetscTrValid(int ,char*);
extern int  PetscTrDebugLevel(int);

extern void  PetscMemcpy(void *,void *,int);
extern void  PetscMemzero(void *,int);
extern int   PetscMemcmp(void*, void*, int);
extern int   PetscStrlen(char *);
extern int   PetscStrcmp(char *,char *);
extern int   PetscStrncmp(char *,char *,int );
extern void  PetscStrcpy(char *,char *);
extern void  PetscStrcat(char *,char *);
extern void  PetscStrncat(char *,char *,int);
extern void  PetscStrncpy(char *,char *,int);
extern char* PetscStrchr(char *,char);
extern char* PetscStrrchr(char *,char);
extern char* PetscStrstr(char*,char*);
extern char* PetscStrtok(char*,char*);
extern char* PetscStrrtok(char*,char*);

#define PetscMin(a,b)      ( ((a)<(b)) ? (a) : (b) )
#define PetscMax(a,b)      ( ((a)<(b)) ? (b) : (a) )
#define PetscAbsInt(a)     ( ((a)<0)   ? -(a) : (a) )

typedef enum { PETSC_FALSE, PETSC_TRUE } PetscTruth;
#define PETSC_NULL            0
#define PETSC_DECIDE         -1
#define PETSC_DEFAULT        -2

/*  Macros for error checking */
#if !defined(__DIR__)
#define __DIR__ 0
#endif

/*
     Error codes (incomplete)
*/
#define PETSC_ERR_MEM 55   /* unable to allocate requested memory */
#define PETSC_ERR_SUP 56   /* no support yet for this operation */
#define PETSC_ERR_ARG 57   /* bad input argument */
#define PETSC_ERR_OBJ 58   /* null or corrupt PETSc object */
#define PETSC_ERR_SIG 59   /* signal received */
#define PETSC_ERR_SIZ 60   /* nonconforming object sizes */

#if defined(PETSC_DEBUG)
#define SETERRQ(n,s)   {return PetscError(__LINE__,__DIR__,__FILE__,n,s);}
#define SETERRA(n,s)   {int _ierr = PetscError(__LINE__,__DIR__,__FILE__,n,s);\
                          MPI_Abort(MPI_COMM_WORLD,_ierr);}
#define CHKERRQ(n)     {if (n) SETERRQ(n,(char *)0);}
#define CHKERRA(n)     {if (n) SETERRA(n,(char *)0);}
#define CHKPTRQ(p)     if (!p) SETERRQ(PETSC_ERR_MEM,(char*)0);
#define CHKPTRA(p)     if (!p) SETERRA(PETSC_ERR_MEM,(char*)0);
#else
#define SETERRQ(n,s)   {return PetscError(__LINE__,__DIR__,__FILE__,n,s);}
#define SETERRA(n,s)   {int _ierr = PetscError(__LINE__,__DIR__,__FILE__,n,s);\
                          MPI_Abort(MPI_COMM_WORLD,_ierr);}
#define CHKERRQ(n)     {if (n) SETERRQ(n,(char *)0);}
#define CHKERRA(n)     {if (n) SETERRA(n,(char *)0);}
#define CHKPTRQ(p)     if (!p) SETERRQ(PETSC_ERR_MEM,(char*)0);
#define CHKPTRA(p)     if (!p) SETERRA(PETSC_ERR_MEM,(char*)0);
#endif

#define PETSC_COOKIE                1211211
#define LARGEST_PETSC_COOKIE_STATIC PETSC_COOKIE + 30
extern int LARGEST_PETSC_COOKIE;

#include "viewer.h"
#include "options.h"

extern double PetscGetTime();
extern double PetscGetFlops();
extern void   PetscSleep(int);

extern int PetscInitialize(int*,char***,char*,char*);
extern int PetscFinalize();

typedef struct _PetscObject* PetscObject;
extern int PetscObjectDestroy(PetscObject);
extern int PetscObjectExists(PetscObject,int*);
extern int PetscObjectGetComm(PetscObject,MPI_Comm *comm);
extern int PetscObjectGetCookie(PetscObject,int *cookie);
extern int PetscObjectGetChild(PetscObject,void **child);
extern int PetscObjectGetType(PetscObject,int *type);
extern int PetscObjectSetName(PetscObject,char*);
extern int PetscObjectGetName(PetscObject,char**);
extern int PetscObjectInherit(PetscObject,void *, int (*)(void *,void **));
#define PetscObjectChild(a) (((PetscObject) (a))->child)

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
#define FP_TRAP_OFF    0
#define FP_TRAP_ON     1
#define FP_TRAP_ALWAYS 2
extern int PetscSetFPTrap(int);

#include "phead.h"
#include "plog.h"

extern int  PetscSequentialPhaseBegin(MPI_Comm,int);
extern int  PetscSequentialPhaseEnd(MPI_Comm,int);
#define PetscBarrier(A) \
  { \
    PetscValidHeader(A); \
    PLogEventBegin(Petsc_Barrier,A,0,0,0); \
    MPI_Barrier(((PetscObject)A)->comm); \
    PLogEventEnd(Petsc_Barrier,A,0,0,0); \
  }

/*
      This code allows one to pass a PETSc object in C
  to a Fortran routine, where (like all PETSc objects in 
  Fortran) it is treated as an integer.
*/
extern int PetscCObjectToFortranObject(void *a,int *b);
extern int PetscFortranObjectToCObject(int a,void *b);

extern FILE *PetscFOpen(MPI_Comm,char *,char *);
extern int  PetscFClose(MPI_Comm,FILE*);
extern int  PetscFPrintf(MPI_Comm,FILE*,char *,...);
extern int  PetscPrintf(MPI_Comm,char *,...);

#endif

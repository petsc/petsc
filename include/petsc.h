/* $Id: petsc.h,v 1.133 1996/10/02 02:47:28 curfman Exp bsmith $ */
/*
   This is the main PETSc include file (for C and C++).  It is included by
   all other PETSc include files so almost never has to be specifically included.
*/
#if !defined(__PETSC_PACKAGE)
#define __PETSC_PACKAGE

#define PETSC_VERSION_NUMBER "PETSc Version 2.0.15, Released ???  ??, 1996."

#include <stdio.h>
#include "mpi.h"

#if defined(PETSC_COMPLEX)
#if defined(PARCH_t3d)
#include "/usr/include/mpp/CC/complex.h"
#else
#include <complex.h>
#endif
extern  MPI_Datatype      MPIU_COMPLEX;
#define MPIU_SCALAR       MPIU_COMPLEX
#define PetscReal(a)      real(a)
#define PetscAbsScalar(a) abs(a)
/*
  The new complex class for GNU C++ is based on templates and is not backward
  compatible with all previous complex class libraries.
*/
#if defined(USES_TEMPLATED_COMPLEX)
#define Scalar            complex<double>
#else
#define Scalar            complex
#endif
#else
#define MPIU_SCALAR       MPI_DOUBLE
#define PetscReal(a)      a
#define PetscAbsScalar(a) ( ((a)<0.0)   ? -(a) : (a) )
#define Scalar            double
#endif

/* PETSc world communicator */
extern MPI_Comm PETSC_COMM_WORLD;
extern int      PetscInitializedCalled;

/* PETSC_i is the imaginary number, i */
extern  Scalar            PETSC_i;

#define PetscMin(a,b)      ( ((a)<(b)) ? (a) : (b) )
#define PetscMax(a,b)      ( ((a)<(b)) ? (b) : (a) )
#define PetscAbsInt(a)     ( ((a)<0)   ? -(a) : (a) )
#define PetscAbsDouble(a)  ( ((a)<0)   ? -(a) : (a) )

/*
    Defines the malloc employed by PETSc. Users may employ these routines as well. 
*/
extern void *(*PetscTrMalloc)(unsigned int,int,char*);
extern int  (*PetscTrFree)(void *,int,char*);
extern int  PetscSetMalloc(void *(*)(unsigned int,int,char*),int (*)(void *,int,char*));
#define PetscMalloc(a)       (*PetscTrMalloc)(a,__LINE__,__FILE__)
#define PetscNew(A)          (A*) PetscMalloc(sizeof(A))
#define PetscFree(a)         (*PetscTrFree)(a,__LINE__,__FILE__)

extern int   PetscTrDump(FILE *);
extern int   PetscTrSpace( double *, double *,double *);
extern int   PetscTrValid(int ,char*);
extern int   PetscTrDebugLevel(int);

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

typedef enum { PETSC_FALSE, PETSC_TRUE } PetscTruth;
#define PETSC_NULL            0
#define PETSC_DECIDE         -1
#define PETSC_DEFAULT        -2

/*
   Defines the directory where the compiled source is located; used
   in print error messages. __DIR__ is usually defined in the makefile.
*/
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

/*
    Each PETSc object class has it's own cookie (internal integer in the 
  data structure used for error checking). These are all defined by an offset 
  from the lowest one, PETSC_COOKIE.
*/
#define PETSC_COOKIE                1211211
#define LARGEST_PETSC_COOKIE_STATIC PETSC_COOKIE + 30
extern int LARGEST_PETSC_COOKIE;

#include "viewer.h"
#include "options.h"

extern double PetscGetTime();
extern void   PetscSleep(int);

extern int    PetscInitialize(int*,char***,char*,char*);
extern int    PetscFinalize();
extern void   PetscInitializeFortran();

/*
    Functions that can act on any PETSc object.
*/
typedef struct _PetscObject* PetscObject;
extern int PetscObjectDestroy(PetscObject);
extern int PetscObjectExists(PetscObject,int*);
extern int PetscObjectGetComm(PetscObject,MPI_Comm *comm);
extern int PetscObjectGetCookie(PetscObject,int *cookie);
extern int PetscObjectGetChild(PetscObject,void **child);
extern int PetscObjectGetType(PetscObject,int *type);
extern int PetscObjectSetName(PetscObject,char*);
extern int PetscObjectGetName(PetscObject,char**);
extern int PetscObjectInherit(PetscObject,void *, int (*)(void *,void **),int (*)(void*));
extern int PetscObjectReference(PetscObject);
extern int PetscObjectGetNewTag(PetscObject,int *);
extern int PetscObjectRestoreNewTag(PetscObject,int *);
extern int PetscSetCommWorld(MPI_Comm);

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

#include "phead.h"
#include "plog.h"

extern int  PetscSequentialPhaseBegin(MPI_Comm,int);
extern int  PetscSequentialPhaseEnd(MPI_Comm,int);

/*M PetscBarrier - Blocks Until this routine is executed by all
    processors owning the object A

   Input Parameters:
.  A - PETSc object  ( Mat, Vec, IS, SNES etc...)

   Synopsis:
   void PetscBarrier(PetscObject obj)

  Notes: 
  This routine calls MPI_Barrier with the communicator
  of the PETSc Object "A". 

.keywords: barrier, petscobject
M*/

#define PetscBarrier(A) \
  { \
    PetscValidHeader(A); \
    PLogEventBegin(Petsc_Barrier,A,0,0,0); \
    MPI_Barrier(((PetscObject)A)->comm); \
    PLogEventEnd(Petsc_Barrier,A,0,0,0); \
  }

extern int PetscMPIDump(FILE *);

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

extern int PetscIntView(int,int*,Viewer);
extern int PetscDoubleView(int,double *,Viewer);

/*
   For use in debuggers 
*/
extern int PetscGlobalRank,PetscGlobalSize;

#endif

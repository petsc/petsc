/* $Id: petsc.h,v 1.57 1995/09/06 03:06:53 bsmith Exp bsmith $ */

#if !defined(__PETSC_PACKAGE)
#define __PETSC_PACKAGE

#define PETSC_VERSION_NUMBER "PETSc Version 2.0.Beta.8 Released ?, 1995."

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
extern int  TrDump(FILE *);
extern int  TrGetMaximumAllocated(double*);

#define PETSCNEW(A)         (A*) PETSCMALLOC(sizeof(A))
#define PETSCMEMCPY(a,b,n)   memcpy((char*)(a),(char*)(b),n)
#define PETSCMEMSET(a,b,n)   memset((char*)(a),(int)(b),n)
#include <memory.h>

#define PETSCMIN(a,b)      ( ((a)<(b)) ? (a) : (b) )
#define PETSCMAX(a,b)      ( ((a)<(b)) ? (b) : (a) )
#define PETSCABS(a)        ( ((a)<0)   ? -(a) : (a) )

/*  Macros for error checking */
#if !defined(__DIR__)
#define __DIR__ 0
#endif

/*
       Unable to malloc error and no supported function
*/
#define PETSC_ERR_MEM 55   /* unalbe to allocate requested memory */
#define PETSC_ERR_SUP 56   /* no support yet for this operation */
#define PETSC_ERR_ARG 57   /* bad input argument */
#define PETSC_ERR_OBJ 58   /* null or corrupt PETSc object */

#if defined(PETSC_DEBUG)
#define SETERRQ(n,s)     {return PetscError(__LINE__,__DIR__,__FILE__,s,n);}
#define SETERRA(n,s)    \
                {int _ierr = PetscError(__LINE__,__DIR__,__FILE__,s,n);\
                 MPI_Abort(MPI_COMM_WORLD,_ierr);}
#define CHKERRQ(n)       {if (n) SETERRQ(n,(char *)0);}
#define CHKERRA(n)      {if (n) SETERRA(n,(char *)0);}
#define CHKPTRQ(p)       if (!p) SETERRQ(PETSC_ERR_MEM,(char*)0);
#define CHKPTRA(p)       if (!p) SETERRA(PETSC_ERR_MEM,(char*)0);
#else
#define SETERRQ(n,s)     {return PetscError(__LINE__,__DIR__,__FILE__,s,n);}
#define SETERRA(n,s)    \
                {int _ierr = PetscError(__LINE__,__DIR__,__FILE__,s,n);\
                 MPI_Abort(MPI_COMM_WORLD,_ierr);}
#define CHKERRQ(n)       {if (n) SETERRQ(n,(char *)0);}
#define CHKERRA(n)      {if (n) SETERRA(n,(char *)0);}
#define CHKPTRQ(p)       if (!p) SETERRQ(PETSC_ERR_MEM,(char*)0);
#define CHKPTRA(p)       if (!p) SETERRA(PETSC_ERR_MEM,(char*)0);
#endif

typedef struct _PetscObject* PetscObject;
#define PETSC_COOKIE         1211211
#define PETSC_DECIDE         -1
#define PETSC_DEFAULT        -2

typedef enum { PETSC_FALSE, PETSC_TRUE } PetscTruth;

#include "viewer.h"
#include "options.h"

extern int PetscInitialize(int*,char***,char*,char*);
extern int PetscFinalize();

extern int PetscObjectDestroy(PetscObject);
extern int PetscObjectExists(PetscObject,int*);
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

/*
   Definitions used for the Fortran interface:
   FORTRANCAPS:       Names are uppercase, no trailing underscore
   FORTRANUNDERSCORE: Names are lowercase, trailing underscore
 */    
#if defined(PARCH_cray) || defined(PARCH_NCUBE) || defined(PARCH_t3d)
#define FORTRANCAPS
#elif !defined(PARCH_rs6000) && !defined(PARCH_NeXT) && !defined(PARCH_hpux)
#define FORTRANUNDERSCORE
#endif

#include "phead.h"
#include "plog.h"

extern void PetscSleep(int);

#endif

/* $Id: petsc.h,v 1.155 1997/04/05 15:27:50 bsmith Exp bsmith $ */
/*
   This is the main PETSc include file (for C and C++).  It is included by
   all other PETSc include files so almost never has to be specifically included.
*/
#if !defined(__PETSC_PACKAGE)
#define __PETSC_PACKAGE

#define PETSC_VERSION_NUMBER "PETSc Version 2.0.17, Released April 5, 1997."

#include <stdio.h>
#include "mpi.h"

#if defined(PETSC_COMPLEX)
#if defined(HAVE_NONSTANDARD_COMPLEX_H)
#include HAVE_NONSTANDARD_COMPLEX_H
#else
#include <complex.h>
#endif
extern  MPI_Datatype      MPIU_COMPLEX;
#define MPIU_SCALAR       MPIU_COMPLEX
#define PetscReal(a)      real(a)
#define PetscImaginary(a) imag(a)
#define PetscAbsScalar(a) abs(a)
#define PetscConj(a)      conj(a)
/*
  The new complex class for GNU C++ is based on templates and is not backward
  compatible with all previous complex class libraries.
*/
#if defined(USES_TEMPLATED_COMPLEX)
#define Scalar            complex<double>
#else
#define Scalar            complex
#endif

/* Compiling for real numbers only */
#else
#define MPIU_SCALAR       MPI_DOUBLE
#define PetscReal(a)      (a)
#define PetscImaginary(a) (a)
#define PetscAbsScalar(a) ( ((a)<0.0)   ? -(a) : (a) )
#define Scalar            double
#define PetscConj(a)      (a)
#endif

/*
   Certain objects may be created using either single
  or double precision.
*/
typedef enum { SCALAR_DOUBLE, SCALAR_SINGLE } ScalarPrecision;

extern MPI_Comm PETSC_COMM_WORLD;
extern MPI_Comm PETSC_COMM_SELF;
extern int      PetscInitializedCalled;
extern int      PetscSetCommWorld(MPI_Comm);

/* PETSC_i is the imaginary number, i */
extern  Scalar            PETSC_i;

#define PetscMin(a,b)      ( ((a)<(b)) ? (a) : (b) )
#define PetscMax(a,b)      ( ((a)<(b)) ? (b) : (a) )
#define PetscAbsInt(a)     ( ((a)<0)   ? -(a) : (a) )
#define PetscAbsDouble(a)  ( ((a)<0)   ? -(a) : (a) )

/*
    PLogDouble variables are used to contain double precision numbers
  that are not used in the numerical computations, but rather in logging,
  timing etc.
*/
typedef double PLogDouble;

/*
    Defines the malloc employed by PETSc. Users may employ these routines as well. 
*/
extern void *(*PetscTrMalloc)(unsigned int,int,char*,char*,char*);
extern int  (*PetscTrFree)(void *,int,char*,char*,char*);
extern int  PetscSetMalloc(void *(*)(unsigned int,int,char*,char*,char*),
                           int (*)(void *,int,char*,char*,char*));
#define PetscMalloc(a)       (*PetscTrMalloc)(a,__LINE__,__FUNC__,__FILE__,__SDIR__)
#define PetscNew(A)          (A*) PetscMalloc(sizeof(A))
#define PetscFree(a)         (*PetscTrFree)(a,__LINE__,__FUNC__,__FILE__,__SDIR__)

extern int   PetscTrDump(FILE *);
extern int   PetscTrSpace( PLogDouble *, PLogDouble *,PLogDouble *);
extern int   PetscTrValid(int,char *,char *,char *);
extern int   PetscTrDebugLevel(int);
extern int   PetscTrLog();
extern int   PetscTrLogDump(FILE *);
extern int   PetscGetResidentSetSize(PLogDouble *);

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
    Each PETSc object class has it's own cookie (internal integer in the 
  data structure used for error checking). These are all defined by an offset 
  from the lowest one, PETSC_COOKIE. If you increase these you must 
  increase the field sizes in petsc/src/plog/src/plog.c
*/
#define PETSC_COOKIE                    1211211
#define LARGEST_PETSC_COOKIE_PREDEFINED PETSC_COOKIE + 30
#define LARGEST_PETSC_COOKIE_ALLOWED    PETSC_COOKIE + 50
extern int LARGEST_PETSC_COOKIE;

#include "viewer.h"
#include "options.h"
#include "draw.h"

extern PLogDouble PetscGetTime();
extern void       PetscSleep(int);

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

#include "petscerror.h"
#include "petschead.h"
#include "petsclog.h"

extern int  PetscSequentialPhaseBegin(MPI_Comm,int);
extern int  PetscSequentialPhaseEnd(MPI_Comm,int);

/*M 
    PetscBarrier - Blocks until this routine is executed by all
                   processors owning the object A.

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
extern int  PetscCObjectToFortranObject(void *a,int *b);
extern int  PetscFortranObjectToCObject(int a,void *b);

extern FILE *PetscFOpen(MPI_Comm,char *,char *);
extern int  PetscFClose(MPI_Comm,FILE*);
extern int  PetscFPrintf(MPI_Comm,FILE*,char *,...);
extern int  PetscPrintf(MPI_Comm,char *,...);

extern int  PetscSynchronizedPrintf(MPI_Comm,char *,...);
extern int  PetscSynchronizedFlush(MPI_Comm);

/*
   For incremental debugging
*/
extern int PetscCompare;
extern int PetscCompareDouble(double);
extern int PetscCompareScalar(Scalar);
extern int PetscCompareInt(int);

/*
   For use in debuggers 
*/
extern int PetscGlobalRank,PetscGlobalSize;
extern int PetscIntView(int,int*,Viewer);
extern int PetscDoubleView(int,double *,Viewer);

#endif

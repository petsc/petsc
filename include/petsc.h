/* $Id: petsc.h,v 1.181 1997/10/07 19:50:08 balay Exp bsmith $ */
/*
   This is the main PETSc include file (for C and C++).  It is included by
   all other PETSc include files so almost never has to be specifically included.
*/
#if !defined(__PETSC_PACKAGE)
#define __PETSC_PACKAGE

/* 
   Current PETSc Version 
*/
#define PETSC_VERSION_NUMBER "PETSc Version 2.0.20, Released October 8, 1997."

#define PETSC_VERSION_MAJOR    2
#define PETSC_VERSION_MINOR    0
#define PETSC_VERSION_SUBMINOR 20
#define PETSC_VERSION_DATE     "October 8, 1997"

/* Before anything else, include the PETSc configuration file.  This 
   contains various definitions that handle portability issues and the 
   presence of important features.  For backward compatibility while 
   developing, this configuration is itself conditionally included.
 */
#ifdef HAVE_PETSCCONF_H
#include "petscconf.h"
#else

/* These are temporary; they contain PARCH_xxxx -> feature-specific
   definitions */
/* Common definitions (sometimes undef'ed below) */
#define HAVE_READLINK
#define HAVE_MEMMOVE

#if defined(PARCH_sun4)
/* Fortran BLAS have slow dnrm2 */
#define HAVE_SLOW_NRM2
/* Functions that we count on Sun4's having */
#define HAVE_GETWD
#define HAVE_REALPATH
/* Functions that Sun4's don't have */
#undef HAVE_MEMMOVE
#endif

#if defined(PARCH_rs6000)
/* Use bzero instead of memset( ,0, ) */
#define PREFER_BZERO
/* Some versions of AIX require u_type definitions */
/* #define NEED_UTYPE_TYPEDEFS */
#endif

#if defined(PARCH_IRIX) || defined(PARCH_IRIX64) || defined(PARCH_IRIX5)
/* For some reason, we don't use readlink in grpath.c for IRIX */
#undef HAVE_READLINK
/* gettimeofday required sys/resource.h and C++ prototype for gettimeof
   day */
#define NEEDS_GETTIMEOFDAY_PROTO
#endif

#if defined(PARCH_paragon) ||  defined(PARCH_alpha)
/* Some versions of these systems require u_type definitions */
#define NEED_UTYPE_TYPEDEFS
#endif
#endif


#include <stdio.h>
/*
    Defines the interface to MPI allowing the use of all MPI functions.
*/
#include "mpi.h"

/*
    Defines some elementary mathematics functions and constants.
*/
#include "petscmath.h"

extern MPI_Comm PETSC_COMM_WORLD;
extern MPI_Comm PETSC_COMM_SELF;
extern int      PetscInitializedCalled;
extern int      PetscSetCommWorld(MPI_Comm);

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

/*
    Basic memory and string operations
*/
extern int   PetscMemcpy(void *,void *,int);
extern int   PetscMemmove(void *,void *,int);
extern int   PetscMemzero(void *,int);
extern int   PetscMemcmp(void*, void*, int);
extern int   PetscStrlen(char *);
extern int   PetscStrcmp(char *,char *);
extern int   PetscStrcasecmp(char *,char *);
extern int   PetscStrncmp(char *,char *,int );
extern int   PetscStrcpy(char *,char *);
extern int   PetscStrcat(char *,char *);
extern int   PetscStrncat(char *,char *,int);
extern int   PetscStrncpy(char *,char *,int);
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

/*
    Defines basic graphics available from PETSc.
*/
#include "draw.h"

extern PLogDouble PetscGetTime();
extern PLogDouble PetscGetCPUTime();
extern void       PetscSleep(int);

extern int    PetscInitialize(int*,char***,char*,char*);
extern int    PetscFinalize();
extern void   PetscInitializeFortran();

/*
    Functions that can act on any PETSc object.
*/
typedef struct _p_PetscObject* PetscObject;
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
extern int PetscObjectDereference(PetscObject);
extern int PetscObjectGetNewTag(PetscObject,int *);
extern int PetscObjectRestoreNewTag(PetscObject,int *);
extern int PetscObjectView(PetscObject,Viewer);

typedef enum {PETSC_INT = 0, PETSC_DOUBLE = 1, PETSC_SHORT = 2, PETSC_FLOAT = 3,
              PETSC_DCOMPLEX = 4, PETSC_CHAR = 5} PetscDataType;
#if defined(PETSC_COMPLEX)
#define PETSC_SCALAR PETSC_DCOMPLEX
#else
#define PETSC_SCALAR PETSC_DOUBLE
#endif

typedef enum {PETSC_INT_SIZE = sizeof(int), PETSC_DOUBLE_SIZE = sizeof(double),
              PETSC_SCALAR_SIZE = sizeof(Scalar), PETSC_DCOMPLEX_SIZE = sizeof(double),
              PETSC_CHAR_SIZE = sizeof(char)} PetscDataTypeSize;
extern int PetscDataTypeToMPIDataType(PetscDataType,MPI_Datatype*);
extern int PetscDataTypeGetSize(PetscDataType,int*);
extern int PetscDataTypeGetName(PetscDataType,char**);

/*
    Defines PETSc error handling.
*/
#include "petscerror.h"
#include "petschead.h"

/*
     Defines PETSc profiling.
*/
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
extern int  PetscCObjectToFortranObject(void *,int *);
extern int  PetscFortranObjectToCObject(int,void *);
extern int  MPICCommToFortranComm(MPI_Comm,int *);
extern int  MPIFortranCommToCComm(int,MPI_Comm*);

/*
      Simple PETSc parallel IO for ASCII printing
*/
extern FILE *PetscFOpen(MPI_Comm,char *,char *);
extern int  PetscFClose(MPI_Comm,FILE*);
extern int  PetscFPrintf(MPI_Comm,FILE*,char *,...);
extern int  PetscPrintf(MPI_Comm,char *,...);

extern int  PetscSynchronizedPrintf(MPI_Comm,char *,...);
extern int  PetscSynchronizedFPrintf(MPI_Comm,FILE*,char *,...);
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

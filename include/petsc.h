/* $Id: petsc.h,v 1.276 2000/08/01 20:58:40 bsmith Exp bsmith $ */
/*
   This is the main PETSc include file (for C and C++).  It is included by all
   other PETSc include files, so it almost never has to be specifically included.
*/
#if !defined(__PETSC_H)
#define __PETSC_H

/* ========================================================================== */
/* 
   Current PETSc version number and release date
*/
#include "petscversion.h"

/* ========================================================================== */
/* 
   The PETSc configuration file.  Contains various definitions that
   handle portability issues and the presence of machine features. 

   petscconf.h is contained in bmake/${PETSC_ARCH}/petscconf.h it is 
   found automatically by the compiler due to the -I${PETSC_DIR}/bmake/${PETSC_ARCH}
   in the bmake/common_variables definition of PETSC_INCLUDE
*/
#include "petscconf.h"

/* ========================================================================== */

#include <stdio.h>
/*
    Defines the interface to MPI allowing the use of all MPI functions.
*/
#include "mpi.h"

/*
    EXTERN indicates a PETSc function defined elsewhere
*/
#define EXTERN extern

/*
    Defines some elementary mathematics functions and constants.
*/
#include "petscmath.h"

/*
    Variable type where we stash PETSc object pointers in Fortran.
    Assumes that sizeof(long) == sizeof(void*)which is true on 
    all machines that we know.
*/     
#define PetscFortranAddr   long

/*
       Basic PETSc constants
*/
typedef enum { PETSC_FALSE,PETSC_TRUE } PetscTruth;
#define PETSC_YES            PETSC_TRUE
#define PETSC_NO             PETSC_FALSE
#define PETSC_NULL           0
#define PETSC_IGNORE         PETSC_NULL
#define PETSC_DECIDE         -1
#define PETSC_DETERMINE      PETSC_DECIDE
#define PETSC_DEFAULT        -2

extern MPI_Comm   PETSC_COMM_WORLD;
extern MPI_Comm   PETSC_COMM_SELF;
extern PetscTruth PetscInitializeCalled;
EXTERN int        PetscSetCommWorld(MPI_Comm);
EXTERN int        PetscSetHelpVersionFunctions(int (*)(MPI_Comm),int (*)(MPI_Comm));

/*
    Defines the malloc employed by PETSc. Users may use these routines as well. 
*/
#define PetscMalloc(a)       (*PetscTrMalloc)(a,__LINE__,__FUNC__,__FILE__,__SDIR__)
#define PetscNew(A)          (A*)PetscMalloc(sizeof(A))
#define PetscFree(a)         (*PetscTrFree)(a,__LINE__,__FUNC__,__FILE__,__SDIR__)
EXTERN void *(*PetscTrMalloc)(int,int,char*,char*,char*);
EXTERN int  (*PetscTrFree)(void *,int,char*,char*,char*);
EXTERN int  PetscSetMalloc(void *(*)(int,int,char*,char*,char*),int (*)(void *,int,char*,char*,char*));
EXTERN int  PetscClearMalloc(void);

/*
   Routines for tracing memory corruption/bleeding with default PETSc 
   memory allocation
*/
EXTERN int   PetscTrDump(FILE *);
EXTERN int   PetscTrSpace(PLogDouble *,PLogDouble *,PLogDouble *);
EXTERN int   PetscTrValid(int,const char[],const char[],const char[]);
EXTERN int   PetscTrDebugLevel(int);
EXTERN int   PetscTrLog(void);
EXTERN int   PetscTrLogDump(FILE *);
EXTERN int   PetscGetResidentSetSize(PLogDouble *);
/*
     Constants and functions used for handling different basic data types.
     These are used, for example, in binary IO routines
*/
typedef enum {PETSC_INT = 0,PETSC_DOUBLE = 1,PETSC_SHORT = 2,PETSC_FLOAT = 3,
              PETSC_COMPLEX = 4,PETSC_CHAR = 5,PETSC_LOGICAL = 6} PetscDataType;
#if defined(PETSC_USE_COMPLEX)
#define PETSC_SCALAR PETSC_COMPLEX
#else
#define PETSC_SCALAR PETSC_DOUBLE
#endif
#if defined(PETSC_USE_SINGLE)
#define PETSC_REAL PETSC_FLOAT
#else
#define PETSC_REAL PETSC_DOUBLE
#endif

typedef enum {PETSC_INT_SIZE = sizeof(int),PETSC_DOUBLE_SIZE = sizeof(double),
              PETSC_SCALAR_SIZE = sizeof(Scalar),PETSC_COMPLEX_SIZE = sizeof(double),
              PETSC_CHAR_SIZE = sizeof(char),PETSC_LOGICAL_SIZE = 1} PetscDataTypeSize;
#if defined(PETSC_USE_SINGLE)
#define PETSC_REAL_SIZE PETSC_FLOAT_SIZE
#else
#define PETSC_REAL_SIZE PETSC_DOUBLE_SIZE
#endif

EXTERN int PetscDataTypeToMPIDataType(PetscDataType,MPI_Datatype*);
EXTERN int PetscDataTypeGetSize(PetscDataType,int*);
EXTERN int PetscDataTypeGetName(PetscDataType,char*[]);

/*
    Basic memory and string operations. These are usually simple wrappers
   around the basic Unix system calls, but a few of them have additional
   functionality and/or error checking.
*/
EXTERN int   PetscMemcpy(void *,const void *,int);
EXTERN int   PetscBitMemcpy(void*,int,const void*,int,int,PetscDataType);
EXTERN int   PetscMemmove(void *,void *,int);
EXTERN int   PetscMemzero(void *,int);
EXTERN int   PetscMemcmp(const void*,const void*,int,PetscTruth *);
EXTERN int   PetscStrlen(const char[],int*);
EXTERN int   PetscStrcmp(const char[],const char[],PetscTruth *);
EXTERN int   PetscStrgrt(const char[],const char[],PetscTruth *);
EXTERN int   PetscStrcasecmp(const char[],const char[],PetscTruth*);
EXTERN int   PetscStrncmp(const char[],const char[],int,PetscTruth*);
EXTERN int   PetscStrcpy(char[],const char[]);
EXTERN int   PetscStrcat(char[],const char[]);
EXTERN int   PetscStrncat(char[],const char[],int);
EXTERN int   PetscStrncpy(char[],const char[],int);
EXTERN int   PetscStrchr(const char[],char,char **);
EXTERN int   PetscStrtolower(char[]);
EXTERN int   PetscStrrchr(const char[],char,char **);
EXTERN int   PetscStrstr(const char[],const char[],char **);
EXTERN int   PetscStrtok(const char[],const char[],char **);
EXTERN int   PetscStrallocpy(const char[],char **);
EXTERN int   PetscStrreplace(MPI_Comm,const char[],char*,int);
#define      PetscStrfree(a) ((a) ? PetscFree(a) : 0) 

/*
   These are  MPI operations for MPI_Allreduce() etc
*/
EXTERN MPI_Op PetscMaxSum_Op;
#if defined(PETSC_USE_COMPLEX)
EXTERN MPI_Op PetscSum_Op;
#else
#define PetscSum_Op MPI_SUM
#endif

/*
  Each PETSc object class has it's own cookie (internal integer in the 
  data structure used for error checking). These are all defined by an offset 
  from the lowest one, PETSC_COOKIE.
*/
#define PETSC_COOKIE                    1211211
#define PETSC_MAX_COOKIES               60
#define LARGEST_PETSC_COOKIE_PREDEFINED PETSC_COOKIE + 30
#define LARGEST_PETSC_COOKIE_ALLOWED    PETSC_COOKIE + PETSC_MAX_COOKIES
extern int LARGEST_PETSC_COOKIE;

typedef struct _p_PetscObject* PetscObject;
typedef struct _FList *FList;

#include "petscviewer.h"
#include "petscoptions.h"

EXTERN int PetscShowMemoryUsage(Viewer,char*);
EXTERN int PetscGetTime(PLogDouble*);
EXTERN int PetscGetCPUTime(PLogDouble*);
EXTERN int PetscSleep(int);

/*
    Initialization of PETSc
*/
EXTERN int  PetscInitialize(int*,char***,char[],const char[]);
EXTERN int  PetscInitializeNoArguments(void);
EXTERN int  PetscFinalize(void);
EXTERN int  PetscInitializeFortran(void);

/*
    Functions that can act on any PETSc object.
*/
EXTERN int PetscObjectDestroy(PetscObject);
EXTERN int PetscObjectExists(PetscObject,PetscTruth*);
EXTERN int PetscObjectGetComm(PetscObject,MPI_Comm *);
EXTERN int PetscObjectGetCookie(PetscObject,int *);
EXTERN int PetscObjectGetType(PetscObject,int *);
EXTERN int PetscObjectSetName(PetscObject,const char[]);
EXTERN int PetscObjectGetName(PetscObject,char*[]);
EXTERN int PetscObjectReference(PetscObject);
EXTERN int PetscObjectGetReference(PetscObject,int*);
EXTERN int PetscObjectDereference(PetscObject);
EXTERN int PetscObjectGetNewTag(PetscObject,int *);
EXTERN int PetscCommGetNewTag(MPI_Comm,int *);
EXTERN int PetscObjectView(PetscObject,Viewer);
EXTERN int PetscObjectCompose(PetscObject,const char[],PetscObject);
EXTERN int PetscObjectQuery(PetscObject,const char[],PetscObject *);
EXTERN int PetscObjectComposeFunction(PetscObject,const char[],const char[],void *);
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define PetscObjectComposeFunctionDynamic(a,b,c,d) PetscObjectComposeFunction(a,b,c,0)
#else
#define PetscObjectComposeFunctionDynamic(a,b,c,d) PetscObjectComposeFunction(a,b,c,(void*)(d))
#endif
EXTERN int PetscObjectQueryFunction(PetscObject,const char[],void **);
EXTERN int PetscObjectSetOptionsPrefix(PetscObject,const char[]);
EXTERN int PetscObjectAppendOptionsPrefix(PetscObject,const char[]);
EXTERN int PetscObjectPrependOptionsPrefix(PetscObject,const char[]);
EXTERN int PetscObjectGetOptionsPrefix(PetscObject,char*[]);
EXTERN int PetscObjectPublish(PetscObject);
EXTERN int PetscObjectChangeTypeName(PetscObject,char *);
EXTERN int PetscObjectRegisterDestroy(PetscObject);
EXTERN int PetscObjectRegisterDestroyAll(void);
EXTERN int PetscObjectName(PetscObject);
EXTERN int PetscTypeCompare(PetscObject,char*,PetscTruth*);

/*
    Defines PETSc error handling.
*/
#include "petscerror.h"

/*
    Mechanism for managing lists of objects attached (composed) with 
   a PETSc object.
*/
typedef struct _OList *OList;
EXTERN int OListDestroy(OList *);
EXTERN int OListFind(OList,const char[],PetscObject*);
EXTERN int OListReverseFind(OList,PetscObject,char**);
EXTERN int OListAdd(OList *,const char[],PetscObject);
EXTERN int OListDuplicate(OList,OList *);

/*
    Dynamic library lists. Lists of names of routines in dynamic 
  link libraries that will be loaded as needed.
*/
EXTERN int FListAdd(FList*,const char[],const char[],int (*)(void *));
EXTERN int FListDestroy(FList*);
EXTERN int FListFind(MPI_Comm,FList,const char[],int (**)(void*));
EXTERN int FListPrintTypes(MPI_Comm,FILE*,const char[],const char[],FList);
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define    FListAddDynamic(a,b,p,c) FListAdd(a,b,p,0)
#else
#define    FListAddDynamic(a,b,p,c) FListAdd(a,b,p,(int (*)(void *))c)
#endif
EXTERN int FListDuplicate(FList,FList *);
EXTERN int FListView(FList,Viewer);
EXTERN int FListConcat(const char [],const char [],char []);
EXTERN int FListGet(FList,char ***,int*);

/*
   Routines for handling dynamic libraries. PETSc uses dynamic libraries
  by default on most machines (except IBM). This is controlled by the
  flag PETSC_USE_DYNAMIC_LIBRARIES in petscconf.h
*/
typedef struct _DLLibraryList *DLLibraryList;
extern DLLibraryList DLLibrariesLoaded;
EXTERN int DLLibraryRetrieve(MPI_Comm,const char[],char *,int,PetscTruth *);
EXTERN int DLLibraryOpen(MPI_Comm,const char[],void **);
EXTERN int DLLibrarySym(MPI_Comm,DLLibraryList *,const char[],const char[],void **);
EXTERN int DLLibraryAppend(MPI_Comm,DLLibraryList *,const char[]);
EXTERN int DLLibraryPrepend(MPI_Comm,DLLibraryList *,const char[]);
EXTERN int DLLibraryClose(DLLibraryList);
EXTERN int DLLibraryPrintPath(void);
EXTERN int DLLibraryGetInfo(void *,char *,char **);

/*
    Mechanism for translating PETSc object representations between languages
    Not currently used.
*/
typedef enum {PETSC_LANGUAGE_C,PETSC_LANGUAGE_CPP} PetscLanguage;
#define PETSC_LANGUAGE_F77 PETSC_LANGUAGE_C
EXTERN int PetscObjectComposeLanguage(PetscObject,PetscLanguage,void *);
EXTERN int PetscObjectQueryLanguage(PetscObject,PetscLanguage,void **);

/*
     Useful utility routines
*/
EXTERN int PetscSplitOwnership(MPI_Comm,int*,int*);
EXTERN int PetscSequentialPhaseBegin(MPI_Comm,int);
EXTERN int PetscSequentialPhaseEnd(MPI_Comm,int);
EXTERN int PetscBarrier(PetscObject);
EXTERN int PetscMPIDump(FILE*);

#define PetscNot(a) ((a) ? PETSC_FALSE : PETSC_TRUE)
/*
    Defines basic graphics available from PETSc.
*/
#include "petscdraw.h"

/*
    Defines the base data structures for all PETSc objects
*/
#include "petschead.h"

/*
     Defines PETSc profiling.
*/
#include "petsclog.h"

/*
          For locking, unlocking and destroying AMS memories associated with 
    PETSc objects
*/
#if defined(PETSC_HAVE_AMS)

extern PetscTruth PetscAMSPublishAll;
#define PetscPublishAll(v) 0;\
  { if (PetscAMSPublishAll) { \
    int __ierr; __ierr = PetscObjectPublish((PetscObject)v);CHKERRQ(__ierr);\
  }}
#define PetscObjectTakeAccess(obj)  ((((PetscObject)(obj))->amem == -1) ? 0 : AMS_Memory_take_access(((PetscObject)(obj))->amem))
#define PetscObjectGrantAccess(obj) ((((PetscObject)(obj))->amem == -1) ? 0 : AMS_Memory_grant_access(((PetscObject)(obj))->amem))
#define PetscObjectDepublish(obj)   ((((PetscObject)(obj))->amem == -1) ? 0 : AMS_Memory_destroy(((PetscObject)(obj))->amem)); \
    ((PetscObject)(obj))->amem = -1;

#else

#define PetscPublishAll(v)           0
#define PetscObjectTakeAccess(obj)   0
#define PetscObjectGrantAccess(obj)  0
#define PetscObjectDepublish(obj)      0

#endif



/*
      This code allows one to pass a MPI communicator between 
    C and Fortran. MPI 2.0 defines a standard API for doing this.
    The code here is provided to allow PETSc to work with MPI 1.1
    standard MPI libraries.
*/
EXTERN int  MPICCommToFortranComm(MPI_Comm,int *);
EXTERN int  MPIFortranCommToCComm(int,MPI_Comm*);

/*
      Simple PETSc parallel IO for ASCII printing
*/
EXTERN int  PetscFixFilename(const char[],char[]);
EXTERN int  PetscFOpen(MPI_Comm,const char[],const char[],FILE**);
EXTERN int  PetscFClose(MPI_Comm,FILE*);
EXTERN int  PetscFPrintf(MPI_Comm,FILE*,const char[],...);
EXTERN int  PetscPrintf(MPI_Comm,const char[],...);
EXTERN int  (*PetscErrorPrintf)(const char[],...);
EXTERN int  (*PetscHelpPrintf)(MPI_Comm,const char[],...);
EXTERN int  PetscPOpen(MPI_Comm,char *,char*,const char[],FILE **);
EXTERN int  PetscPClose(MPI_Comm,FILE*);
EXTERN int  PetscSynchronizedPrintf(MPI_Comm,const char[],...);
EXTERN int  PetscSynchronizedFPrintf(MPI_Comm,FILE*,const char[],...);
EXTERN int  PetscSynchronizedFlush(MPI_Comm);
EXTERN int  PetscSynchronizedFGets(MPI_Comm,FILE*,int,char[]);
EXTERN int  PetscStartMatlab(MPI_Comm,char *,char*,FILE**);
EXTERN int  PetscStartJava(MPI_Comm,char *,char*,FILE**);

EXTERN int  PetscPopUpSelect(MPI_Comm,char*,char*,int,char**,int*);
/*
    Simple PETSc object that contains a pointer to any required data
*/
typedef struct _p_PetscObjectContainer*  PetscObjectContainer;
EXTERN int PetscObjectContainerGetPointer(PetscObjectContainer,void **);
EXTERN int PetscObjectContainerSetPointer(PetscObjectContainer,void *);
EXTERN int PetscObjectContainerDestroy(PetscObjectContainer);
EXTERN int PetscObjectContainerCreate(MPI_Comm comm,PetscObjectContainer *);

/*
   For incremental debugging
*/
extern PetscTruth PetscCompare;
EXTERN int        PetscCompareDouble(double);
EXTERN int        PetscCompareScalar(Scalar);
EXTERN int        PetscCompareInt(int);

/*
   For use in debuggers 
*/
extern int PetscGlobalRank,PetscGlobalSize;
EXTERN int PetscIntView(int,int[],Viewer);
EXTERN int PetscDoubleView(int,double[],Viewer);
EXTERN int PetscScalarView(int,Scalar[],Viewer);

/*
    Allows accessing Matlab Engine
*/
#include "petscengine.h"

/*
    C code optimization is often enhanced by telling the compiler 
  that certain pointer arguments to functions are not aliased to 
  to other arguments. This is not yet ANSI C standard so we define 
  the macro "restrict" to indicate that the variable is not aliased 
  to any other argument.
*/
#if defined(PETSC_HAVE_RESTRICT) && !defined(__cplusplus)
#define restrict _Restrict
#else
#define restrict
#endif

/*
      Determine if some of the kernel computation routines use
   Fortran (rather than C) for the numerical calculations. On some machines
   and compilers (like complex numbers) the Fortran version of the routines
   is faster than the C/C++ versions. The flag PETSC_USE_FORTRAN_KERNELS  
   would be set in the petscconf.h file
*/
#if defined(PETSC_USE_FORTRAN_KERNELS)

#if !defined(PETSC_USE_FORTRAN_KERNEL_MULTAIJ)
#define PETSC_USE_FORTRAN_KERNEL_MULTAIJ
#endif

#if !defined(PETSC_USE_FORTRAN_KERNEL_NORMSQR)
#define PETSC_USE_FORTRAN_KERNEL_NORMSQR
#endif

#if !defined(PETSC_USE_FORTRAN_KERNEL_MAXPY)
#define PETSC_USE_FORTRAN_KERNEL_MAXPY
#endif

#if !defined(PETSC_USE_FORTRAN_KERNEL_SOLVEAIJ)
#define PETSC_USE_FORTRAN_KERNEL_SOLVEAIJ
#endif

#if !defined(PETSC_USE_FORTRAN_KERNEL_SOLVEBAIJ)
#define PETSC_USE_FORTRAN_KERNEL_SOLVEBAIJ
#endif

#if !defined(PETSC_USE_FORTRAN_KERNEL_MULTADDAIJ)
#define PETSC_USE_FORTRAN_KERNEL_MULTADDAIJ
#endif

#if !defined(PETSC_USE_FORTRAN_KERNEL_MDOT)
#define PETSC_USE_FORTRAN_KERNEL_MDOT
#endif

#if !defined(PETSC_USE_FORTRAN_KERNEL_XTIMESY)
#define PETSC_USE_FORTRAN_KERNEL_XTIMESY
#endif

#endif

/*
    Macros for indicating code that should be compiled with a C interface,
   rather than a C++ interface. Any routines that are dynamically loaded
   (such as the PCCreate_XXX() routines) must be wrapped so that the name
   mangler does not change the functions symbol name. This just hides the 
   ugly extern "C" {} wrappers.
*/
#if defined(__cplusplus)
#define EXTERN_C_BEGIN extern "C" {
#define EXTERN_C_END }
#else
#define EXTERN_C_BEGIN 
#define EXTERN_C_END 
#endif

/* --------------------------------------------------------------------*/
/*
    DVF (win32) uses STDCALL calling convention by default.
    The following is used by the fortran interface.
*/
#if defined (PETSC_USE_FORTRAN_STDCALL)
#define PETSC_STDCALL __stdcall
#else
#define PETSC_STDCALL 
#endif

#endif


/* $Id: petsc.h,v 1.256 1999/10/06 22:16:50 bsmith Exp balay $ */
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
   in the bmake/common definition of PETSC_INCLUDE
*/
#include "petscconf.h"

/* ========================================================================== */

#include <stdio.h>
/*
    Defines the interface to MPI allowing the use of all MPI functions.
*/
#include "mpi.h"

/*
    Defines some elementary mathematics functions and constants.
*/
#include "petscmath.h"

/*
    Variable type where we stash PETSc object pointers in Fortran.
    Assumes that sizeof(long) == sizeof(void *) which is true on 
    all machines that we know.
*/     
#define PetscFortranAddr   long

extern MPI_Comm PETSC_COMM_WORLD;
extern MPI_Comm PETSC_COMM_SELF;
extern int      PetscInitializedCalled;
extern int      PetscSetCommWorld(MPI_Comm);

/*
    Defines the malloc employed by PETSc. Users may use these routines as well. 
*/
#define PetscMalloc(a)       (*PetscTrMalloc)(a,__LINE__,__FUNC__,__FILE__,__SDIR__)
#define PetscNew(A)          (A*) PetscMalloc(sizeof(A))
#define PetscFree(a)         (*PetscTrFree)(a,__LINE__,__FUNC__,__FILE__,__SDIR__)
extern void *(*PetscTrMalloc)(int,int,char*,char*,char*);
extern int  (*PetscTrFree)(void *,int,char*,char*,char*);
extern int  PetscSetMalloc(void *(*)(int,int,char*,char*,char*),
                           int (*)(void *,int,char*,char*,char*));
extern int  PetscClearMalloc(void);

/*
   Routines for tracing memory corruption/bleeding with default PETSc 
   memory allocation
*/
extern int   PetscTrDump(FILE *);
extern int   PetscTrSpace(PLogDouble *, PLogDouble *,PLogDouble *);
extern int   PetscTrValid(int,const char[],const char[],const char[]);
extern int   PetscTrDebugLevel(int);
extern int   PetscTrLog(void);
extern int   PetscTrLogDump(FILE *);
extern int   PetscGetResidentSetSize(PLogDouble *);

/*
     Constants and functions used for handling different basic data types.
     These are used, for example, in binary IO routines
*/
typedef enum {PETSC_INT = 0, PETSC_DOUBLE = 1, PETSC_SHORT = 2, PETSC_FLOAT = 3,
              PETSC_COMPLEX = 4, PETSC_CHAR = 5, PETSC_LOGICAL = 6} PetscDataType;
#if defined(PETSC_USE_COMPLEX)
#define PETSC_SCALAR PETSC_COMPLEX
#else
#define PETSC_SCALAR PETSC_DOUBLE
#endif
typedef enum {PETSC_INT_SIZE = sizeof(int), PETSC_DOUBLE_SIZE = sizeof(double),
              PETSC_SCALAR_SIZE = sizeof(Scalar), PETSC_COMPLEX_SIZE = sizeof(double),
              PETSC_CHAR_SIZE = sizeof(char), PETSC_LOGICAL_SIZE = 1} PetscDataTypeSize;
extern int PetscDataTypeToMPIDataType(PetscDataType,MPI_Datatype*);
extern int PetscDataTypeGetSize(PetscDataType,int*);
extern int PetscDataTypeGetName(PetscDataType,char*[]);

/*
       Basic PETSc constants
*/
typedef enum { PETSC_FALSE, PETSC_TRUE } PetscTruth;
#define PETSC_NULL            0
#define PETSC_DECIDE         -1
#define PETSC_DETERMINE      PETSC_DECIDE
#define PETSC_DEFAULT        -2

/*
    Basic memory and string operations. These are usually simple wrappers
   around the basic Unix system calls, but a few of them have additional
   functionality and/or error checking.
*/
extern int   PetscMemcpy(void *,const void *,int);
extern int   PetscBitMemcpy(void*,int,const void*,int,int,PetscDataType);
extern int   PetscMemmove(void *,void *,int);
extern int   PetscMemzero(void *,int);
extern int   PetscMemcmp(const void*,const void*, int,PetscTruth *);
extern int   PetscStrlen(const char[],int*);
extern int   PetscStrcmp(const char[],const char[]);
extern int   PetscStrgrt(const char[],const char[],PetscTruth *);
extern int   PetscStrcasecmp(const char[],const char[],PetscTruth*);
extern int   PetscStrncmp(const char[],const char[],int,PetscTruth*);
extern int   PetscStrcpy(char[],const char[]);
extern int   PetscStrcat(char[],const char[]);
extern int   PetscStrncat(char[],const char[],int);
extern int   PetscStrncpy(char[],const char[],int);
extern int   PetscStrchr(const char[],char,char **);
extern int   PetscStrrchr(const char[],char,char **);
extern int   PetscStrstr(const char[],const char[],char **);
extern int   PetscStrtok(const char[],const char[],char **);
extern int   PetscStrallocpy(const char[],char **);

#define PetscTypeCompare(a,b) (!PetscStrcmp((char*)(((PetscObject)(a))->type_name),(char *)(b)))


/*
    Each PETSc object class has it's own cookie (internal integer in the 
  data structure used for error checking). These are all defined by an offset 
  from the lowest one, PETSC_COOKIE. If you increase these you must 
  increase the field sizes in petsc/src/sys/src/plog/plog.c
*/
#define PETSC_COOKIE                    1211211
#define LARGEST_PETSC_COOKIE_PREDEFINED PETSC_COOKIE + 30
#define LARGEST_PETSC_COOKIE_ALLOWED    PETSC_COOKIE + 50
extern int LARGEST_PETSC_COOKIE;

typedef struct _FList *FList;

#include "viewer.h"
#include "options.h"

extern int PetscGetTime(PLogDouble*);
extern int PetscGetCPUTime(PLogDouble*);
extern int PetscSleep(int);

/*
    Initialization of PETSc or its micro-kernel ALICE
*/
extern int  AliceInitialize(int*,char***,const char[],const char[]);
extern int  AliceInitializeNoArguments(void);
extern int  AliceFinalize(void);
extern void AliceInitializeFortran(void);

extern int  PetscInitialize(int*,char***,char[],const char[]);
extern int  PetscInitializeNoArguments(void);
extern int  PetscFinalize(void);
extern void PetscInitializeFortran(void);

/*
    Functions that can act on any PETSc object.
*/
typedef struct _p_PetscObject* PetscObject;
extern int PetscObjectDestroy(PetscObject);
extern int PetscObjectExists(PetscObject,PetscTruth*);
extern int PetscObjectGetComm(PetscObject,MPI_Comm *comm);
extern int PetscObjectGetCookie(PetscObject,int *cookie);
extern int PetscObjectGetType(PetscObject,int *type);
extern int PetscObjectSetName(PetscObject,const char[]);
extern int PetscObjectGetName(PetscObject,char*[]);
extern int PetscObjectReference(PetscObject);
extern int PetscObjectGetReference(PetscObject,int*);
extern int PetscObjectDereference(PetscObject);
extern int PetscObjectGetNewTag(PetscObject,int *);
extern int PetscObjectRestoreNewTag(PetscObject,int *);
extern int PetscCommGetNewTag(MPI_Comm,int *);
extern int PetscCommRestoreNewTag(MPI_Comm,int *);
extern int PetscObjectView(PetscObject,Viewer);
extern int PetscObjectCompose(PetscObject,const char[],PetscObject);
extern int PetscObjectQuery(PetscObject,const char[],PetscObject *);
extern int PetscObjectComposeFunction_Private(PetscObject,const char[],const char[],void *);
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define PetscObjectComposeFunction(a,b,c,d) PetscObjectComposeFunction_Private(a,b,c,0)
#else
#define PetscObjectComposeFunction(a,b,c,d) PetscObjectComposeFunction_Private(a,b,c,d)
#endif
extern int PetscObjectQueryFunction(PetscObject,const char[],void **);
extern int PetscObjectSetOptionsPrefix(PetscObject,const char[]);
extern int PetscObjectAppendOptionsPrefix(PetscObject,const char[]);
extern int PetscObjectPrependOptionsPrefix(PetscObject,const char[]);
extern int PetscObjectGetOptionsPrefix(PetscObject,char*[]);
extern int PetscObjectPublish(PetscObject);
extern int PetscObjectChangeTypeName(PetscObject,char *);

/*
    Defines PETSc error handling.
*/
#include "petscerror.h"

/*
    Mechanism for managing lists of objects attached (composed) with 
   a PETSc object.
*/
typedef struct _OList *OList;
extern int OListDestroy(OList *);
extern int OListFind(OList,const char[],PetscObject*);
extern int OListReverseFind(OList,PetscObject,char**);
extern int OListAdd(OList *,const char[],PetscObject);
extern int OListDuplicate(OList,OList *);

/*
    Dynamic library lists. Lists of names of routines in dynamic 
  link libraries that will be loaded as needed.
*/
extern int FListAdd_Private(FList*,const char[],const char[],int (*)(void *));
extern int FListDestroy(FList);
extern int FListFind(MPI_Comm,FList,const char[],int (**)(void*));
extern int FListPrintTypes(MPI_Comm,FILE*,const char[],const char[],FList);
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define    FListAdd(a,b,p,c) FListAdd_Private(a,b,p,0)
#else
#define    FListAdd(a,b,p,c) FListAdd_Private(a,b,p,(int (*)(void *))c)
#endif
extern int FListDuplicate(FList,FList *);
extern int FListView(FList,Viewer);
extern int FListConcat_Private(char *,char *, char *);

/*
   Routines for handling dynamic libraries. PETSc uses dynamic libraries
  by default on most machines (except IBM). This is controlled by the
  flag PETSC_USE_DYNAMIC_LIBRARIES in petscconf.h
*/
typedef struct _DLLibraryList *DLLibraryList;
extern DLLibraryList DLLibrariesLoaded;
extern int DLLibraryRetrieve(MPI_Comm,const char[],char *,int,PetscTruth *);
extern int DLLibraryOpen(MPI_Comm,const char[],void **);
extern int DLLibrarySym(MPI_Comm,DLLibraryList *,const char[],const char[],void **);
extern int DLLibraryAppend(MPI_Comm,DLLibraryList *,const char[]);
extern int DLLibraryPrepend(MPI_Comm,DLLibraryList *,const char[]);
extern int DLLibraryClose(DLLibraryList);
extern int DLLibraryPrintPath(void);
extern int DLLibraryGetInfo(void *,char *,char **);

/*
    Mechanism for translating PETSc object representations between languages
    Not currently used.
*/
typedef enum {PETSC_LANGUAGE_C,PETSC_LANGUAGE_CPP} PetscLanguage;
#define PETSC_LANGUAGE_F77 PETSC_LANGUAGE_C
extern int PetscObjectComposeLanguage(PetscObject,PetscLanguage,void *);
extern int PetscObjectQueryLanguage(PetscObject,PetscLanguage,void **);

/*
     Useful utility routines
*/
extern int PetscSplitOwnership(MPI_Comm,int*,int*);
extern int PetscSequentialPhaseBegin(MPI_Comm,int);
extern int PetscSequentialPhaseEnd(MPI_Comm,int);
extern int PetscBarrier(PetscObject);
extern int PetscMPIDump(FILE*);

/*
    Defines basic graphics available from PETSc.
*/
#include "draw.h"

/*
    Defines the base data structures for all PETSc objects
*/
#include "petschead.h"

/*
     Defines PETSc profiling.
*/
#include "petsclog.h"

#if defined(PETSC_HAVE_AMS)
extern PetscTruth PetscAMSPublishAll;
#define PetscPublishAll(v)\
  { if (PetscAMSPublishAll) { \
    int __ierr;\
    __ierr = PetscObjectPublish((PetscObject)v);CHKERRQ(__ierr);\
  }}
#else
#define PetscPublishAll(v)
#endif

/*
      This code allows one to pass a MPI communicator between 
    C and Fortran. MPI 2.0 defines a standard API for doing this.
    The code here is provided to allow PETSc to work with MPI 1.1
    standard MPI libraries.
*/
extern int  MPICCommToFortranComm(MPI_Comm,int *);
extern int  MPIFortranCommToCComm(int,MPI_Comm*);

/*
      Simple PETSc parallel IO for ASCII printing
*/
extern int  PetscFixFilename(const char[],char[]);
extern FILE *PetscFOpen(MPI_Comm,const char[],const char[]);
extern int  PetscFClose(MPI_Comm,FILE*);
extern int  PetscFPrintf(MPI_Comm,FILE*,const char[],...);
extern int  PetscPrintf(MPI_Comm,const char[],...);
extern int  (*PetscErrorPrintf)(const char[],...);
extern int  (*PetscHelpPrintf)(MPI_Comm,const char[],...);

extern int  PetscSynchronizedPrintf(MPI_Comm,const char[],...);
extern int  PetscSynchronizedFPrintf(MPI_Comm,FILE*,const char[],...);
extern int  PetscSynchronizedFlush(MPI_Comm);

/*
    Simple PETSc object that contains a pointer to any required data
*/
typedef struct _p_PetscObjectContainer*  PetscObjectContainer;
extern int PetscObjectContainerGetPointer(PetscObjectContainer,void **);
extern int PetscObjectContainerSetPointer(PetscObjectContainer,void *);
extern int PetscObjectContainerDestroy(PetscObjectContainer);
extern int PetscObjectContainerCreate(MPI_Comm comm,PetscObjectContainer *);

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
extern int PetscIntView(int,int[],Viewer);
extern int PetscDoubleView(int,double[],Viewer);
extern int PetscScalarView(int,Scalar[],Viewer);

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

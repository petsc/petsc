/* $Id: petsc.h,v 1.297 2001/09/07 20:13:16 bsmith Exp $ */
/*
   This is the main PETSc include file (for C and C++).  It is included by all
   other PETSc include files, so it almost never has to be specifically included.
*/
#if !defined(__PETSC_H)
#define __PETSC_H
/* ========================================================================== */
/* 
   This facilitates using C version of PETSc from C++
*/

#if defined(PETSC_USE_EXTERN_CXX) && defined(__cplusplus)
#define PETSC_EXTERN_CXX_BEGIN extern "C" {
#define PETSC_EXTERN_CXX_END  }
#else
#define PETSC_EXTERN_CXX_BEGIN
#define PETSC_EXTERN_CXX_END
#endif
/* ========================================================================== */
/* 
   Current PETSc version number and release date
*/
#include "petscversion.h"

/* ========================================================================== */
/* 
   petscconf.h is contained in bmake/${PETSC_ARCH}/petscconf.h it is 
   found automatically by the compiler due to the -I${PETSC_DIR}/bmake/${PETSC_ARCH}
   in the bmake/common_variables definition of PETSC_INCLUDE
*/
#include "petscconf.h"
/*
   Fixes for configure time choices which impact our interface. Currently only
   calling conventions and extra compiler checking falls under this category.
*/
#if !defined(PETSC_PRINTF_FORMAT_CHECK)
#define PETSC_PRINTF_FORMAT_CHECK(a,b)
#endif
#if !defined (PETSC_STDCALL)
#define PETSC_STDCALL
#endif
#if !defined (PETSC_TEMPLATE)
#define PETSC_TEMPLATE
#endif

/* ========================================================================== */

#include <stdio.h>
/*
    Defines the interface to MPI allowing the use of all MPI functions.
*/
#include "mpi.h"


/*
    Declare extern C stuff after incuding external header files
*/

PETSC_EXTERN_CXX_BEGIN

/*
    EXTERN indicates a PETSc function defined elsewhere
*/
#if !defined(EXTERN)
#define EXTERN extern
#endif

/*
    Defines some elementary mathematics functions and constants.
*/
#include "petscmath.h"

/*
       Basic PETSc constants
*/

/*E
    PetscTruth - Logical variable. Actually an integer

   Level: beginner

E*/
typedef enum { PETSC_FALSE,PETSC_TRUE } PetscTruth;

/*M
    PETSC_NULL - standard way of passing in a null or array or pointer

   Level: beginner

   Notes: accepted by many PETSc functions to not set a parameter and instead use
          some default

          This macro does not exist in Fortran; you must use PETSC_NULL_INTEGER, 
          PETSC_NULL_DOUBLE_PRECISION etc

.seealso: PETSC_DECIDE, PETSC_DEFAULT, PETSC_IGNORE, PETSC_DETERMINE

M*/
#define PETSC_NULL           0

/*M
    PETSC_DECIDE - standard way of passing in integer or floating point parameter
       where you wish PETSc to use the default.

   Level: beginner

.seealso: PETSC_NULL, PETSC_DEFAULT, PETSC_IGNORE, PETSC_DETERMINE

M*/
#define PETSC_DECIDE         -1

/*M
    PETSC_DEFAULT - standard way of passing in integer or floating point parameter
       where you wish PETSc to use the default.

   Level: beginner

.seealso: PETSC_DECIDE, PETSC_NULL, PETSC_IGNORE, PETSC_DETERMINE

M*/
#define PETSC_DEFAULT        -2

#define PETSC_YES            PETSC_TRUE
#define PETSC_NO             PETSC_FALSE

/*M
    PETSC_IGNORE - same as PETSC_NULL, means PETSc will ignore this argument

   Level: beginner

   Notes: accepted by many PETSc functions to not set a parameter and instead use
          some default

          This macro does not exist in Fortran; you must use PETSC_NULL_INTEGER, 
          PETSC_NULL_DOUBLE_PRECISION etc

.seealso: PETSC_DECIDE, PETSC_DEFAULT, PETSC_NULL, PETSC_DETERMINE

M*/
#define PETSC_IGNORE         PETSC_NULL

/*M
    PETSC_DETERMINE - standard way of passing in integer or floating point parameter
       where you wish PETSc to compute the required value.

   Level: beginner

.seealso: PETSC_DECIDE, PETSC_DEFAULT, PETSC_IGNORE, PETSC_NULL, VecSetSizes()

M*/
#define PETSC_DETERMINE      PETSC_DECIDE

/*M
    PETSC_COMM_WORLD - a duplicate of the MPI_COMM_WORLD communicator which represents
           all the processs

   Level: beginner

   Notes: PETSC_COMM_WORLD and MPI_COMM_WORLD are equivalent except that passing MPI_COMM_WORLD
          into PETSc object constructors will result in using more MPI resources since an MPI_Comm_dup()
          will be done on it internally. We recommend always using PETSC_COMM_WORLD

.seealso: PETSC_COMM_SELF

M*/
extern MPI_Comm   PETSC_COMM_WORLD;

/*M
    PETSC_COMM_SELF - a duplicate of the MPI_COMM_SELF communicator which represents
           the current process

   Level: beginner

   Notes: PETSC_COMM_SELF and MPI_COMM_SELF are equivalent except that passint MPI_COMM_SELF
          into PETSc object constructors will result in using more MPI resources since an MPI_Comm_dup()
          will be done on it internally. We recommend always using PETSC_COMM_SELF

.seealso: PETSC_COMM_WORLD

M*/
extern MPI_Comm   PETSC_COMM_SELF;

extern PetscTruth PetscInitializeCalled;
EXTERN int        PetscSetCommWorld(MPI_Comm);
EXTERN int        PetscSetHelpVersionFunctions(int (*)(MPI_Comm),int (*)(MPI_Comm));

/*MC
   PetscMalloc - Allocates memory

   Input Parameter:
.  m - number of bytes to allocate

   Output Parameter:
.  result - memory allocated

   Synopsis:
   int PetscMalloc(size_t m,void **result)

   Level: beginner

   Notes: Memory is always allocated at least double aligned

.seealso: PetscFree(), PetscNew()

  Concepts: memory allocation

M*/
#define PetscMalloc(a,b)     (*PetscTrMalloc)((a),__LINE__,__FUNCT__,__FILE__,__SDIR__,(void**)(b))
/*MC
   PetscNew - Allocates memory of a particular type

   Input Parameter:
. type - structure name of space to be allocated. Memory of size sizeof(type) is allocated

   Output Parameter:
.  result - memory allocated

   Synopsis:
   int PetscNew(struct type,((type *))result)

   Level: beginner

.seealso: PetscFree(), PetscMalloc()

  Concepts: memory allocation

M*/
#define PetscNew(A,b)        PetscMalloc(sizeof(A),(b))
/*MC
   PetscFree - Frees memory

   Input Parameter:
.   memory - memory to free

   Synopsis:
   int PetscFree(void *memory)

   Level: beginner

   Notes: Memory must have been obtained with PetscNew() or PetscMalloc()

.seealso: PetscNew(), PetscMalloc()

  Concepts: memory allocation

M*/
#define PetscFree(a)         (*PetscTrFree)((a),__LINE__,__FUNCT__,__FILE__,__SDIR__)
EXTERN int  (*PetscTrMalloc)(size_t,int,const char[],const char[],const char[],void**);
EXTERN int  (*PetscTrFree)(void *,int,const char[],const char[],const char[]);
EXTERN int  PetscSetMalloc(int (*)(size_t,int,const char[],const char[],const char[],void**),int (*)(void *,int,const char[],const char[],const char[]));
EXTERN int  PetscClearMalloc(void);

/*
   Routines for tracing memory corruption/bleeding with default PETSc 
   memory allocation
*/
EXTERN int   PetscTrDump(FILE *);
EXTERN int   PetscTrSpace(PetscLogDouble *,PetscLogDouble *,PetscLogDouble *);
EXTERN int   PetscTrValid(int,const char[],const char[],const char[]);
EXTERN int   PetscTrDebugLevel(int);
EXTERN int   PetscTrLog(void);
EXTERN int   PetscTrLogDump(FILE *);
EXTERN int   PetscGetResidentSetSize(PetscLogDouble *);

/*
    Variable type where we stash PETSc object pointers in Fortran.
    Assumes that sizeof(long) == sizeof(void*)which is true on 
    all machines that we know.
*/     
#define PetscFortranAddr   long

/*E
    PetscDataType - Used for handling different basic data types.

   Level: beginner

.seealso: PetscBinaryRead(), PetscBinaryWrite(), PetscDataTypeToMPIDataType(),
          PetscDataTypeGetSize(), PetscDataTypeGetName()

E*/
typedef enum {PETSC_INT = 0,PETSC_DOUBLE = 1,PETSC_COMPLEX = 2,
              PETSC_LONG =3 ,PETSC_SHORT = 4,PETSC_FLOAT = 5,
              PETSC_CHAR = 6,PETSC_LOGICAL = 7} PetscDataType;
#if defined(PETSC_USE_COMPLEX)
#define PETSC_SCALAR PETSC_COMPLEX
#else
#if defined(PETSC_USE_SINGLE)
#define PETSC_SCALAR PETSC_FLOAT
#else
#define PETSC_SCALAR PETSC_DOUBLE
#endif
#endif
#if defined(PETSC_USE_SINGLE)
#define PETSC_REAL PETSC_FLOAT
#else
#define PETSC_REAL PETSC_DOUBLE
#endif
#define PETSC_FORTRANADDR PETSC_LONG

EXTERN int PetscDataTypeToMPIDataType(PetscDataType,MPI_Datatype*);
EXTERN int PetscDataTypeGetSize(PetscDataType,int*);
EXTERN int PetscDataTypeGetName(PetscDataType,const char*[]);

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
EXTERN int   PetscStrchr(const char[],char,char *[]);
EXTERN int   PetscStrtolower(char[]);
EXTERN int   PetscStrrchr(const char[],char,char *[]);
EXTERN int   PetscStrstr(const char[],const char[],char *[]);
EXTERN int   PetscStrallocpy(const char[],char *[]);
EXTERN int   PetscStrreplace(MPI_Comm,const char[],char[],int);
#define      PetscStrfree(a) ((a) ? PetscFree(a) : 0) 
typedef struct {char token;char *array;char *current;} PetscToken;
EXTERN int   PetscTokenCreate(const char[],const char,PetscToken**);
EXTERN int   PetscTokenFind(PetscToken*,char *[]);
EXTERN int   PetscTokenDestroy(PetscToken*);

/*
   These are  MPI operations for MPI_Allreduce() etc
*/
EXTERN MPI_Op PetscMaxSum_Op;
#if defined(PETSC_USE_COMPLEX)
EXTERN MPI_Op PetscSum_Op;
#else
#define PetscSum_Op MPI_SUM
#endif
EXTERN int PetscMaxSum(MPI_Comm,const int[],int*,int*);

/*S
     PetscObject - any PETSc object, PetscViewer, Mat, Vec, KSP etc

   Level: beginner

   Note: This is the base class from which all objects appear.

.seealso:  PetscObjectDestroy(), PetscObjectView(), PetscObjectGetName(), PetscObjectSetName()
S*/
typedef struct _p_PetscObject* PetscObject;

/*S
     PetscFList - Linked list of functions, possibly stored in dynamic libraries, accessed
      by string name

   Level: advanced

.seealso:  PetscFListAdd(), PetscFListDestroy()
S*/
typedef struct _PetscFList *PetscFList;

#include "petscviewer.h"
#include "petscoptions.h"

EXTERN int PetscShowMemoryUsage(PetscViewer,const char[]);
EXTERN int PetscGetTime(PetscLogDouble*);
EXTERN int PetscGetCPUTime(PetscLogDouble*);
EXTERN int PetscSleep(int);

/*
    Initialization of PETSc
*/
EXTERN int  PetscInitialize(int*,char***,const char[],const char[]);
EXTERN int  PetscInitializeNoArguments(void);
EXTERN int  PetscFinalize(void);
EXTERN int  PetscInitializeFortran(void);
EXTERN int  PetscGetArgs(int*,char ***);
EXTERN int  PetscEnd(void);

/*
   ParameterDict is an abstraction for arguments to interface mechanisms
*/
extern int DICT_COOKIE;
typedef struct _p_Dict *ParameterDict;

typedef void (**PetscVoidFunction)(void);

/*
   PetscTryMethod - Queries an object for a method, if it exists then calls it.
          Can support argument checking. This was an experiment that seems to work ok
          (see KSPGMRESSetRestart()) but we never followed through on it.
*/
#if defined(PETSC_FORTRAN_STUBS)
#define  PetscTryMethod(obj,A,B,C) \
  0;{ int (*f)B; \
   *ierr = PetscObjectQueryFunction((PetscObject)obj,#A,(PetscVoidFunction)&f);if (*ierr) return; \
    if (f) {*ierr = (*f)C;if (*ierr) return;}\
  }
#else
#define  PetscTryMethod(obj,A,B,C) \
  0;{ int (*f)B, __ierr; \
    __ierr = PetscObjectQueryFunction((PetscObject)obj,#A,(PetscVoidFunction)&f);CHKERRQ(__ierr); \
    if (f) {__ierr = (*f)C;CHKERRQ(__ierr);}\
  }
#endif

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
EXTERN int PetscObjectSetParameterDict(PetscObject,ParameterDict);
EXTERN int PetscObjectGetParameterDict(PetscObject,ParameterDict*);
EXTERN int PetscCommGetNewTag(MPI_Comm,int *);
EXTERN int PetscObjectView(PetscObject,PetscViewer);
EXTERN int PetscObjectCompose(PetscObject,const char[],PetscObject);
EXTERN int PetscObjectQuery(PetscObject,const char[],PetscObject *);
EXTERN int PetscObjectComposeFunction(PetscObject,const char[],const char[],void (*)(void));

/*MC
   PetscObjectComposeFunctionDynamic - Associates a function with a given PETSc object. 
                       
   Collective on PetscObject

   Input Parameters:
+  obj - the PETSc object; this must be cast with a (PetscObject), for example, 
         PetscObjectCompose((PetscObject)mat,...);
.  name - name associated with the child function
.  fname - name of the function
-  ptr - function pointer (or PETSC_NULL if using dynamic libraries)

   Level: advanced

    Synopsis:
    int PetscObjectComposeFunctionDynamic(PetscObject obj,const char name[],const char fname[],void *ptr)

   Notes:
   PetscObjectComposeFunctionDynamic() can be used with any PETSc object (such as
   Mat, Vec, KSP, SNES, etc.) or any user-provided object. 

   The composed function must be wrapped in a EXTERN_C_BEGIN/END for this to
   work in C++/complex with dynamic link libraries (PETSC_USE_DYNAMIC_LIBRARIES)
   enabled.

   Concepts: objects^composing functions
   Concepts: composing functions
   Concepts: functions^querying
   Concepts: objects^querying
   Concepts: querying objects

.seealso: PetscObjectQueryFunction()
M*/
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define PetscObjectComposeFunctionDynamic(a,b,c,d) PetscObjectComposeFunction(a,b,c,0)
#else
#define PetscObjectComposeFunctionDynamic(a,b,c,d) PetscObjectComposeFunction(a,b,c,(void (*)(void))(d))
#endif

EXTERN int PetscObjectQueryFunction(PetscObject,const char[],void (**)(void));
EXTERN int PetscObjectSetOptionsPrefix(PetscObject,const char[]);
EXTERN int PetscObjectAppendOptionsPrefix(PetscObject,const char[]);
EXTERN int PetscObjectPrependOptionsPrefix(PetscObject,const char[]);
EXTERN int PetscObjectGetOptionsPrefix(PetscObject,char*[]);
EXTERN int PetscObjectPublish(PetscObject);
EXTERN int PetscObjectChangeTypeName(PetscObject,const char[]);
EXTERN int PetscObjectRegisterDestroy(PetscObject);
EXTERN int PetscObjectRegisterDestroyAll(void);
EXTERN int PetscObjectName(PetscObject);
EXTERN int PetscTypeCompare(PetscObject,const char[],PetscTruth*);

/*
    Defines PETSc error handling.
*/
#include "petscerror.h"

/*S
     PetscOList - Linked list of PETSc objects, accessable by string name

   Level: advanced

.seealso:  PetscOListAdd(), PetscOListDestroy(), PetscOListFind()
S*/
typedef struct _PetscOList *PetscOList;

EXTERN int PetscOListDestroy(PetscOList *);
EXTERN int PetscOListFind(PetscOList,const char[],PetscObject*);
EXTERN int PetscOListReverseFind(PetscOList,PetscObject,char**);
EXTERN int PetscOListAdd(PetscOList *,const char[],PetscObject);
EXTERN int PetscOListDuplicate(PetscOList,PetscOList *);

/*
    Dynamic library lists. Lists of names of routines in dynamic 
  link libraries that will be loaded as needed.
*/
EXTERN int PetscFListAdd(PetscFList*,const char[],const char[],void (*)(void));
EXTERN int PetscFListDestroy(PetscFList*);
EXTERN int PetscFListFind(MPI_Comm,PetscFList,const char[],void (**)(void));
EXTERN int PetscFListPrintTypes(MPI_Comm,FILE*,const char[],const char[],const char[],const char[],PetscFList);
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define    PetscFListAddDynamic(a,b,p,c) PetscFListAdd(a,b,p,0)
#else
#define    PetscFListAddDynamic(a,b,p,c) PetscFListAdd(a,b,p,(void (*)(void))c)
#endif
EXTERN int PetscFListDuplicate(PetscFList,PetscFList *);
EXTERN int PetscFListView(PetscFList,PetscViewer);
EXTERN int PetscFListConcat(const char [],const char [],char []);
EXTERN int PetscFListGet(PetscFList,char ***,int*);

/*S
     PetscDLLibraryList - Linked list of dynamics libraries to search for functions

   Level: advanced

   PETSC_USE_DYNAMIC_LIBRARIES must be defined in petscconf.h to use dynamic libraries

.seealso:  PetscDLLibraryOpen()
S*/
typedef struct _PetscDLLibraryList *PetscDLLibraryList;
extern PetscDLLibraryList DLLibrariesLoaded;
EXTERN int PetscDLLibraryRetrieve(MPI_Comm,const char[],char *,int,PetscTruth *);
EXTERN int PetscDLLibraryOpen(MPI_Comm,const char[],void **);
EXTERN int PetscDLLibrarySym(MPI_Comm,PetscDLLibraryList *,const char[],const char[],void **);
EXTERN int PetscDLLibraryAppend(MPI_Comm,PetscDLLibraryList *,const char[]);
EXTERN int PetscDLLibraryPrepend(MPI_Comm,PetscDLLibraryList *,const char[]);
EXTERN int PetscDLLibraryClose(PetscDLLibraryList);
EXTERN int PetscDLLibraryPrintPath(void);
EXTERN int PetscDLLibraryGetInfo(void *,const char[],const char *[]);

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
EXTERN int PetscSplitOwnershipBlock(MPI_Comm,int,int*,int*);
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
#define PetscPublishAll(v) (PetscAMSPublishAll ? PetscObjectPublish((PetscObject)v) : 0)
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
EXTERN int  PetscFPrintf(MPI_Comm,FILE*,const char[],...) PETSC_PRINTF_FORMAT_CHECK(3,4);
EXTERN int  PetscPrintf(MPI_Comm,const char[],...)  PETSC_PRINTF_FORMAT_CHECK(2,3);

/*MC
    PetscErrorPrintf - Prints error messages.

    Not Collective

   Synopsis:
     int (*PetscErrorPrintf)(const char format[],...);

    Input Parameters:
.   format - the usual printf() format string 

   Options Database Keys:
.    -error_output_stderr - cause error messages to be printed to stderr instead of the
         (default) stdout


   Level: developer

    Fortran Note:
    This routine is not supported in Fortran.

    Concepts: error messages^printing
    Concepts: printing^error messages

.seealso: PetscFPrintf(), PetscSynchronizedPrintf(), PetscHelpPrintf()
M*/
EXTERN int  (*PetscErrorPrintf)(const char[],...);

/*MC
    PetscHelpPrintf - Prints help messages.

    Not Collective

   Synopsis:
     int (*PetscHelpPrintf)(const char format[],...);

    Input Parameters:
.   format - the usual printf() format string 

   Level: developer

    Fortran Note:
    This routine is not supported in Fortran.

    Concepts: help messages^printing
    Concepts: printing^help messages

.seealso: PetscFPrintf(), PetscSynchronizedPrintf(), PetscErrorPrintf()
M*/
EXTERN int  (*PetscHelpPrintf)(MPI_Comm,const char[],...);

EXTERN int  PetscPOpen(MPI_Comm,const char[],const char[],const char[],FILE **);
EXTERN int  PetscPClose(MPI_Comm,FILE*);
EXTERN int  PetscSynchronizedPrintf(MPI_Comm,const char[],...) PETSC_PRINTF_FORMAT_CHECK(2,3);
EXTERN int  PetscSynchronizedFPrintf(MPI_Comm,FILE*,const char[],...) PETSC_PRINTF_FORMAT_CHECK(3,4);
EXTERN int  PetscSynchronizedFlush(MPI_Comm);
EXTERN int  PetscSynchronizedFGets(MPI_Comm,FILE*,int,char[]);
EXTERN int  PetscStartMatlab(MPI_Comm,const char[],const char[],FILE**);
EXTERN int  PetscStartJava(MPI_Comm,const char[],const char[],FILE**);
EXTERN int  PetscGetPetscDir(const char*[]);

EXTERN int  PetscPopUpSelect(MPI_Comm,char*,char*,int,char**,int*);
/*S
     PetscObjectContainer - Simple PETSc object that contains a pointer to any required data

   Level: advanced

.seealso:  PetscObject, PetscObjectContainerCreate()
S*/
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
EXTERN int        PetscCompareScalar(PetscScalar);
EXTERN int        PetscCompareInt(int);

/*
   For use in debuggers 
*/
extern int PetscGlobalRank,PetscGlobalSize;
EXTERN int PetscIntView(int,int[],PetscViewer);
EXTERN int PetscRealView(int,PetscReal[],PetscViewer);
EXTERN int PetscScalarView(int,PetscScalar[],PetscViewer);

/*
    Allows accessing Matlab Engine
*/
#include "petscmatlab.h"

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
#if defined(restrict)
#undef restrict
#endif
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

#if !defined(PETSC_USE_FORTRAN_KERNEL_MULTTRANSPOSEAIJ)
#define PETSC_USE_FORTRAN_KERNEL_MULTTRANSPOSEAIJ
#endif

#if !defined(PETSC_USE_FORTRAN_KERNEL_NORM)
#define PETSC_USE_FORTRAN_KERNEL_NORM
#endif

#if !defined(PETSC_USE_FORTRAN_KERNEL_MAXPY)
#define PETSC_USE_FORTRAN_KERNEL_MAXPY
#endif

#if !defined(PETSC_USE_FORTRAN_KERNEL_SOLVEAIJ)
#define PETSC_USE_FORTRAN_KERNEL_SOLVEAIJ
#endif

#if !defined(PETSC_USE_FORTRAN_KERNEL_RELAXAIJ)
#define PETSC_USE_FORTRAN_KERNEL_RELAXAIJ
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

#if !defined(PETSC_USE_FORTRAN_KERNEL_AYPX)
#define PETSC_USE_FORTRAN_KERNEL_AYPX
#endif

#if !defined(PETSC_USE_FORTRAN_KERNEL_WAXPY)
#define PETSC_USE_FORTRAN_KERNEL_WAXPY
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

/*M
    size - integer variable used to contain the number of processors in
           the relevent MPI_Comm

   Level: beginner

.seealso: rank, comm
M*/

/*M
    rank - integer variable used to contain the number of this processor relative
           to all in the relevent MPI_Comm

   Level: beginner

.seealso: size, comm
M*/

/*M
    comm - MPI_Comm used in the current routine or object

   Level: beginner

.seealso: size, rank
M*/

/*M
    MPI_Comm - the basic object used by MPI to determine which processes are involved in a 
        communication

   Level: beginner

   Note: This manual page is a place-holder because MPICH does not have a manual page for MPI_Comm

.seealso: size, rank, comm, PETSC_COMM_WORLD, PETSC_COMM_SELF
M*/

/*M
    PetscScalar - PETSc type that represents either a double precision real number or 
       a double precision complex number if the code is compiled with BOPT=g_complex or O_complex

   Level: beginner

.seealso: PetscReal, PassiveReal, PassiveScalar
M*/

/*M
    PetscReal - PETSc type that represents a double precision real number

   Level: beginner

.seealso: PetscScalar, PassiveReal, PassiveScalar
M*/

/*M
    PassiveScalar - PETSc type that represents either a double precision real number or 
       a double precision complex number if the code is compiled with BOPT=g_complex or O_complex

   Level: beginner

    This is the same as a PetscScalar except in code that is automatically differentiated it is
   treated as a constant (not an indendent or dependent variable)

.seealso: PetscReal, PassiveReal, PetscScalar
M*/

/*M
    PassiveReal - PETSc type that represents a double precision real number

   Level: beginner

    This is the same as a PetscReal except in code that is automatically differentiated it is
   treated as a constant (not an indendent or dependent variable)

.seealso: PetscScalar, PetscReal, PassiveScalar
M*/

/*
     The IBM include files define hz, here we hide it so that it may be used
   as a regular user variable.
*/
#if defined(hz)
#undef hz
#endif

/*  For arrays that contain filenames or paths */


#if defined(PETSC_HAVE_LIMITS_H)
#include <limits.h>
#endif
#if defined(PETSC_HAVE_SYS_PARAM_H)
#include <sys/param.h>
#endif

#if defined(MAXPATHLEN)
#  define PETSC_MAX_PATH_LEN     MAXPATHLEN
#elif defined(MAX_PATH)
#  define PETSC_MAX_PATH_LEN     MAX_PATH
#elif defined(_MAX_PATH)
#  define PETSC_MAX_PATH_LEN     _MAX_PATH
#else
#  define PETSC_MAX_PATH_LEN     4096
#endif

PETSC_EXTERN_CXX_END
#endif



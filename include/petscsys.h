/*
    Provides access to system related and general utility routines.
*/
#if !defined(__PETSCSYS_H)
#define __PETSCSYS_H
#include "petsc.h"
PETSC_EXTERN_CXX_BEGIN

EXTERN int  PetscGetArchType(char[],int);
EXTERN int  PetscGetHostName(char[],int);
EXTERN int  PetscGetUserName(char[],int);
EXTERN int  PetscGetProgramName(char[],int);
EXTERN int  PetscSetProgramName(const char[]);
EXTERN int  PetscGetDate(char[],int);
EXTERN int  PetscSetInitialDate(void);
EXTERN int  PetscGetInitialDate(char[],int);

EXTERN int  PetscSortInt(int,int[]);
EXTERN int  PetscSortIntWithPermutation(int,const int[],int[]);
EXTERN int  PetscSortStrWithPermutation(int,const char*[],int[]);
EXTERN int  PetscSortIntWithArray(int,int[],int[]);
EXTERN int  PetscSortReal(int,PetscReal[]);
EXTERN int  PetscSortRealWithPermutation(int,const PetscReal[],int[]);

EXTERN int  PetscSetDisplay(void);
EXTERN int  PetscGetDisplay(char[],int);

extern int PETSC_RANDOM_COOKIE;

typedef enum { RANDOM_DEFAULT,RANDOM_DEFAULT_REAL,
               RANDOM_DEFAULT_IMAGINARY } PetscRandomType;

/*S
     PetscRandom - Abstract PETSc object that manages generating random numbers

   Level: intermediate

  Concepts: random numbers

.seealso:  PetscRandomCreate(), PetscRandomGetValue()
S*/
typedef struct _p_PetscRandom*   PetscRandom;

EXTERN int PetscRandomCreate(MPI_Comm,PetscRandomType,PetscRandom*);
EXTERN int PetscRandomGetValue(PetscRandom,PetscScalar*);
EXTERN int PetscRandomSetInterval(PetscRandom,PetscScalar,PetscScalar);
EXTERN int PetscRandomDestroy(PetscRandom);

EXTERN int PetscGetFullPath(const char[],char[],int);
EXTERN int PetscGetRelativePath(const char[],char[],int);
EXTERN int PetscGetWorkingDirectory(char[],int);
EXTERN int PetscGetRealPath(char[],char[]);
EXTERN int PetscGetHomeDirectory(char[],int);
EXTERN int PetscTestFile(const char[],char,PetscTruth*);
EXTERN int PetscTestDirectory(const char[],char,PetscTruth*);
EXTERN int PetscBinaryRead(int,void*,int,PetscDataType);
EXTERN int PetscSynchronizedBinaryRead(MPI_Comm,int,void*,int,PetscDataType);
EXTERN int PetscBinaryWrite(int,void*,int,PetscDataType,int);
EXTERN int PetscBinaryOpen(const char[],int,int *);
EXTERN int PetscBinaryClose(int);
EXTERN int PetscSharedTmp(MPI_Comm,PetscTruth *);
EXTERN int PetscSharedWorkingDirectory(MPI_Comm,PetscTruth *);
EXTERN int PetscGetTmp(MPI_Comm,char *,int);
EXTERN int PetscFileRetrieve(MPI_Comm,const char *,char *,int,PetscTruth*);
EXTERN int PetscLs(MPI_Comm,const char[],char*,int,PetscTruth*);
EXTERN int PetscDLLibraryCCAAppend(MPI_Comm,PetscDLLibraryList*,const char[]);

/*
   In binary files variables are stored using the following lengths,
  regardless of how they are stored in memory on any one particular
  machine. Use these rather then sizeof() in computing sizes for 
  PetscBinarySeek().
*/
#define PETSC_BINARY_INT_SIZE    (32/8)
#define PETSC_BINARY_FLOAT_SIZE  (32/8)
#define PETSC_BINARY_CHAR_SIZE    (8/8)
#define PETSC_BINARY_SHORT_SIZE  (16/8)
#define PETSC_BINARY_DOUBLE_SIZE (64/8)
#define PETSC_BINARY_SCALAR_SIZE sizeof(PetscScalar)

/*E
  PetscBinarySeekType - argument to PetscBinarySeek()

  Level: advanced

.seealso: PetscBinarySeek(), PetscSynchronizedBinarySeek()
E*/
typedef enum {PETSC_BINARY_SEEK_SET = 0,PETSC_BINARY_SEEK_CUR = 1,PETSC_BINARY_SEEK_END = 2} PetscBinarySeekType;
EXTERN int PetscBinarySeek(int,int,PetscBinarySeekType,int*);
EXTERN int PetscSynchronizedBinarySeek(MPI_Comm,int,int,PetscBinarySeekType,int*);

EXTERN int PetscSetDebugger(const char[],PetscTruth);
EXTERN int PetscSetDefaultDebugger(void);
EXTERN int PetscSetDebuggerFromString(char*);
EXTERN int PetscAttachDebugger(void);
EXTERN int PetscStopForDebugger(void);

EXTERN int PetscGatherNumberOfMessages(MPI_Comm,int*,int*,int*);
EXTERN int PetscGatherMessageLengths(MPI_Comm,int,int,int*,int**,int**);
EXTERN int PetscPostIrecvInt(MPI_Comm,int,int,int*,int*,int***,MPI_Request**);
EXTERN int PetscPostIrecvScalar(MPI_Comm,int,int,int*,int*,PetscScalar***,MPI_Request**);

EXTERN int PetscSSEIsEnabled(MPI_Comm,PetscTruth *,PetscTruth *);

/* ParameterDict objects encapsulate arguments to generic functions, like mechanisms over interfaces */
EXTERN int ParameterDictCreate(MPI_Comm, ParameterDict *);
EXTERN int ParameterDictDestroy(ParameterDict);
EXTERN int ParameterDictRemove(ParameterDict, const char []);
EXTERN int ParameterDictSetInteger(ParameterDict, const char [], int);
EXTERN int ParameterDictSetDouble(ParameterDict, const char [], double);
EXTERN int ParameterDictSetObject(ParameterDict, const char [], void *);
EXTERN int ParameterDictGetInteger(ParameterDict, const char [], int *);
EXTERN int ParameterDictGetDouble(ParameterDict, const char [], double *);
EXTERN int ParameterDictGetObject(ParameterDict, const char [], void **);

/* Parallel communication routines */
/*E
  InsertMode - Whether entries are inserted or added into vectors or matrices

  Level: beginner

.seealso: VecSetValues(), MatSetValues(), VecSetValue(), VecSetValuesBlocked(),
          VecSetValuesLocal(), VecSetValuesBlockedLocal(), MatSetValuesBlocked(),
          MatSetValuesBlockedLocal(), MatSetValuesLocal(), VecScatterBegin(), VecScatterEnd()
E*/
typedef enum {NOT_SET_VALUES, INSERT_VALUES, ADD_VALUES, MAX_VALUES} InsertMode;

/*M
    INSERT_VALUES - Put a value into a vector or matrix, overwrites any previous value

    Level: beginner

.seealso: InsertMode, VecSetValues(), MatSetValues(), VecSetValue(), VecSetValuesBlocked(),
          VecSetValuesLocal(), VecSetValuesBlockedLocal(), MatSetValuesBlocked(), ADD_VALUES, INSERT_VALUES,
          MatSetValuesBlockedLocal(), MatSetValuesLocal(), VecScatterBegin(), VecScatterEnd()

M*/

/*M
    ADD_VALUES - Adds a value into a vector or matrix, if there previously was no value, just puts the
                value into that location

    Level: beginner

.seealso: InsertMode, VecSetValues(), MatSetValues(), VecSetValue(), VecSetValuesBlocked(),
          VecSetValuesLocal(), VecSetValuesBlockedLocal(), MatSetValuesBlocked(), ADD_VALUES, INSERT_VALUES,
          MatSetValuesBlockedLocal(), MatSetValuesLocal(), VecScatterBegin(), VecScatterEnd()

M*/

/*M
    MAX_VALUES - Puts the maximum of the scattered/gathered value and the current value into each location

    Level: beginner

.seealso: InsertMode, VecScatterBegin(), VecScatterEnd(), ADD_VALUES, INSERT_VALUES

M*/

/*E
  ScatterMode - Determines the direction of a scatter

  Level: beginner

.seealso: VecScatter, VecScatterBegin(), VecScatterEnd()
E*/
typedef enum {SCATTER_FORWARD=0, SCATTER_REVERSE=1, SCATTER_FORWARD_LOCAL=2, SCATTER_REVERSE_LOCAL=3, SCATTER_LOCAL=2} ScatterMode;

/*M
    SCATTER_FORWARD - Scatters the values as dictated by the VecScatterCreate() call

    Level: beginner

.seealso: VecScatter, ScatterMode, VecScatterCreate(), VecScatterBegin(), VecScatterEnd(), SCATTER_REVERSE, SCATTER_FORWARD_LOCAL,
          SCATTER_REVERSE_LOCAL

M*/

/*M
    SCATTER_REVERSE - Moves the values in the opposite direction then the directions indicated in
         in the VecScatterCreate()

    Level: beginner

.seealso: VecScatter, ScatterMode, VecScatterCreate(), VecScatterBegin(), VecScatterEnd(), SCATTER_FORWARD, SCATTER_FORWARD_LOCAL,
          SCATTER_REVERSE_LOCAL

M*/

/*M
    SCATTER_FORWARD_LOCAL - Scatters the values as dictated by the VecScatterCreate() call except NO parallel communication
       is done. Any variables that have be moved between processes are ignored

    Level: developer

.seealso: VecScatter, ScatterMode, VecScatterCreate(), VecScatterBegin(), VecScatterEnd(), SCATTER_REVERSE, SCATTER_FORWARD,
          SCATTER_REVERSE_LOCAL

M*/

/*M
    SCATTER_REVERSE_LOCAL - Moves the values in the opposite direction then the directions indicated in
         in the VecScatterCreate()  except NO parallel communication
       is done. Any variables that have be moved between processes are ignored

    Level: developer

.seealso: VecScatter, ScatterMode, VecScatterCreate(), VecScatterBegin(), VecScatterEnd(), SCATTER_FORWARD, SCATTER_FORWARD_LOCAL,
          SCATTER_REVERSE

M*/

EXTERN int PetscGhostExchange(MPI_Comm, int, int *, int *, PetscDataType, int *, InsertMode, ScatterMode, void *, void *);

/* 
  Initialize a linked list 
  Input Parameters:
    lnk_init  - the initial index value indicating the entry in the list is not set yet
    nlnk      - max length of the list
    lnk       - linked list(an integer array) that is allocated
  output Parameters:
    lnk       - the linked list with all values set as lnk_int
*/
#define PetscLLInitialize(lnk_init,nlnk,lnk) 0;\
{\
  int _i;\
  for (_i=0; _i<nlnk; _i++) lnk[_i] = lnk_init;\
}

/*
  Add a index set into a sorted linked list
  Input Parameters:
    nidx      - number of input indices
    indices   - interger array
    lnk_head  - the header of the list
    lnk_init  - the initial index value indicating the entry in the list is not set yet
    lnk       - linked list(an integer array) that is created
  output Parameters:
    nlnk      - number of newly added indices
    lnk       - the sorted(increasing order) linked list containing new and non-redundate entries from indices
*/
#define PetscLLAdd(nidx,indices,lnk_head,lnk_init,nlnk,lnk) 0;\
{\
  int _k,_entry,_location,_lnkdata;\
  nlnk = 0;\
  _k=nidx;\
  while (_k){/* assume indices are almost in increasing order, starting from its end saves computation */\
    _entry = indices[--_k];\
    /* search for insertion location */\
    _lnkdata  = lnk_head;\
    do {\
      _location = _lnkdata;\
      _lnkdata  = lnk[_location];\
    } while (_entry > _lnkdata);\
    /* insertion location is found, add entry into lnk if it is new */\
    if (_entry <  _lnkdata){/* new entry */\
      lnk[_location] = _entry;\
      lnk[_entry]    = _lnkdata;\
      nlnk++;\
    }\
  }\
}
/*
  Copy data on the list into an array, then initialize the list 
  Input Parameters:
    lnk_head  - the header of the list
    lnk_init  - the initial index value indicating the entry in the list is not set yet
    nlnk      - number of data on the list to be copied
    lnk       - linked list
  output Parameters:
    indices   - array that contains the copied data
*/
#define PetscLLClear(lnk_head,lnk_init,nlnk,lnk,indices) 0;\
{\
  int _j,_idx=lnk_head,_idx0;\
  for (_j=0; _j<nlnk; _j++){\
    _idx0 = _idx; _idx = lnk[_idx0];\
    *(indices+_j) = _idx;\
    lnk[_idx0] = lnk_init;\
  }\
  lnk[_idx] = lnk_init;\
}

PETSC_EXTERN_CXX_END
#endif /* __PETSCSYS_H */

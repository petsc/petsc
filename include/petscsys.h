/*
    Provides access to system related and general utility routines.
*/
#if !defined(__PETSCSYS_H)
#define __PETSCSYS_H
#include "petsc.h"
PETSC_EXTERN_CXX_BEGIN

EXTERN PetscErrorCode  PetscGetArchType(char[],size_t);
EXTERN PetscErrorCode  PetscGetHostName(char[],size_t);
EXTERN PetscErrorCode  PetscGetUserName(char[],size_t);
EXTERN PetscErrorCode  PetscGetProgramName(char[],size_t);
EXTERN PetscErrorCode  PetscSetProgramName(const char[]);
EXTERN PetscErrorCode  PetscGetDate(char[],size_t);

EXTERN PetscErrorCode  PetscSortInt(PetscInt,PetscInt[]);
EXTERN PetscErrorCode  PetscSortIntWithPermutation(PetscInt,const PetscInt[],PetscInt[]);
EXTERN PetscErrorCode  PetscSortStrWithPermutation(PetscInt,const char*[],PetscInt[]);
EXTERN PetscErrorCode  PetscSortIntWithArray(PetscInt,PetscInt[],PetscInt[]);
EXTERN PetscErrorCode  PetscSortIntWithScalarArray(PetscInt,PetscInt[],PetscScalar[]);
EXTERN PetscErrorCode  PetscSortReal(PetscInt,PetscReal[]);
EXTERN PetscErrorCode  PetscSortRealWithPermutation(PetscInt,const PetscReal[],PetscInt[]);

EXTERN PetscErrorCode  PetscSetDisplay(void);
EXTERN PetscErrorCode  PetscGetDisplay(char[],size_t);

extern PetscCookie PETSC_RANDOM_COOKIE;

typedef enum { RANDOM_DEFAULT,RANDOM_DEFAULT_REAL,
               RANDOM_DEFAULT_IMAGINARY } PetscRandomType;

/*S
     PetscRandom - Abstract PETSc object that manages generating random numbers

   Level: intermediate

  Concepts: random numbers

.seealso:  PetscRandomCreate(), PetscRandomGetValue()
S*/
typedef struct _p_PetscRandom*   PetscRandom;

EXTERN PetscErrorCode PetscRandomCreate(MPI_Comm,PetscRandomType,PetscRandom*);
EXTERN PetscErrorCode PetscRandomGetValue(PetscRandom,PetscScalar*);
EXTERN PetscErrorCode PetscRandomSetInterval(PetscRandom,PetscScalar,PetscScalar);
EXTERN PetscErrorCode PetscRandomDestroy(PetscRandom);

EXTERN PetscErrorCode PetscGetFullPath(const char[],char[],size_t);
EXTERN PetscErrorCode PetscGetRelativePath(const char[],char[],size_t);
EXTERN PetscErrorCode PetscGetWorkingDirectory(char[],size_t);
EXTERN PetscErrorCode PetscGetRealPath(char[],char[]);
EXTERN PetscErrorCode PetscGetHomeDirectory(char[],size_t);
EXTERN PetscErrorCode PetscTestFile(const char[],char,PetscTruth*);
EXTERN PetscErrorCode PetscTestDirectory(const char[],char,PetscTruth*);
EXTERN PetscErrorCode PetscBinaryRead(int,void*,PetscInt,PetscDataType);
EXTERN PetscErrorCode PetscSynchronizedBinaryRead(MPI_Comm,int,void*,PetscInt,PetscDataType);
EXTERN PetscErrorCode PetscBinaryWrite(int,void*,PetscInt,PetscDataType,PetscTruth);
EXTERN PetscErrorCode PetscBinaryOpen(const char[],int,int *);
EXTERN PetscErrorCode PetscBinaryClose(int);
EXTERN PetscErrorCode PetscSharedTmp(MPI_Comm,PetscTruth *);
EXTERN PetscErrorCode PetscSharedWorkingDirectory(MPI_Comm,PetscTruth *);
EXTERN PetscErrorCode PetscGetTmp(MPI_Comm,char *,size_t);
EXTERN PetscErrorCode PetscFileRetrieve(MPI_Comm,const char *,char *,size_t,PetscTruth*);
EXTERN PetscErrorCode PetscLs(MPI_Comm,const char[],char*,size_t,PetscTruth*);
EXTERN PetscErrorCode PetscDLLibraryCCAAppend(MPI_Comm,PetscDLLibraryList*,const char[]);

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
EXTERN PetscErrorCode PetscBinarySeek(int,off_t,PetscBinarySeekType,off_t*);
EXTERN PetscErrorCode PetscSynchronizedBinarySeek(MPI_Comm,int,off_t,PetscBinarySeekType,off_t*);

EXTERN PetscErrorCode PetscSetDebugger(const char[],PetscTruth);
EXTERN PetscErrorCode PetscSetDefaultDebugger(void);
EXTERN PetscErrorCode PetscSetDebuggerFromString(char*);
EXTERN PetscErrorCode PetscAttachDebugger(void);
EXTERN PetscErrorCode PetscStopForDebugger(void);

EXTERN PetscErrorCode PetscGatherNumberOfMessages(MPI_Comm,PetscMPIInt*,PetscMPIInt*,PetscMPIInt*);
EXTERN PetscErrorCode PetscGatherMessageLengths(MPI_Comm,PetscMPIInt,PetscMPIInt,PetscMPIInt*,PetscMPIInt**,PetscMPIInt**);
EXTERN PetscErrorCode PetscGatherMessageLengths2(MPI_Comm,PetscMPIInt,PetscMPIInt,PetscMPIInt*,PetscMPIInt*,PetscMPIInt**,PetscMPIInt**,PetscMPIInt**);
EXTERN PetscErrorCode PetscPostIrecvInt(MPI_Comm,PetscMPIInt,PetscMPIInt,PetscMPIInt*,PetscMPIInt*,PetscInt***,MPI_Request**);
EXTERN PetscErrorCode PetscPostIrecvScalar(MPI_Comm,PetscMPIInt,PetscMPIInt,PetscMPIInt*,PetscMPIInt*,PetscScalar***,MPI_Request**);

EXTERN PetscErrorCode PetscSSEIsEnabled(MPI_Comm,PetscTruth *,PetscTruth *);

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

/* 
  Create and initialize a linked list 
  Input Parameters:
    idx_start - starting index of the list
    lnk_max   - max value of lnk indicating the end of the list
    nlnk      - max length of the list
  Output Parameters:
    lnk       - list initialized
    bt        - PetscBT (bitarray) with all bits set to false
*/
#define PetscLLCreate(idx_start,lnk_max,nlnk,lnk,bt) \
  (PetscMalloc(nlnk*sizeof(PetscInt),&lnk) || PetscBTCreate(nlnk,bt) || PetscBTMemzero(nlnk,bt) || (lnk[idx_start] = lnk_max,0))

/*
  Add a index set into a sorted linked list
  Input Parameters:
    nidx      - number of input indices
    indices   - interger array
    idx_start - starting index of the list
    lnk       - linked list(an integer array) that is created
    bt        - PetscBT (bitarray), bt[idx]=true marks idx is in lnk
  output Parameters:
    nlnk      - number of newly added indices
    lnk       - the sorted(increasing order) linked list containing new and non-redundate entries from indices
    bt        - updated PetscBT (bitarray) 
*/
#define PetscLLAdd(nidx,indices,idx_start,nlnk,lnk,bt) 0;\
{\
  int _k,_entry,_location,_lnkdata;\
  nlnk     = 0;\
  _lnkdata = idx_start;\
  for (_k=0; _k<nidx; _k++){\
    _entry = indices[_k];\
    if (!PetscBTLookupSet(bt,_entry)){  /* new entry */\
      /* search for insertion location */\
      /* start from the beginning if _entry < previous _entry */\
      if (_k && _entry < _lnkdata) _lnkdata  = idx_start;\
      do {\
        _location = _lnkdata;\
        _lnkdata  = lnk[_location];\
      } while (_entry > _lnkdata);\
      /* insertion location is found, add entry into lnk */\
      lnk[_location] = _entry;\
      lnk[_entry]    = _lnkdata;\
      nlnk++;\
      _lnkdata = _entry; /* next search starts from here if next_entry > _entry */\
    }\
  }\
}

/*
  Add a SORTED index set into a sorted linked list
  Input Parameters:
    nidx      - number of input indices
    indices   - sorted interger array 
    idx_start - starting index of the list
    lnk       - linked list(an integer array) that is created
    bt        - PetscBT (bitarray), bt[idx]=true marks idx is in lnk
  output Parameters:
    nlnk      - number of newly added indices
    lnk       - the sorted(increasing order) linked list containing new and non-redundate entries from indices
    bt        - updated PetscBT (bitarray) 
*/
#define PetscLLAddSorted(nidx,indices,idx_start,nlnk,lnk,bt) 0;\
{\
  int _k,_entry,_location,_lnkdata;\
  nlnk      = 0;\
  _lnkdata  = idx_start;\
  for (_k=0; _k<nidx; _k++){\
    _entry = indices[_k];\
    if (!PetscBTLookupSet(bt,_entry)){  /* new entry */\
      /* search for insertion location */\
      do {\
        _location = _lnkdata;\
        _lnkdata  = lnk[_location];\
      } while (_entry > _lnkdata);\
      /* insertion location is found, add entry into lnk */\
      lnk[_location] = _entry;\
      lnk[_entry]    = _lnkdata;\
      nlnk++;\
      _lnkdata = _entry; /* next search starts from here */\
    }\
  }\
}

/*
  Add a SORTED index set into a sorted linked list used for LUFactorSymbolic()
  Same as PetscLLAddSorted() with an additional operation:
       count the number of input indices that are no larger than 'diag'
  Input Parameters:
    nidx      - number of input indices
    indices   - sorted interger array 
    idx_start - starting index of the list
    lnk       - linked list(an integer array) that is created
    bt        - PetscBT (bitarray), bt[idx]=true marks idx is in lnk
    diag      - index of the active row in LUFactorSymbolic
  output Parameters:
    nlnk      - number of newly added indices
    lnk       - the sorted(increasing order) linked list containing new and non-redundate entries from indices
    bt        - updated PetscBT (bitarray) 
    nzbd      - number of input indices that are no larger than 'diag'
*/
#define PetscLLAddSortedLU(nidx,indices,idx_start,nlnk,lnk,bt,diag,nzbd) 0;\
{\
  int _k,_entry,_location,_lnkdata;\
  nlnk     = 0;\
  _lnkdata = idx_start;\
  nzbd     = 0;\
  for (_k=0; _k<nidx; _k++){\
    _entry = indices[_k];\
    if (_entry <= diag) nzbd++;\
    if (!PetscBTLookupSet(bt,_entry)){  /* new entry */\
      /* search for insertion location */\
      do {\
        _location = _lnkdata;\
        _lnkdata  = lnk[_location];\
      } while (_entry > _lnkdata);\
      /* insertion location is found, add entry into lnk */\
      lnk[_location] = _entry;\
      lnk[_entry]    = _lnkdata;\
      nlnk++;\
      _lnkdata = _entry; /* next search starts from here */\
    }\
  }\
}

/*
  Copy data on the list into an array, then initialize the list 
  Input Parameters:
    idx_start - starting index of the list 
    lnk_max   - max value of lnk indicating the end of the list 
    nlnk      - number of data on the list to be copied
    lnk       - linked list
    bt        - PetscBT (bitarray), bt[idx]=true marks idx is in lnk
  output Parameters:
    indices   - array that contains the copied data
    lnk       - linked list that is cleaned and initialize
    bt        - PetscBT (bitarray) with all bits set to false
*/
#define PetscLLClean(idx_start,lnk_max,nlnk,lnk,indices,bt) 0;\
{\
  int _j,_idx=idx_start;\
  for (_j=0; _j<nlnk; _j++){\
    _idx = lnk[_idx];\
    *(indices+_j) = _idx;\
    PetscBTClear(bt,_idx);\
  }\
  lnk[idx_start] = lnk_max;\
}
/*
  Free memories used by the list
*/
#define PetscLLDestroy(lnk,bt) (PetscFree(lnk) || PetscBTDestroy(bt))

/* Routines below are used for incomplete matrix factorization */
/* 
  Create and initialize a linked list and its levels
  Input Parameters:
    idx_start - starting index of the list
    lnk_max   - max value of lnk indicating the end of the list
    nlnk      - max length of the list
  Output Parameters:
    lnk       - list initialized
    lnk_lvl   - array of size nlnk for storing levels of lnk
    bt        - PetscBT (bitarray) with all bits set to false
*/
#define PetscIncompleteLLCreate(idx_start,lnk_max,nlnk,lnk,lnk_lvl,bt)\
  (PetscMalloc(2*nlnk*sizeof(PetscInt),&lnk) || PetscBTCreate(nlnk,bt) || PetscBTMemzero(nlnk,bt) || (lnk[idx_start] = lnk_max,lnk_lvl = lnk + nlnk,0))

/*
  Add a index set into a sorted linked list
  Input Parameters:
    nidx      - number of input indices
    indices   - interger array used for storing column indices
    level     - level of fill, e.g., ICC(level)
    indices_lvl - level of indices 
    idx_start - starting index of the list
    lnk       - linked list(an integer array) that is created
    lnk_lvl   - levels of lnk
    bt        - PetscBT (bitarray), bt[idx]=true marks idx is in lnk
  output Parameters:
    nlnk      - number of newly added indices
    lnk       - the sorted(increasing order) linked list containing new and non-redundate entries from indices
    lnk_lvl   - levels of lnk
    bt        - updated PetscBT (bitarray) 
*/
#define PetscIncompleteLLAdd(nidx,indices,level,indices_lvl,idx_start,nlnk,lnk,lnk_lvl,bt) 0;\
{\
  int _k,_entry,_location,_lnkdata,_incrlev;\
  nlnk     = 0;\
  _lnkdata = idx_start;\
  for (_k=0; _k<nidx; _k++){\
    _incrlev = indices_lvl[_k] + 1;\
    if (_incrlev > level) {_k++; continue;} \
    _entry = indices[_k];\
    if (!PetscBTLookupSet(bt,_entry)){  /* new entry */\
      /* search for insertion location */\
      if (_k && _entry < _lnkdata) _lnkdata  = idx_start;\
      do {\
        _location = _lnkdata;\
        _lnkdata  = lnk[_location];\
      } while (_entry > _lnkdata);\
      /* insertion location is found, add entry into lnk */\
      lnk[_location]  = _entry;\
      lnk[_entry]     = _lnkdata;\
      lnk_lvl[_entry] = _incrlev;\
      nlnk++;\
      _lnkdata = _entry; /* next search starts from here if next_entry > _entry */\
    } else { /* existing entry: update lnk_lvl */\
      if (lnk_lvl[_entry] > _incrlev) lnk_lvl[_entry] = _incrlev;\
    }\
  }\
}

/*
  Add a SORTED index set into a sorted linked list
  Input Parameters:
    nidx      - number of input indices
    indices   - sorted interger array used for storing column indices
    level     - level of fill, e.g., ICC(level)
    indices_lvl - level of indices 
    idx_start - starting index of the list
    lnk       - linked list(an integer array) that is created
    lnk_lvl   - levels of lnk
    bt        - PetscBT (bitarray), bt[idx]=true marks idx is in lnk
  output Parameters:
    nlnk      - number of newly added indices
    lnk       - the sorted(increasing order) linked list containing new and non-redundate entries from indices
    lnk_lvl   - levels of lnk
    bt        - updated PetscBT (bitarray) 
*/
#define PetscIncompleteLLAddSorted(nidx,indices,level,indices_lvl,idx_start,nlnk,lnk,lnk_lvl,bt) 0;\
{\
  int _k,_entry,_location,_lnkdata,_incrlev;\
  nlnk = 0;\
  _lnkdata = idx_start;\
  for (_k=0; _k<nidx; _k++){\
    _incrlev = indices_lvl[_k] + 1;\
    if (_incrlev > level) {_k++; continue;} \
    _entry = indices[_k];\
    if (!PetscBTLookupSet(bt,_entry)){  /* new entry */\
      /* search for insertion location */\
      do {\
        _location = _lnkdata;\
        _lnkdata  = lnk[_location];\
      } while (_entry > _lnkdata);\
      /* insertion location is found, add entry into lnk */\
      lnk[_location] = _entry;\
      lnk[_entry]    = _lnkdata;\
      lnk_lvl[_entry] = _incrlev;\
      nlnk++;\
      _lnkdata = _entry; /* next search starts from here */\
    } else { /* existing entry: update lnk_lvl */\
      if (lnk_lvl[_entry] > _incrlev) lnk_lvl[_entry] = _incrlev;\
    }\
  }\
}

/*
  Copy data on the list into an array, then initialize the list 
  Input Parameters:
    idx_start - starting index of the list 
    lnk_max   - max value of lnk indicating the end of the list 
    nlnk      - number of data on the list to be copied
    lnk       - linked list
    lnk_lvl   - level of lnk
    bt        - PetscBT (bitarray), bt[idx]=true marks idx is in lnk
  output Parameters:
    indices   - array that contains the copied data
    lnk       -llinked list that is cleaned and initialize
    lnk_lvl   - level of lnk that is reinitialized 
    bt        - PetscBT (bitarray) with all bits set to false
*/
#define PetscIncompleteLLClean(idx_start,lnk_max,nlnk,lnk,lnk_lvl,indices,indices_lvl,bt) 0;\
{\
  int _j,_idx=idx_start;\
  for (_j=0; _j<nlnk; _j++){\
    _idx = lnk[_idx];\
    *(indices+_j) = _idx;\
    *(indices_lvl+_j) = lnk_lvl[_idx];\
    lnk_lvl[_idx] = -1;\
    PetscBTClear(bt,_idx);\
  }\
  lnk[idx_start] = lnk_max;\
}
/*
  Free memories used by the list
*/
#define PetscIncompleteLLDestroy(lnk,bt) \
  (PetscFree(lnk) || PetscBTDestroy(bt) || (0))

PETSC_EXTERN_CXX_END
#endif /* __PETSCSYS_H */
